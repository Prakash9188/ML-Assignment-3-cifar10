"""Shared utilities for ML Assignment 3 (data, split, models, training loop).

Introduced here (first used in Q1): MLP / CNN builders, train(), run() cache.
Everything else (q1..q4 scripts) imports from this file.
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
# Roll number -> seed = all numeric characters in order. Override with env ROLL_NO.
ROLL_NO = os.environ.get("ROLL_NO", "M26CSA015")
SEED = int("".join(ch for ch in ROLL_NO if ch.isdigit()))
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def P(*a):
    return os.path.join(ROOT, *a)


def ensure(d):
    os.makedirs(P(d), exist_ok=True)


def save_json(obj, path):
    os.makedirs(os.path.dirname(P(path)), exist_ok=True)
    with open(P(path), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(P(path)) as f:
        return json.load(f)


# ---------------------------------------------------------------- data
def get_data():
    """Returns dict of uint8 NHWC arrays. Stratified 45k/5k split, seed = roll number digits."""
    from sklearn.model_selection import train_test_split
    raw = P("data", "cifar_raw.npz")
    if not os.path.exists(raw):
        import torchvision
        ensure("data")
        tr = torchvision.datasets.CIFAR10(P("data"), train=True, download=True)
        te = torchvision.datasets.CIFAR10(P("data"), train=False, download=True)
        np.savez(raw, xtr=tr.data, ytr=np.array(tr.targets), xte=te.data, yte=np.array(te.targets))
    r = np.load(raw)
    xall, yall = r["xtr"], r["ytr"]
    sp = P("split.npz")
    if not os.path.exists(sp):
        tri, vai = train_test_split(np.arange(len(yall)), test_size=5000, stratify=yall, random_state=SEED)
        np.savez(sp, train_idx=np.sort(tri), val_idx=np.sort(vai), seed=SEED)
    s = np.load(sp)
    ti, vi = s["train_idx"], s["val_idx"]
    return dict(xtr=xall[ti], ytr=yall[ti], xva=xall[vi], yva=yall[vi], xte=r["xte"], yte=r["yte"],
                train_idx=ti, val_idx=vi)


def to01(x):
    """uint8 NHWC -> float32 NCHW in [0,1]."""
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().float().div_(255.0)


def tens(d):
    """All splits as ([0,1] tensors, long labels)."""
    out = {}
    for s in ("tr", "va", "te"):
        out["X" + s] = to01(d["x" + s])
        out["y" + s] = torch.from_numpy(d["y" + s]).long()
    return out


# ---------------------------------------------------------------- transforms (applied identically to train/val/test)
def block_shuffle(X, b, perm):
    """Split image into (32/b)^2 blocks of b x b, apply fixed permutation `perm` of blocks.
    b=1 is a full pixel permutation (colour triplet of each pixel stays together)."""
    N = X.shape[0]
    g = 32 // b
    B = X.reshape(N, 3, g, b, g, b).permute(0, 1, 2, 4, 3, 5).reshape(N, 3, g * g, b, b)
    B = B[:, :, torch.as_tensor(perm)]
    return B.reshape(N, 3, g, g, b, b).permute(0, 1, 2, 4, 3, 5).reshape(N, 3, 32, 32).contiguous()


def make_perm(b, seed=SEED):
    return np.random.RandomState(seed).permutation((32 // b) ** 2)


# ---------------------------------------------------------------- models
class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("m", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.m) / self.s


def mlp(hidden, nc=10):
    layers, d = [Normalize(), nn.Flatten()], 3072
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, nc))
    return nn.Sequential(*layers)


def cnn(w=(24, 48, 96, 96), nc=10):
    layers, cin = [Normalize()], 3
    for i, c in enumerate(w):
        layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()]
        if i < 3:
            layers.append(nn.MaxPool2d(2))
        cin = c
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, nc)]
    return nn.Sequential(*layers)


MODELS = {
    "mlp_matched": lambda: mlp([44, 44]),     # ~ same params as cnn (137.7k vs 136.9k)
    "mlp_big": lambda: mlp([256, 256]),       # ~6x params
    "mlp_mem": lambda: mlp([1024, 1024]),     # Q1b memorisation model
    "cnn": cnn,
}


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- training / eval
@torch.inference_mode()
def predict(model, X, bs=1000):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def evaluate(model, X, y):
    lg = predict(model, X)
    return F.cross_entropy(lg, y).item(), (lg.argmax(1) == y).float().mean().item()


def train(model, X, y, Xva, yva, epochs, lr, bs=256, wd=1e-2, seed=0, cosine=True, callback=None, log=print):
    """AdamW, optional cosine schedule. No augmentation. Returns history dict; model ends at final epoch.
    best_state = state_dict at the epoch with highest validation accuracy."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cosine else None
    h = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[], extra=[], time=[])
    best, best_state = -1, None
    n, t0 = len(X), time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tl = tc = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = model(X[idx])
            loss = F.cross_entropy(out, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == y[idx]).sum().item()
        if sched:
            sched.step()
        vl, va = evaluate(model, Xva, yva)
        h["train_loss"].append(tl / n); h["train_acc"].append(tc / n)
        h["val_loss"].append(vl); h["val_acc"].append(va)
        h["time"].append(time.time() - t0)
        if callback:
            h["extra"].append(callback(model))
        if va > best:
            best = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            h["best_epoch"] = ep + 1
        log(f"  ep {ep + 1:3d}/{epochs} train {tl / n:.3f}/{tc / n:.3f} val {vl:.3f}/{va:.3f} [{time.time() - t0:.0f}s]")
    return h, best_state


def run(name, make_model, X, y, Xva, yva, seed=0, **kw):
    """Cached training run. Saves results/runs/<name>.json and ckpt/<name>.pt (final) + _best.pt."""
    ensure("results/runs"); ensure("ckpt"); ensure("logs")
    jf, cf = P("results/runs", name + ".json"), P("ckpt", name + ".pt")
    torch.manual_seed(seed)
    model = make_model()
    if os.path.exists(jf) and os.path.exists(cf):
        model.load_state_dict(torch.load(cf))
        return model, load_json("results/runs/" + name + ".json")
    print(f"[run] {name}  params={nparams(model)}  kw={ {k: v for k, v in kw.items() if k != 'callback'} }", flush=True)
    h, best = train(model, X, y, Xva, yva, seed=seed, **kw)
    h["params"] = nparams(model)
    h["config"] = {k: v for k, v in kw.items() if k != "callback"}
    h["seed"] = seed
    save_json(h, "results/runs/" + name + ".json")
    torch.save(model.state_dict(), cf)
    torch.save(best, P("ckpt", name + "_best.pt"))
    return model, h


def calib_stats(probs, y, nbins=15):
    """probs: (N,10) numpy, y: (N,). accuracy, NLL, mean confidence (correct/wrong), ECE, per-class acc."""
    probs = np.asarray(probs, dtype=np.float64); y = np.asarray(y)
    conf, pred = probs.max(1), probs.argmax(1)
    ok = pred == y
    ece, edges = 0.0, np.linspace(0, 1, nbins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return dict(acc=float(ok.mean()), nll=float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()),
                conf=float(conf.mean()), conf_correct=float(conf[ok].mean()) if ok.any() else None,
                conf_wrong=float(conf[~ok].mean()) if (~ok).any() else None, ece=float(ece),
                per_class=[float(ok[y == c].mean()) for c in range(10)])
