"""Q3. Convolutional autoencoders, denoising, and linear probing of frozen encoders.

(a) three AEs differing only in bottleneck dim d in {16, 64, 256}; same arch/training otherwise.
(b) Corruption: additive Gaussian noise sigma=0.2 (pixels in [0,1], clipped) -- a corruption where a
    reconstruction model could plausibly help. DAE: input=corrupted (fresh noise every batch), target=clean,
    no labels. Fixed classifier = Q2 clean CNN (q1a cnn, seed 0, introduced in Q1/Q2).
(c) linear probe (frozen encoder) vs reconstruction MSE.
"""
from common import *
from scipy.stats import spearmanr

EPOCHS, SIGMA, DIMS = 20, 0.2, [16, 64, 256]
d = get_data(); T = tens(d)
Xtr, ytr, Xva, yva, Xte, yte = (T[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))


class AE(nn.Module):
    def __init__(self, dz):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(2048, dz))
        self.dec_fc = nn.Linear(dz, 2048)
        self.dec = nn.Sequential(nn.ReLU(), nn.Unflatten(1, (128, 4, 4)), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        return self.dec(self.dec_fc(self.enc(x)))


def corrupt(x, gen=None):
    return (x + SIGMA * torch.randn(x.shape, generator=gen)).clamp(0, 1)


@torch.inference_mode()
def recon(model, X, noisy=None, bs=1000):
    model.eval()
    src = X if noisy is None else noisy
    return torch.cat([model(src[i:i + bs]) for i in range(0, len(X), bs)])


def mse(a, b):
    return F.mse_loss(a, b).item()


def train_ae(name, dz, denoise, lr=1e-3, bs=128):
    ensure("ckpt"); ensure("results/runs")
    jf, cf = P("results/runs", name + ".json"), P("ckpt", name + ".pt")
    torch.manual_seed(0)
    m = AE(dz)
    if os.path.exists(cf):
        m.load_state_dict(torch.load(cf)); return m, load_json("results/runs/" + name + ".json")
    g = torch.Generator().manual_seed(0)
    gv = torch.Generator().manual_seed(123)
    Vin = corrupt(Xva, gv) if denoise else Xva             # fixed corrupted validation inputs
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    h = dict(train_mse=[], val_mse=[], params=nparams(m), dz=dz, denoise=denoise)
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr), generator=g); tot = 0.0
        for i in range(0, len(Xtr), bs):
            x = Xtr[perm[i:i + bs]]
            inp = corrupt(x, g) if denoise else x
            loss = F.mse_loss(m(inp), x)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(x)
        vm = mse(recon(m, Xva, Vin if denoise else None), Xva)
        h["train_mse"].append(tot / len(Xtr)); h["val_mse"].append(vm)
        print(f"  {name} ep {ep + 1} train {tot / len(Xtr):.5f} val {vm:.5f}", flush=True)
    save_json(h, "results/runs/" + name + ".json"); torch.save(m.state_dict(), cf)
    return m, h


# ------------------------------------------------------------------ (a)
aes, A = {}, {}
for dz in DIMS:
    aes[dz], h = train_ae(f"q3_ae_d{dz}", dz, False)
    A[dz] = dict(params=h["params"], val_mse=h["val_mse"][-1], train_mse=h["train_mse"][-1])
sel = min(DIMS, key=lambda k: A[k]["val_mse"])           # selection on validation reconstruction error only
A["selected_d"] = sel
A["test_mse_selected"] = mse(recon(aes[sel], Xte), Xte)
A["test_mse_all"] = {str(k): mse(recon(aes[k], Xte), Xte) for k in DIMS}   # reported for (c) only, not for selection
A["test_psnr_selected"] = float(-10 * np.log10(A["test_mse_selected"]))
print("(a)", A, flush=True)
save_json(A, "results/q3a_summary.json")

# ------------------------------------------------------------------ (b)
dae, hd = train_ae(f"q3_dae_d{sel}_sigma{SIGMA}", sel, True)
clf = MODELS["cnn"](); clf.load_state_dict(torch.load(P("ckpt", "q1a_cnn_lr%s_s0.pt" % load_json("results/q1a_summary.json")["cnn"]["lr"])))
gt = torch.Generator().manual_seed(2026)
Ete = corrupt(Xte, gt)                                    # fixed corrupted test set
B = {"sigma": SIGMA, "selected_d": sel}
for nm, inp in {"clean": Xte, "corrupted": Ete, "clean_AE(corrupted)": recon(aes[sel], Xte, Ete),
                "DAE(corrupted)": recon(dae, Xte, Ete), "clean_AE(clean)": recon(aes[sel], Xte),
                "DAE(clean)": recon(dae, Xte)}.items():
    pr = F.softmax(predict(clf, inp), 1).numpy()
    B[nm] = calib_stats(pr, yte.numpy()); B[nm]["mse_to_clean"] = mse(inp, Xte)
    print("(b)", nm, B[nm]["acc"], B[nm]["mse_to_clean"], flush=True)
save_json(B, "results/q3b_summary.json")
np.savez(P("results", "q3_examples.npz"), clean=Xte[:16].numpy(), noisy=Ete[:16].numpy(),
         cae=recon(aes[sel], Xte[:16], Ete[:16]).numpy(), dae=recon(dae, Xte[:16], Ete[:16]).numpy(),
         **{f"ae{k}": recon(aes[k], Xte[:16]).numpy() for k in DIMS})

# ------------------------------------------------------------------ (c) linear probe on frozen encoders
@torch.inference_mode()
def feats(model, X, bs=1000):
    model.eval(); return torch.cat([model.encode(X[i:i + bs]) for i in range(0, len(X), bs)])


def probe(Ztr, Zva, Zte):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr, Zva, Zte = (Ztr - mu) / sd, (Zva - mu) / sd, (Zte - mu) / sd
    best = None
    for wd in [1e-4, 1e-3, 1e-2]:
        torch.manual_seed(0)
        lin = nn.Linear(Ztr.shape[1], 10)
        opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=300, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = F.cross_entropy(lin(Ztr), ytr) + wd * (lin.weight ** 2).sum()
            l.backward(); return l
        opt.step(closure)
        with torch.no_grad():
            va = (lin(Zva).argmax(1) == yva).float().mean().item()
            te = (lin(Zte).argmax(1) == yte).float().mean().item()
        if best is None or va > best[0]:
            best = (va, te, wd)
    return dict(val_acc=best[0], test_acc=best[1], wd=best[2])      # wd chosen on validation


C = {}
torch.manual_seed(0)
feat_sets = {f"ae_d{k}": aes[k] for k in DIMS}
feat_sets[f"dae_d{sel}"] = dae
rnd = AE(sel)                                                        # random, untrained encoder baseline
feat_sets[f"random_enc_d{sel}"] = rnd
for nm, mdl in feat_sets.items():
    C[nm] = probe(feats(mdl, Xtr), feats(mdl, Xva), feats(mdl, Xte))
    C[nm]["dim"] = int(feats(mdl, Xva[:2]).shape[1])
    print("(c)", nm, C[nm], flush=True)
C["raw_pixels"] = probe(Xtr.flatten(1), Xva.flatten(1), Xte.flatten(1))
C["test_mse"] = {f"ae_d{k}": A["test_mse_all"][str(k)] for k in DIMS}
C["test_mse"][f"dae_d{sel}"] = mse(recon(dae, Xte, Ete), Xte)         # noisy input -> clean (not comparable 1:1)
C["test_mse"][f"dae_d{sel}_cleaninput"] = mse(recon(dae, Xte), Xte)
rho = spearmanr([C["test_mse"][f"ae_d{k}"] for k in DIMS], [C[f"ae_d{k}"]["test_acc"] for k in DIMS])
C["spearman_mse_vs_probe_acc"] = float(rho.statistic)
print("(c)", C["spearman_mse_vs_probe_acc"], flush=True)
save_json(C, "results/q3c_summary.json")
