"""Q1. (a) inductive bias vs capacity/optimisation; (b) memorisation with noisy labels.

Hypothesis tested in (a): the MLP is poor on CIFAR10 mainly because it lacks the locality / weight-sharing
inductive bias, NOT because it has too few parameters or is under-trained.
Fair comparison: identical data, epochs, optimiser, batch size, no augmentation; lr picked per model from the
same grid on the validation set; (i) cnn vs mlp_matched have matched parameters (~137k);
(ii) mlp_big has ~6x more parameters than the CNN to test the capacity explanation.
"""
from common import *

EPOCHS, LRS = 10, [1e-3, 3e-3]
d = get_data(); T = tens(d)
Xtr, ytr, Xva, yva, Xte, yte = (T[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))

# ------------------------------------------------------------------ (a)
summary = {}
for name in ["mlp_matched", "mlp_big", "cnn"]:
    res = {}
    for lr in LRS:
        m, h = run(f"q1a_{name}_lr{lr}_s0", MODELS[name], Xtr, ytr, Xva, yva, seed=0, epochs=EPOCHS, lr=lr, bs=256)
        res[lr] = (h["val_acc"][-1], m, h)
    lr_best = max(res, key=lambda k: res[k][0])         # selection on validation only
    m, h = res[lr_best][1], res[lr_best][2]
    tl, ta = evaluate(m, Xte, yte)                      # test only after selection
    summary[name] = dict(params=h["params"], lr=lr_best, val_acc=h["val_acc"][-1], train_acc=h["train_acc"][-1],
                         train_loss=h["train_loss"][-1], val_loss=h["val_loss"][-1], test_acc=ta, test_loss=tl,
                         run=f"q1a_{name}_lr{lr_best}_s0", grid={str(k): v[0] for k, v in res.items()})
    print(name, summary[name], flush=True)
save_json(summary, "results/q1a_summary.json")

# ------------------------------------------------------------------ (b)
def make_noisy(frac, seed=1):
    rs = np.random.RandomState(seed)
    y = ytr.clone().numpy()
    n = len(y)
    idx = rs.permutation(n)[: int(frac * n)]
    y[idx] = rs.randint(0, 10, size=len(idx))        # uniformly random label
    changed = np.zeros(n, bool); changed[idx] = y[idx] != ytr.numpy()[idx]
    return torch.from_numpy(y).long(), torch.from_numpy(changed)


EPOCHS_B = 60
out = {}
for frac in [0.0, 0.4, 1.0]:
    ynoisy, changed = make_noisy(frac)

    def cb(model, ynoisy=ynoisy, changed=changed):
        pred = predict(model, Xtr).argmax(1)
        r = dict(clean_fit=(pred[~changed] == ynoisy[~changed]).float().mean().item())
        if changed.any():
            r["noisy_fit_assigned"] = (pred[changed] == ynoisy[changed]).float().mean().item()   # memorised the wrong label
            r["noisy_pred_true"] = (pred[changed] == ytr[changed]).float().mean().item()          # still predicts true class
        return r

    m, h = run(f"q1b_mem_noise{frac}", MODELS["mlp_mem"], Xtr, ynoisy, Xva, yva, seed=0, epochs=EPOCHS_B, lr=1e-3,
               bs=128, wd=0.0, cosine=False, callback=cb)
    tl, ta = evaluate(m, Xte, yte)
    mb = MODELS["mlp_mem"](); mb.load_state_dict(torch.load(P("ckpt", f"q1b_mem_noise{frac}_best.pt")))
    _, ta_b = evaluate(mb, Xte, yte)
    out[str(frac)] = dict(params=h["params"], test_acc_final=ta, test_acc_at_best_val_epoch=ta_b,
                          best_val_epoch=h["best_epoch"], val_acc_final=h["val_acc"][-1], val_acc_best=max(h["val_acc"]),
                          n_noisy=int(changed.sum()))
    print(frac, out[str(frac)], flush=True)
save_json(out, "results/q1b_summary.json")
