"""Q2. CNN vs MLP, spatial disturbance, and a block-shuffle scale experiment.

Fairness criterion (fixed BEFORE looking at results): (1) parameter counts within 1% (cnn 136.9k vs mlp_matched),
(2) same data, epochs (10), optimiser (AdamW, wd 1e-2, cosine), batch size, no augmentation,
(3) lr chosen per model and per condition from the same grid {1e-3, 3e-3} by validation accuracy,
(4) 3 seeds (0,1,2) for clean / pixel-permuted; seed 0 only for CNN block-shuffle (cost).
Reused from Q1: models mlp_matched & cnn (clean condition, seed 0 runs q1a_*).

Explanation proposed for (a): the CNN wins because its locality + weight sharing exploit 2-D spatial
neighbourhood structure; the MLP must learn each pixel's role separately.
EXPECTATIONS (written before running):
 - clean: CNN >> MLP (gap >= 20 points).
 - fixed random pixel permutation (same for train/val/test): MLP unchanged (within seed noise, +-1 pt);
   CNN drops a lot, to roughly MLP level or below. If so, spatial structure is what the CNN exploits.
 - block shuffle (fixed block permutation): MLP unchanged; CNN degrades monotonically as blocks get smaller
   (b=16 < b=8 < b=4 < b=1 loss). Two competing explanations for the CNN behaviour tested here:
   H1 'CNN mainly uses small local features/textures' -> block-8 keeps most of the accuracy;
   H2 'CNN needs global arrangement/shape' -> block-8 collapses to near the pixel-permuted level.
   We expect H1 partly: block-8 accuracy clearly above permuted but below clean.
"""
from common import *

EPOCHS, LRS = 10, [1e-3, 3e-3]
d = get_data(); T = tens(d)
Xtr, ytr, Xva, yva, Xte, yte = (T[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))
q1 = load_json("results/q1a_summary.json")
save_json(dict(expectations=__doc__), "results/q2_expectations.json")

CONDS = {"clean": None, "perm1": 1, "block4": 4, "block8": 8, "block16": 16}
ENT = {}   # (model,cond) -> dict


def tf(X, b):
    return X if b is None else block_shuffle(X, b, make_perm(b))


results = {}
for cond, b in CONDS.items():
    A, V, E = tf(Xtr, b), tf(Xva, b), tf(Xte, b)       # identical transform for train/val/test
    for name in ["mlp_matched", "cnn"]:
        seeds = [0, 1, 2] if (cond in ("clean", "perm1") or name == "mlp_matched") else [0]
        # --- lr selection on validation, seed 0
        if cond == "clean":
            lr = q1[name]["lr"]                           # reuse Q1 selection and its seed-0 run
            names0 = f"q1a_{name}_lr{lr}_s0"
            runs = {0: names0}
        elif cond == "perm1" or name == "mlp_matched":
            vals = {}
            for l in LRS:
                _, h = run(f"q2_{name}_{cond}_lr{l}_s0", MODELS[name], A, ytr, V, yva, seed=0, epochs=EPOCHS, lr=l)
                vals[l] = h["val_acc"][-1]
            lr = max(vals, key=vals.get)
            runs = {0: f"q2_{name}_{cond}_lr{lr}_s0"}
        else:
            lr = results[("clean", name)]["lr"]           # CNN block conditions: lr from clean (cost)
            runs = {0: f"q2_{name}_{cond}_lr{lr}_s0"}
        accs, vaccs, probs0 = [], [], None
        for s in seeds:
            nm = runs.get(s, f"q2_{name}_{cond}_lr{lr}_s{s}")
            m, h = run(nm, MODELS[name], A, ytr, V, yva, seed=s, epochs=EPOCHS, lr=lr)
            lg = predict(m, E)
            accs.append((lg.argmax(1) == yte).float().mean().item())
            vaccs.append(h["val_acc"][-1])
            if s == 0:
                probs0 = F.softmax(lg, 1).numpy()
                tr0 = h
        np.save(P("results", f"q2_probs_{name}_{cond}.npy"), probs0.astype(np.float32))
        cs = calib_stats(probs0, yte.numpy())
        results[(cond, name)] = dict(lr=lr, params=tr0["params"], test_acc=accs, test_mean=float(np.mean(accs)),
                                     test_std=float(np.std(accs)), val_acc=vaccs, train_acc_final=tr0["train_acc"][-1],
                                     calib=cs, seeds=seeds)
        print(cond, name, {k: v for k, v in results[(cond, name)].items() if k != "calib"}, flush=True)

save_json({f"{c}|{n}": v for (c, n), v in results.items()}, "results/q2_summary.json")
