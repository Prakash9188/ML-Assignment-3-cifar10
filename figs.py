"""Make all report figures from results/*.json (run after q1..q4)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

ensure("figs")
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.grid": True, "grid.alpha": .3})
S = lambda n: load_json(f"results/{n}.json")
R = lambda n: load_json(f"results/runs/{n}.json")


def save(fig, n):
    fig.tight_layout(); fig.savefig(P("figs", n + ".pdf")); fig.savefig(P("figs", n + ".png")); plt.close(fig)


# ---- Q1a
q1 = S("q1a_summary")
fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.2))
for nm, c in zip(["mlp_matched", "mlp_big", "cnn"], ["C0", "C1", "C3"]):
    h = R(q1[nm]["run"]); e = np.arange(1, len(h["val_acc"]) + 1)
    ax[0].plot(e, h["train_acc"], c + "--", label=f"{nm} train"); ax[0].plot(e, h["val_acc"], c, label=f"{nm} val")
    ax[1].plot(e, h["train_loss"], c + "--"); ax[1].plot(e, h["val_loss"], c)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy"); ax[0].legend(fontsize=6)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("cross-entropy")
save(fig, "q1a_curves")

# ---- Q1b
fig, ax = plt.subplots(1, 3, figsize=(5.2, 2.1), sharey=True)
for a, frac in zip(ax, ["0.0", "0.4", "1.0"]):
    h = R(f"q1b_mem_noise{frac}"); e = np.arange(1, len(h["val_acc"]) + 1)
    a.plot(e, h["val_acc"], "k", label="val acc (clean labels)")
    a.plot(e, [x["clean_fit"] for x in h["extra"]], "C0", label="train fit, untouched labels")
    if frac != "0.0":
        a.plot(e, [x["noisy_fit_assigned"] for x in h["extra"]], "C3", label="noisy samples: predicts assigned wrong label")
        a.plot(e, [x["noisy_pred_true"] for x in h["extra"]], "C2", label="noisy samples: predicts true label")
    a.set_title(f"label noise {float(frac):.0%}"); a.set_xlabel("epoch")
ax[0].set_ylabel("accuracy"); ax[2].legend(*ax[1].get_legend_handles_labels(), fontsize=4.5, loc="upper right")
save(fig, "q1b_memorisation")

# ---- Q2
q2 = S("q2_summary")
conds = ["clean", "block16", "block8", "block4", "perm1"]
fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.2))
w = .38
for k, (nm, c) in enumerate([("mlp_matched", "C0"), ("cnn", "C3")]):
    ax[0].bar(np.arange(5) + (k - .5) * w, [q2[f"{c_}|{nm}"]["test_mean"] for c_ in conds], w,
              yerr=[q2[f"{c_}|{nm}"]["test_std"] for c_ in conds], color=c, label=nm)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(conds, rotation=20); ax[0].set_ylabel("test accuracy"); ax[0].legend()
for nm, c in [("mlp_matched", "C0"), ("cnn", "C3")]:
    for cond, ls in [("clean", "-"), ("perm1", ":")]:
        pr = np.load(P("results", f"q2_probs_{nm}_{cond}.npy")); y = load_json("results/q3a_summary.json") and None
d_ = get_data(); yte = d_["yte"]
for nm, c in [("mlp_matched", "C0"), ("cnn", "C3")]:
    for cond, ls in [("clean", "-"), ("block8", "--"), ("perm1", ":")]:
        pr = np.load(P("results", f"q2_probs_{nm}_{cond}.npy")); conf = pr.max(1); ok = pr.argmax(1) == yte
        bins = np.linspace(.1, 1, 10); xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 20: xs.append(conf[m].mean()); ys.append(ok[m].mean())
        ax[1].plot(xs, ys, c + ls, label=f"{nm} {cond}")
ax[1].plot([0, 1], [0, 1], "k:", lw=.6); ax[1].set_xlabel("confidence"); ax[1].set_ylabel("accuracy"); ax[1].legend(fontsize=5)
save(fig, "q2_results")

# ---- Q3
ex = np.load(P("results", "q3_examples.npz")); keys = ["clean", "ae16", "ae64", "ae256", "noisy", "cae", "dae"]
fig, ax = plt.subplots(len(keys), 8, figsize=(4.2, 3.7))
for r, k in enumerate(keys):
    for c in range(8):
        ax[r, c].imshow(ex[k][c].transpose(1, 2, 0)); ax[r, c].axis("off")
    ax[r, 0].set_title({"cae": "clean-AE(noisy)", "dae": "DAE(noisy)"}.get(k, k), fontsize=6, loc="left")
save(fig, "q3_recon")
c3 = S("q3c_summary"); a3 = S("q3a_summary")
fig, ax = plt.subplots(figsize=(3.2, 2.4))
for k in DIMS if (DIMS := [16, 64, 256]) else []:
    ax.scatter(c3["test_mse"][f"ae_d{k}"], c3[f"ae_d{k}"]["test_acc"]); ax.annotate(f"d={k}", (c3["test_mse"][f"ae_d{k}"], c3[f"ae_d{k}"]["test_acc"]))
ax.set_xlabel("test reconstruction MSE"); ax.set_ylabel("linear-probe test acc")
save(fig, "q3c_mse_vs_probe")

# ---- Q4
if os.path.exists(P("results", "q4_cluster_summary.json")):
    sw = S("q4_sweep")
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.1))
    for kn, c in zip(["linear", "poly3", "rbf"], ["C0", "C2", "C3"]):
        rr = sorted([r for r in sw if r["kernel"] == kn], key=lambda r: r["C"])
        ax[0].semilogx([r["C"] for r in rr], [r["val_acc"] for r in rr], c + "-o", label=kn)
        ax[0].semilogx([r["C"] for r in rr], [r["train_acc"] for r in rr], c + "--", alpha=.6)
        ax[1].semilogx([r["C"] for r in rr], [r["sv_frac"] for r in rr], c + "-o", label=kn)
    ax[0].set_xlabel("C"); ax[0].set_ylabel("val acc (dashed: train)"); ax[0].legend(); ax[1].set_xlabel("C"); ax[1].set_ylabel("support-vector fraction")
    save(fig, "q4_sweep_cluster")
    q4 = S("q4_svm_summary"); pz = np.load(P("results", "q4_pair_images.npz"), allow_pickle=True)
    sub = load_json("results/q4_subset.json")["subset_idx_into_train"]
    fig, ax = plt.subplots(3 * 2, 8, figsize=(4.6, 3.7))
    for r, pr in enumerate(q4["hard_pairs"]):
        i, j = pr["pair"]
        svs = pr["sv_train_idx_in_subset"][str(i)][:4] + pr["sv_train_idx_in_subset"][str(j)][:4]
        for c in range(8):
            ax[2 * r, c].imshow(pz["sub_imgs"][svs[c]] if c < len(svs) else np.zeros((32, 32, 3), np.uint8)); ax[2 * r, c].axis("off")
            ax[2 * r, c].set_title(("SV " + CLASSES[i if c < 4 else j]) if c in (0, 4) else "", fontsize=5, loc="left")
            t = pr["near_test_idx"][c]
            ax[2 * r + 1, c].imshow(pz["test_imgs"][t]); ax[2 * r + 1, c].axis("off")
            okc = "ok" if pr['near_true'][c] == pr['near_pred'][c] else "ERR"
            ax[2 * r + 1, c].set_title(f"{CLASSES[pr['near_true'][c]][:3]} {okc}\n{pr['near_scores'][c]:+.4f}", fontsize=4.2, pad=1.5)
    save(fig, "q4_pairs")
print("figures done")
