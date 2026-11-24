"""Q4 (bonus). SVM sweep, hard class pairs, clustering on a FIXED representation.

Representation: frozen encoder latent of the autoencoder selected in Q3(a) (lowest validation MSE),
standardised with training statistics. Fixed stratified training subset: 1000 images/class (10,000),
drawn with RandomState(SEED) from the 45k train set BEFORE any model selection; used for every SVM.
"""
import time, warnings
from common import *
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, silhouette_score, confusion_matrix
warnings.filterwarnings("ignore")
import sys; sys.argv = sys.argv[:1]
d = get_data(); T = tens(d)
ytr_n, yva_n, yte_n = d["ytr"], d["yva"], d["yte"]

# ---- representation (reuses AE class from q3 without re-running q3: re-declare minimal import)
import importlib.util
src = open(P("q3.py")).read().split("class AE")[1].split("def corrupt")[0]
exec("class AE" + src)
sel = load_json("results/q3a_summary.json")["selected_d"]
ae = AE(sel); ae.load_state_dict(torch.load(P("ckpt", f"q3_ae_d{sel}.pt"))); ae.eval()


@torch.inference_mode()
def feats(X):
    return torch.cat([ae.encode(X[i:i + 1000]) for i in range(0, len(X), 1000)]).numpy()


Ztr, Zva, Zte = feats(T["Xtr"]), feats(T["Xva"]), feats(T["Xte"])
mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
Ztr, Zva, Zte = [(z - mu) / sd for z in (Ztr, Zva, Zte)]

rs = np.random.RandomState(SEED)
sub = np.sort(np.concatenate([rs.choice(np.where(ytr_n == c)[0], 1000, replace=False) for c in range(10)]))
save_json(dict(subset_idx_into_train=sub.tolist(), per_class=1000), "results/q4_subset.json")
Zs, ys = Ztr[sub], ytr_n[sub]

# ------------------------------------------------------------------ (a)
KERNELS = {"linear": dict(kernel="linear"), "poly3": dict(kernel="poly", degree=3, gamma="scale", coef0=1.0),
           "rbf": dict(kernel="rbf", gamma="scale")}
CS = [0.01, 0.1, 1, 10, 100, 1000]
sweep = []
jf = P("results", "q4_sweep.json")
if os.path.exists(jf):
    sweep = load_json("results/q4_sweep.json")
done = {(r["kernel"], r["C"]) for r in sweep}
for kn, kw in KERNELS.items():
    for C in CS:
        if (kn, C) in done:
            continue
        t = time.time()
        clf = SVC(C=C, cache_size=2000, max_iter=300000, **kw).fit(Zs, ys)
        r = dict(kernel=kn, C=C, train_acc=float(clf.score(Zs, ys)), val_acc=float(clf.score(Zva, yva_n)),
                 sv_frac=float(len(clf.support_) / len(ys)), fit_time=time.time() - t)
        sweep.append(r); print(r, flush=True)
        save_json(sweep, "results/q4_sweep.json")
best = max(sweep, key=lambda r: r["val_acc"])         # validation only
print("selected", best, flush=True)
kw = KERNELS[best["kernel"]]
clf = SVC(C=best["C"], cache_size=2000, max_iter=300000, decision_function_shape="ovo", **kw).fit(Zs, ys)
pred = clf.predict(Zte)
cm = confusion_matrix(yte_n, pred)
Q = dict(selected=best, test_acc=float((pred == yte_n).mean()), per_class_recall=(cm.diagonal() / cm.sum(1)).tolist(),
         confusion=cm.tolist())
print("test acc", Q["test_acc"], flush=True)

# ------------------------------------------------------------------ (b) hard pairs
pairs = sorted([(cm[i, j] + cm[j, i], i, j) for i in range(10) for j in range(i + 1, 10)], reverse=True)[:3]
imgs_tr = d["xtr"][sub]
PB = []
for tot, i, j in pairs:
    mtr = np.isin(ys, [i, j]); Zp, yp = Zs[mtr], ys[mtr]
    b = SVC(C=best["C"], **kw).fit(Zp, (yp == j).astype(int))     # same libsvm sub-problem as the OvO pair classifier
    sv_global = np.where(mtr)[0][b.support_]
    alpha = np.abs(b.dual_coef_[0]); bounded = alpha >= best["C"] - 1e-6
    mte = np.isin(yte_n, [i, j]); Zq, yq = Zte[mte], yte_n[mte]
    sc = b.decision_function(Zq)                                   # >0 -> class j
    bp = np.where(sc > 0, j, i); err = bp != yq
    order = np.argsort(np.abs(sc)); k = 50
    ci, cj = Zp[yp == i].mean(0), Zp[yp == j].mean(0)
    within = 0.5 * (np.linalg.norm(Zp[yp == i] - ci, axis=1).mean() + np.linalg.norm(Zp[yp == j] - cj, axis=1).mean())
    te_idx = np.where(mte)[0]
    near = order[:8]
    # support vectors with the largest alpha per class
    svi = {c: sv_global[np.argsort(-alpha)][np.isin(ys[sv_global[np.argsort(-alpha)]], [c])][:6].tolist() for c in (i, j)}
    PB.append(dict(pair=[int(i), int(j)], names=[CLASSES[i], CLASSES[j]], test_pair_errors=int(tot),
                   n_sv=int(len(b.support_)), sv_frac_pair=float(len(b.support_) / len(yp)), bounded_sv_frac=float(bounded.mean()),
                   binary_test_acc=float(1 - err.mean()), err_rate_nearest50=float(err[order[:k]].mean()),
                   err_rate_rest=float(err[order[k:]].mean()), mean_abs_score_err=float(np.abs(sc[err]).mean()),
                   mean_abs_score_correct=float(np.abs(sc[~err]).mean()), centroid_dist=float(np.linalg.norm(ci - cj)),
                   within_spread=float(within), sep_ratio=float(np.linalg.norm(ci - cj) / within),
                   near_test_idx=te_idx[near].tolist(), near_scores=sc[near].tolist(), near_true=yq[near].tolist(),
                   near_pred=bp[near].tolist(), sv_train_idx_in_subset={str(c): v for c, v in svi.items()}))
    print(PB[-1]["names"], {k_: v for k_, v in PB[-1].items() if not k_.startswith(("near", "sv_train"))}, flush=True)
Q["hard_pairs"] = PB
np.savez(P("results", "q4_pair_images.npz"), sub_imgs=imgs_tr, test_imgs=d["xte"][:])   # for figure scripts
save_json(Q, "results/q4_svm_summary.json")

# ------------------------------------------------------------------ (c) clustering (labels never used in fit)
rows, models = [], {}
rs2 = np.random.RandomState(SEED + 1); sil_idx = rs2.choice(len(Ztr), 8000, replace=False)
for method, ks in [("kmeans", [5, 10, 15, 20, 30]), ("gmm_diag", [10, 20])]:
    for k in ks:
        m = KMeans(k, n_init=3, random_state=SEED) if method == "kmeans" else GaussianMixture(k, covariance_type="diag", max_iter=100, random_state=SEED)
        m.fit(Ztr)                                    # fit on training representations only
        ltr, lva = m.predict(Ztr), m.predict(Zva)
        r = dict(method=method, k=k, train_ari=ARI(ytr_n, ltr), train_nmi=NMI(ytr_n, ltr), val_ari=ARI(yva_n, lva),
                 val_nmi=NMI(yva_n, lva), train_silhouette=float(silhouette_score(Ztr[sil_idx], ltr[sil_idx])))
        rows.append(r); models[(method, k)] = m; print(r, flush=True)
bc = max(rows, key=lambda r: r["val_ari"])           # chosen by validation ARI (labels used only for selection, never for fitting)
m = models[(bc["method"], bc["k"])]
ltr, lte = m.predict(Ztr), m.predict(Zte)
K = bc["k"]
cont = np.zeros((K, 10), int)
np.add.at(cont, (lte, yte_n), 1)
# cluster -> class mapping from TRAIN labels (majority), then evaluated on test
cmap = np.array([np.bincount(ytr_n[ltr == c], minlength=10).argmax() if (ltr == c).any() else 0 for c in range(K)])
cpred = cmap[lte]
clu_recall = [float((cpred[yte_n == c] == c).mean()) for c in range(10)]
svm_recall = Q["per_class_recall"]
same = {}
for (a, b2) in [(p_[1], p_[2]) for p_ in pairs]:
    # fraction of test (a,b) cross-pairs sharing a cluster vs SVM confusion rate
    ca, cb = np.bincount(lte[yte_n == a], minlength=K) / (yte_n == a).sum(), np.bincount(lte[yte_n == b2], minlength=K) / (yte_n == b2).sum()
    same[f"{CLASSES[a]}-{CLASSES[b2]}"] = dict(cluster_overlap=float(np.minimum(ca, cb).sum()),
                                              svm_confusion_rate=float((cm[a, b2] + cm[b2, a]) / (cm[a].sum() + cm[b2].sum())))
C_ = dict(sweep=rows, chosen=bc, test_ari=float(ARI(yte_n, lte)), test_nmi=float(NMI(yte_n, lte)),
          contingency=cont.tolist(), cluster_to_class=cmap.tolist(), cluster_mapped_test_acc=float((cpred == yte_n).mean()),
          cluster_recall=clu_recall, svm_recall=svm_recall, recall_gap=[s - c for s, c in zip(svm_recall, clu_recall)],
          pair_overlap=same, svm_test_acc=Q["test_acc"])
print("test ARI/NMI", C_["test_ari"], C_["test_nmi"], "cluster-mapped acc", C_["cluster_mapped_test_acc"], flush=True)
save_json(C_, "results/q4_cluster_summary.json")
