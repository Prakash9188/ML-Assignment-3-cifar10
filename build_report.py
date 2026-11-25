"""Builds report/report.pdf (two-column, CVPR-like layout) from results/*.json and figs/*.png."""
from common import *
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table, TableStyle,
                                FrameBreak, NextPageTemplate, KeepTogether)
from PIL import Image as PILImage

ensure("report")
L = lambda n: load_json(f"results/{n}.json")
q1a, q1b, q2, q3a, q3b, q3c = L("q1a_summary"), L("q1b_summary"), L("q2_summary"), L("q3a_summary"), L("q3b_summary"), L("q3c_summary")
q4s, q4c, sw = L("q4_svm_summary"), L("q4_cluster_summary"), L("q4_sweep")
RUN = lambda n: L(f"runs/{n}")

W, H = letter; M = 0.75 * inch; GAP = 0.25 * inch; CW = (W - 2 * M - GAP) / 2
body = ParagraphStyle("b", fontName="Times-Roman", fontSize=9, leading=10.4, alignment=TA_JUSTIFY, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=10.5, leading=12, spaceBefore=6, spaceAfter=3)
h2 = ParagraphStyle("h2", fontName="Times-BoldItalic", fontSize=9.2, leading=11, spaceBefore=3, spaceAfter=2)
cap = ParagraphStyle("c", fontName="Times-Roman", fontSize=7.8, leading=9, alignment=TA_JUSTIFY, spaceAfter=5)
ttl = ParagraphStyle("t", fontName="Times-Bold", fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle("s", fontName="Times-Roman", fontSize=9.5, leading=11.5, alignment=TA_CENTER)
tc = ParagraphStyle("tc", fontName="Times-Roman", fontSize=6.6, leading=7.4)
P_ = lambda t, s=body: Paragraph(t, s)
story = []


def fig(name, caption, w=CW):
    im = PILImage.open(P("figs", name + ".png")); r = im.size[1] / im.size[0]
    story.append(KeepTogether([Image(P("figs", name + ".png"), width=w, height=w * r), P_(caption, cap)]))


def table(rows, caption, colw=None, fs=6.6):
    data = [[Paragraph(str(c), ParagraphStyle("x", parent=tc, fontSize=fs, leading=fs + .8)) for c in r] for r in rows]
    t = Table(data, colWidths=colw, repeatRows=1)
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), .7, colors.black), ("LINEBELOW", (0, 0), (-1, 0), .4, colors.black),
                           ("LINEBELOW", (0, -1), (-1, -1), .7, colors.black), ("TOPPADDING", (0, 0), (-1, -1), 1),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story.append(KeepTogether([t, Spacer(1, 2), P_(caption, cap)]))


pc = lambda x: f"{100 * x:.1f}"
ms = lambda r: f"{100 * r['test_mean']:.1f} ± {100 * r['test_std']:.1f}"

# ------------------------------------------------------------------ title
story += [P_("Machine Learning Assignment 3: MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR10", ttl),
          P_("<b>Prakash Singh</b> &nbsp;·&nbsp; Roll no. <b>M26CSA015</b> &nbsp;·&nbsp; Seed = 26015", sub),
          P_("Repository: github.com/Prakash9188/ML-Assignment-3-cifar10 &nbsp;·&nbsp; Final commit: 9e46585", sub),
          Spacer(1, 4), FrameBreak()]

story += [P_("Abstract", h1), P_(
    "We study CIFAR10 with MLPs, CNNs, convolutional autoencoders, SVMs and clustering under a CPU-only budget "
    "(10 training epochs, small models, no augmentation). Main findings: (i) at matched parameters a CNN beats an MLP by 27.8 points "
    "(78.7 vs 50.9 test), and a 6x larger MLP closes only 5 of those points; (ii) a fixed pixel permutation leaves the MLP unchanged "
    "(51.1) but drops the CNN to 50.9, i.e. the CNN advantage is entirely spatial structure; block-shuffling shows about two thirds of it is "
    "recoverable from 8x8 local structure; (iii) denoising lowers pixel error but does not recover what the classifier needs "
    "(44.3 vs 43.6 for the plain AE, 78.7 clean), and reconstruction error and linear-probe accuracy agree across bottleneck sizes but "
    "not between the clean and denoising AE; (iv) on the AE latent (a weak representation) the best SVM reaches 49.7 test and clustering "
    "gives ARI 0.056. All numbers below are read from <font face='Courier'>results/*.json</font> written by the scripts.")]

story += [P_("1. Setup and reuse", h1), P_(
    "<b>Data.</b> CIFAR10; the 50,000 training images are split once into 45,000 train / 5,000 validation, stratified "
    "(<font face='Courier'>train_test_split</font>, <font face='Courier'>random_state</font>=26015, the digits of M26CSA015, saved in <font face='Courier'>split.npz</font>). "
    "Validation is used for every selection (learning rate, bottleneck, C, kernel, clustering model). The official test set is evaluated only after the "
    "choice is fixed; all &quot;final&quot; numbers are test numbers. Inputs are scaled to [0,1]; models normalise internally. "
    "No pretrained models or external data. <b>Training protocol</b> (all classifiers): AdamW, weight decay 0.01, cosine schedule, batch 256, "
    "10 epochs, no augmentation, learning rate from {1e-3, 3e-3} picked on validation accuracy at the final epoch. "
    "<b>Compute note.</b> The machine has no GPU, so budgets are small; accuracies are therefore lower than published CIFAR10 numbers and "
    "epoch counts are the same for all models in a comparison.")]
story += [P_(
    "<b>Reuse map.</b> <i>Introduced in Q1:</i> <font face='Courier'>common.py</font> (split, training loop, calibration metrics), CNN "
    "(4 conv-BN-ReLU, widths 24/48/96/96, 3 max-pools, global-average-pool, linear; 136,930 params), MLP-matched (3072-44-44-10, 137,642), MLP-big "
    "(3072-256-256-10, 855,050), MLP-mem (3072-1024-1024-10, 4.2M). <i>Reused in Q2:</i> CNN and MLP-matched seed-0 clean runs from Q1(a). "
    "<i>Reused in Q3:</i> the Q1/Q2 clean seed-0 CNN is the fixed classifier. <i>Reused in Q4:</i> the encoder of the Q3(a)-selected AE is the fixed "
    "representation.")]

# ------------------------------------------------------------------ Q1
story += [P_("2. Q1: Why does an MLP struggle?", h1), P_("2.1 Inductive bias versus capacity (Q1a)", h2), P_(
    "<b>Reason chosen.</b> The MLP lacks the locality / weight-sharing inductive bias of convolutions. The plausible competing explanations are "
    "<i>too few parameters</i> and <i>under-training</i>. <b>Design.</b> Three models, identical data, optimiser, epochs and lr grid: "
    "(A) CNN, 136,930 params; (B) MLP-matched, 137,642 params (0.7% apart); (C) MLP-big, 855,050 params (6.2x the CNN). "
    "B vs A isolates architecture at equal parameters; C vs A asks whether just adding parameters closes the gap. "
    "If capacity were the problem, C should approach A and the MLPs should underfit (low train accuracy). "
    "<b>Limit of fairness:</b> the CNN spends more FLOPs per image than either MLP, so parameter-matching is not compute-matching.")]
rows = [["Model", "Params", "lr", "Train acc", "Val acc", "Test acc", "Train-val gap"]]
for nm, lab in [("cnn", "CNN"), ("mlp_matched", "MLP-matched"), ("mlp_big", "MLP-big")]:
    r = q1a[nm]
    rows.append([lab, f"{r['params']:,}", r["lr"], pc(r["train_acc"]), pc(r["val_acc"]), pc(r["test_acc"]), pc(r["train_acc"] - r["val_acc"])])
table(rows, "Table 1. Q1a, seed 0, after 10 epochs (train accuracy is the running accuracy during the last epoch). Test accuracy for the selected lr.",
      colw=[.78 * inch, .55 * inch, .3 * inch, .5 * inch, .4 * inch, .45 * inch, .5 * inch])
fig("q1a_curves", "Figure 1. Q1a train (dashed) and validation (solid) accuracy and loss per epoch for the selected learning rates.")
story += [P_(
    "<b>Evidence.</b> At equal parameters the CNN reaches 78.7 test vs 51.2 for the MLP (Table 1). Making the MLP 6.2x larger raises it to 55.7, "
    "closing about 16% of the gap, while its train-validation gap grows to 19.8 points (MLP-matched: 7.6; CNN: 11.0) and its validation loss plateaus at about 1.33 (Fig. 1). "
    "The small MLP does underfit somewhat (59.7 train), so capacity matters a little, but the 6x larger one fits the training set better (76.3) and gains only 4.5 test points. "
    "We did not test longer training, so the claim is limited to the 10-epoch budget: <i>more parameters help the MLP little and mainly add overfitting, "
    "while the CNN is far ahead with 1/6 of the parameters</i>. This is consistent with inductive bias being the main factor; Q2 tests it directly.")]

story += [P_("2.2 Memorisation (Q1b)", h2), P_(
    "MLP-mem (4.2M params) is trained 60 epochs with Adam (lr 1e-3, no weight decay, no schedule, batch 128) on training labels "
    "of which a fraction <i>p</i> is replaced by uniformly random labels (p = 0, 0.4, 1.0; seed fixed). Every epoch we measure on the "
    "full training set, in eval mode: accuracy on the untouched samples; and, for the changed samples only, how often the model predicts "
    "the <i>assigned wrong label</i> (memorisation) versus the <i>true label</i> (structure learned despite the wrong label). "
    "Validation accuracy uses clean labels.")]
fig("q1b_memorisation", "Figure 2. Q1b training trajectories for 0%, 40% and 100% label noise.")
h4 = RUN("q1b_mem_noise0.4"); e = h4["extra"]
story += [P_(
    f"<b>Evidence.</b> With 40% noise, validation accuracy rises to a peak of {pc(max(h4['val_acc']))} at epoch {h4['best_epoch']} and then falls to "
    f"{pc(h4['val_acc'][-1])}. Over the same epochs the model predicts the <i>true</i> label of the corrupted samples {pc(e[2]['noisy_pred_true'])}% of the "
    f"time at epoch 3 (while fitting their assigned wrong label only {pc(e[2]['noisy_fit_assigned'])}%), "
    f"and this drops to {pc(e[-1]['noisy_pred_true'])}% by epoch 60 as the assigned-wrong-label fit climbs to {pc(e[-1]['noisy_fit_assigned'])}%. "
    "So useful structure is learned first (epochs ~1-6) and memorisation of the wrong labels takes over later, "
    "with the crossover (assigned-label fit exceeds true-label prediction on the corrupted samples) at epoch 13. With 100% random labels nothing generalises "
    f"(val {pc(q1b['1.0']['val_acc_final'])}, test {pc(q1b['1.0']['test_acc_final'])}) and, in 60 epochs, this model has only started to memorise "
    f"(fit {pc(RUN('q1b_mem_noise1.0')['extra'][-1]['noisy_fit_assigned'])}% at epoch 60): it has the capacity but needs far longer, "
    "consistent with the structured labels being fitted much faster than random ones. "
    f"Test accuracy of the final 40%-noise model is {pc(q1b['0.4']['test_acc_final'])} versus {pc(q1b['0.4']['test_acc_at_best_val_epoch'])} at the "
    f"best-validation epoch ({q1b['0.4']['best_val_epoch']}); the clean-label model gives {pc(q1b['0.0']['test_acc_final'])} final. "
    "<b>Caveat:</b> one seed per setting; the claim 'structure before memorisation' rests on the 40% run and the sign of the trajectories, which are large effects compared with seed noise (std about 0.3 points in Q2).")]

# ------------------------------------------------------------------ Q2
story += [P_("3. Q2: CNN versus MLP", h1), P_("3.1 Fair comparison (Q2a)", h2), P_(
    "<b>Fairness criterion (written in <font face='Courier'>q2.py</font> before the runs):</b> parameter counts within 1%, same data, epochs, optimiser and batch size, no augmentation, "
    "lr chosen from the same grid on validation, 3 seeds. The compared models are the CNN and MLP-matched of Q1 (Table 2). "
    "<b>Proposed reason for the difference:</b> the CNN's locality and weight sharing exploit the 2-D neighbourhood structure of images, which the MLP "
    "must otherwise learn pixel by pixel.")]
rows = [["Model", "Params", "Clean test", "Pixel-permuted test", "Block-16", "Block-8", "Block-4"]]
for nm, lab in [("mlp_matched", "MLP-matched"), ("cnn", "CNN")]:
    ps = q2[f"clean|{nm}"]["params"]
    rows.append([lab, f"{ps:,}", ms(q2[f"clean|{nm}"]), ms(q2[f"perm1|{nm}"])] + [(ms(q2[f'{b}|{nm}']) if nm == 'mlp_matched' else pc(q2[f'{b}|{nm}']['test_mean'])) for b in ("block16", "block8", "block4")])
table(rows, "Table 2. Q2 test accuracy (%). Mean ± std over seeds 0-2 (CNN block conditions: seed 0 only, lr taken from the clean run, to save compute). "
            "Each transform is a fixed permutation applied identically to train, validation and test.",
      colw=[.75 * inch, .5 * inch, .55 * inch, .6 * inch, .4 * inch, .4 * inch, .4 * inch])

story += [P_("3.2 Spatial disturbance (Q2b)", h2), P_(
    "<b>Disturbance.</b> One fixed random permutation of the 1,024 pixel positions (the three channel values of a pixel move together, so pixel values are unchanged and only their "
    "layout is destroyed), applied identically to train, validation and test images. <b>Expectation stated beforehand:</b> MLP unchanged (a fully connected first layer is "
    "permutation-equivariant up to initialisation); CNN falls to about MLP level. <b>Observed:</b> MLP 50.9 &rarr; 51.1; CNN 78.7 &rarr; 50.9 (Table 2, std about 0.3). "
    "The prediction held: the whole CNN advantage disappears, and the CNN becomes as overfit as the MLP (train 65.9 vs val 50.7). This supports the explanation of Q2a: the "
    "advantage comes from exploiting spatial arrangement, not from anything else the CNN does better (normalisation, depth, parameter use). "
    "The permuted CNN has exactly the same depth, BatchNorm, parameters and training as the clean CNN, so none of these can by themselves explain the clean advantage.")]

story += [P_("3.3 Local features versus global layout (Q2c)", h2), P_(
    "<b>Two explanations for the CNN's accuracy.</b> H1: it relies on <i>small local patches</i> (texture, colour blobs, small parts). H2: it needs the <i>global arrangement</i> of "
    "parts. <b>Experiment.</b> Block shuffling with block size b in {16, 8, 4} (permutation of the (32/b)<super>2</super> blocks, fixed, applied to train/val/test; b=32 is clean and b=1 the pixel "
    "permutation of Q2b); both models are retrained. Local structure inside each block survives, global layout does not. H1 predicts accuracy stays high for large "
    "b and falls only when blocks become smaller than the useful patches; H2 predicts a collapse already at b=16 or 8. "
    "<b>Result</b> (Table 2): CNN 78.7 (clean), 74.9 (b=16), 68.4 (b=8), 62.2 (b=4), 50.9 (b=1); the MLP stays at 50.9-51.3 throughout. "
    "At b=8 the CNN keeps 63% of its advantage over the MLP, at b=4 40%, so a graded mixture of the two: local structure at "
    "scales of 4-8 px carries most of the accuracy (against H2's collapse), but the remainder depends on how those pieces are arranged "
    "(against pure H1). <b>Caveat:</b> block borders introduce artificial edges, so b=8 and b=4 also test robustness to those edges; the CNN block runs use a single seed.")]
fig("q2_results", "Figure 3. Left: test accuracy by condition (error bars: std over 3 seeds where available). Right: reliability curves (test, seed 0) for clean, block-8 and permuted.")
rows = [["Condition (CNN)", "Test acc", "Mean conf", "Conf. correct", "Conf. wrong", "ECE"]]
for c in ["clean", "block16", "block8", "block4", "perm1"]:
    r = q2[f"{c}|cnn"]["calib"]
    rows.append([c, pc(r["acc"]), f"{r['conf']:.3f}", f"{r['conf_correct']:.3f}", f"{r['conf_wrong']:.3f}", f"{r['ece']:.3f}"])
r = q2["clean|mlp_matched"]["calib"]; rows.append(["MLP clean", pc(r["acc"]), f"{r['conf']:.3f}", f"{r['conf_correct']:.3f}", f"{r['conf_wrong']:.3f}", f"{r['ece']:.3f}"])
table(rows, "Table 3. Q2 test-set confidence analysis, seed 0 (confidence = max softmax probability, ECE with 15 bins).",
      colw=[.9 * inch, .5 * inch, .55 * inch, .65 * inch, .6 * inch, .4 * inch])
pcn, pcp, pcb = q2["clean|cnn"]["calib"]["per_class"], q2["perm1|cnn"]["calib"]["per_class"], q2["block8|cnn"]["calib"]["per_class"]
drops = sorted([(pcn[i] - pcp[i], CLASSES[i]) for i in range(10)], reverse=True)
story += [P_(
    f"<b>Errors and confidence.</b> Shuffling not only lowers accuracy but changes how the CNN fails. Its mean confidence on wrong predictions falls (0.63 clean &rarr; 0.58 b=8 &rarr; 0.50 permuted) but "
    f"its mean confidence stays above its accuracy, so ECE rises from 0.039 to 0.077 under permutation: it becomes <i>over-confident</i> relative to its accuracy, like the MLP "
    f"(ECE 0.045, accuracy 51%). Every class loses between {100*drops[-1][0]:.0f} ({drops[-1][1]}) and {100*drops[0][0]:.0f} points when pixels are permuted; the largest drops are {drops[0][1]} (-{100*drops[0][0]:.0f}), "
    f"{drops[1][1]} (-{100*drops[1][0]:.0f}) and {drops[2][1]} (-{100*drops[2][0]:.0f}), so the loss is broad rather than concentrated in a few classes. <b>Conclusion linking Q2a-c:</b> the CNN-MLP gap at equal parameters is a spatial-structure effect; "
    "removing structure removes the gap; and the structure used is mostly local (4-8 px) but not exclusively so.")]

# ------------------------------------------------------------------ Q3
story += [P_("4. Q3: Autoencoders", h1), P_("4.1 Bottleneck capacity (Q3a)", h2), P_(
    "Encoder: conv(3,32,4,s2)-conv(32,64,4,s2)-conv(64,128,4,s2) (ReLU) &rarr; flatten (2048) &rarr; linear to <i>d</i>; decoder mirrors it (linear, three transposed convs, sigmoid). "
    "Only <i>d</i> differs (16, 64, 256); Adam, lr 1e-3, batch 128, 20 epochs, MSE on [0,1] pixels, no labels. The selected model is the one with the lowest validation "
    "MSE, which is simply the largest bottleneck, because reconstruction error is monotone in capacity; this criterion says nothing about whether the code is useful (Q3c).")]
rows = [["d", "Params", "Train MSE", "Val MSE", "Test MSE"]]
for k in (16, 64, 256):
    rows.append([k, f"{q3a[str(k)]['params']:,}", f"{q3a[str(k)]['train_mse']:.5f}", f"{q3a[str(k)]['val_mse']:.5f}", f"{q3a['test_mse_all'][str(k)]:.5f}" + (" (selected)" if k == 256 else "")])
table(rows, f"Table 4. Q3a. Selection by validation MSE only; test MSE for the selected model is {q3a['test_mse_selected']:.5f} (PSNR {q3a['test_psnr_selected']:.1f} dB). Test MSE of d=16/64 is shown for Q3c only.",
      colw=[.3 * inch, .75 * inch, .65 * inch, .65 * inch, .95 * inch])

story += [P_("4.2 Denoising (Q3b)", h2), P_(
    "<b>Corruption:</b> additive Gaussian noise, &sigma;=0.2 on [0,1] pixels, clipped to [0,1]. This is a corruption where reconstruction could plausibly help (independent per-pixel noise on a "
    "correlated image). <b>DAE protocol:</b> same architecture as the selected AE (d=256), input = clean training image + fresh noise sampled every batch, target = the clean image, MSE, "
    "no labels, 20 epochs; validation uses one fixed noisy copy of the validation set. The test set is corrupted once with a fixed generator seed. "
    "<b>Classifier:</b> the fixed clean-trained CNN of Q1 (never retrained or adapted).")]
rows = [["Classifier input (test)", "Acc", "MSE to clean", "Mean conf", "Conf wrong"]]
for k, lab in [("clean", "clean images"), ("corrupted", "noisy images"), ("clean_AE(corrupted)", "clean AE(noisy)"), ("DAE(corrupted)", "DAE(noisy)"),
               ("clean_AE(clean)", "clean AE(clean)"), ("DAE(clean)", "DAE(clean)")]:
    r = q3b[k]; rows.append([lab, pc(r["acc"]), f"{r['mse_to_clean']:.4f}", f"{r['conf']:.3f}", f"{r['conf_wrong']:.3f}"])
table(rows, "Table 5. Q3b. Test classification with the fixed CNN. 'MSE to clean' is pixel error of the classifier input w.r.t. the clean test image.",
      colw=[1.15 * inch, .4 * inch, .75 * inch, .6 * inch, .6 * inch])
fig("q3_recon", "Figure 4. Rows: clean; AE d=16, 64, 256 reconstructions; noisy input; clean AE (d=256) on noisy; DAE on noisy.")
story += [P_(
    "<b>Decision.</b> Noise at &sigma;=0.2 drops the CNN from 78.7 to 18.4 (and it stays confident, mean confidence 0.68, so it fails with little warning). "
    "Both reconstructions help a lot (clean AE 43.6, DAE 44.3), and the DAE reduces pixel error most (MSE to clean 0.0054 vs 0.0072 for the clean AE and 0.0333 for the noisy input), "
    "but the classification gain of the DAE over the clean AE is only 0.65 points, which is about the sampling error of a 10,000-image test set (&asymp;0.5 points) and we have a single DAE. "
    "The key observation is the row 'clean AE(clean)': simply passing <i>clean</i> images through the AE gives 54.6, i.e. the reconstruction itself discards 24 points of class information "
    "that the classifier relies on (fine texture and edges), regardless of noise. So <b>denoising does not recover the information the classifier needs</b>: "
    "it removes the noise but is limited by what the 256-d bottleneck keeps, and lower pixel MSE does not translate to classification accuracy. "
    "(Fine-tuning the classifier on reconstructions or a larger-capacity AE might change this; not tested.)")]

story += [P_("4.3 Class information in frozen codes (Q3c)", h2), P_(
    "Encoders are frozen; features are standardised with train statistics and a linear softmax classifier (L-BFGS, L2 strength from {1e-4, 1e-3, 1e-2}, chosen on validation) is fit on the 45,000 "
    "training codes. Baselines: raw pixels, and an untrained random encoder with the same architecture (d=256).")]
rows = [["Representation", "Dim", "Test MSE", "Val acc", "Test acc"]]
for k, lab, mk in [("ae_d16", "AE d=16", "ae_d16"), ("ae_d64", "AE d=64", "ae_d64"), ("ae_d256", "AE d=256", "ae_d256"), ("dae_d256", "DAE d=256", "dae_d256_cleaninput"),
                   ("random_enc_d256", "random enc. d=256", None), ("raw_pixels", "raw pixels", None)]:
    r = q3c[k]; rows.append([lab, r.get("dim", 3072), (f"{q3c['test_mse'][mk]:.4f}" if mk else "-"), pc(r["val_acc"]), pc(r["test_acc"])])
table(rows, "Table 6. Q3c. Linear-probe accuracy (%) on frozen codes vs test reconstruction MSE on clean test images (the DAE is evaluated on clean input here).",
      colw=[1.1 * inch, .4 * inch, .65 * inch, .5 * inch, .5 * inch])
fig("q3c_mse_vs_probe", "Figure 5. Probe accuracy versus reconstruction error for the three clean AEs.", w=CW * .75)
story += [P_(
    "<b>Do reconstruction quality and class information agree?</b> Across the three clean AEs, yes: lower MSE goes with higher probe accuracy (34.1, 38.5, 44.5; Spearman &minus;1.0, which with three points is only an ordering statement). "
    "But the agreement is weak and breaks elsewhere. (1) An <i>untrained</i> random encoder of the same size gets 41.7, only 2.8 points below the trained d=256 AE, so most of the apparent class information is "
    "available from random convolutional features of the right dimensionality (and raw pixels give 41.3 test). (2) The DAE has <i>worse</i> clean-input MSE (0.0055 vs 0.0032) "
    "but <i>better</i> class information (47.4 vs 44.5), so the two quantities disagree: the denoising objective forces the code to ignore pixel-level noise-like detail and keep more stable structure. "
    "Reconstruction error is therefore not a reliable proxy for representation quality; these are single-seed results, and the d-trend is confounded with dimensionality (probe dimension grows with d).")]

# ------------------------------------------------------------------ Q4
story += [P_("5. Q4 (bonus): SVMs and clustering", h1), P_(
    "<b>Fixed representation:</b> the 256-d frozen encoder code of the Q3a-selected AE (d=256), standardised with training statistics; unchanged throughout Q4. "
    "<b>Training subset:</b> a fixed stratified subset of 10,000 training images (1,000 per class), drawn with <font face='Courier'>RandomState(26015)</font> "
    "before any model selection (indices in <font face='Courier'>results/q4_subset.json</font>) and used for every SVM. Validation (5,000) and test (10,000) are the full sets. "
    "scikit-learn <font face='Courier'>SVC</font> (one-vs-one, <font face='Courier'>gamma='scale'</font>, poly degree 3 with coef0=1, <font face='Courier'>max_iter</font> 300,000).")]
story += [P_("5.1 C and kernel sweep (Q4a)", h2)]
rows = [["Kernel", "C", "Train acc", "Val acc", "SV frac"]]
for r in sw:
    rows.append([r["kernel"], f"{r['C']:g}", pc(r["train_acc"]), pc(r["val_acc"]), f"{r['sv_frac']:.3f}"])
table(rows, "Table 7. Complete sweep (6 values of C over 5 orders of magnitude, 3 kernels). SV frac = unique support vectors / 10,000.",
      colw=[.6 * inch, .4 * inch, .6 * inch, .6 * inch, .6 * inch], fs=6.2)
fig("q4_sweep_cluster", "Figure 6. Left: validation accuracy (solid) and training accuracy (dashed) vs C; right: support-vector fraction vs C.")
sel = q4s["selected"]
story += [P_(
    f"<b>Selection (validation only):</b> {sel['kernel']} kernel, C={sel['C']:g} (val {pc(sel['val_acc'])}). "
    f"<b>Evaluated once on test: {pc(q4s['test_acc'])}%</b> (train {pc(sel['train_acc'])}, subset of 10,000). "
    "<b>Behaviour.</b> <i>Underfitting:</i> at small C all kernels have low training and validation accuracy together (rbf C=0.01: 32.8 / 31.3) with a support-vector fraction near 1, "
    "since a wide margin leaves almost every point inside it. <i>Overfitting:</i> rbf and poly3 reach &ge;99.5% training accuracy at C&ge;10 while validation stays at about 50.7 and 48.4 (and "
    "poly3 declines slightly above its best 49.3 at C=1): the gap of about 50 points is memorisation, and larger C changes nothing once all points are separated "
    "(rows for C=100 and 1000 are identical for poly3 and rbf). Linear is different: it has too little flexibility (best val 41.5 at C=0.01, train never above 57), and C&ge;100 collapses to 26.6 / 22.9 train accuracy "
    "(fits take 150 s, about 4x the others, so most likely the libsvm iteration cap was hit: an optimisation failure, not a property of the model; we did not verify this and report the numbers as observed). "
    "Support-vector fractions of 0.86-1.0 throughout are a symptom of the representation: classes overlap heavily in the AE code, so nearly every point is on or inside the margin. "
    "The sweep also shows the chosen model is barely better than rbf C=100 (50.7 vs 50.3, within noise on 5,000 validation images, std &asymp; 0.7).")]

story += [P_("5.2 Difficult class pairs (Q4b)", h2), P_(
    "The three most confused unordered class pairs on the <i>test</i> confusion matrix of the selected SVM are cat/dog, automobile/truck and bird/deer. For each pair we retrain the binary SVM "
    "(same kernel, C and gamma, on the pair's training subset); this is the same libsvm sub-problem that the one-vs-one classifier solves, so its support vectors and decision scores are those of the "
    "pairwise classifier. Figure 7 shows, per pair, training support vectors with the largest dual coefficient (first row: left four class A, right four class B) and the 8 test images closest to the "
    "decision boundary (with correctness of the pairwise classifier and signed score).")]
rows = [["Pair", "Test errs", "Pair SV frac", "Bounded SV", "Binary acc", "Err rate: nearest-50", "Err rate: rest", "Centroid dist / spread"]]
for p in q4s["hard_pairs"]:
    rows.append(["/".join(p["names"]), p["test_pair_errors"], f"{p['sv_frac_pair']:.2f}", f"{p['bounded_sv_frac']:.3f}", pc(p["binary_test_acc"]),
                 pc(p["err_rate_nearest50"]), pc(p["err_rate_rest"]), f"{p['sep_ratio']:.2f}"])
table(rows, "Table 8. Q4b. Pairwise statistics. 'nearest-50' = the 50 test examples of the pair with the smallest |score|; 'Centroid dist / spread' = distance between class centroids divided by mean within-class distance to centroid (in the standardised 256-d space).",
      colw=[.85 * inch, .35 * inch, .4 * inch, .4 * inch, .35 * inch, .45 * inch, .35 * inch, .45 * inch], fs=5.8)
fig("q4_pairs", "Figure 7. Hard pairs (cat/dog, automobile/truck, bird/deer). Each pair: row 1 training support vectors (class A left, B right); row 2 nearest-boundary test images: true class (3 letters), whether the pairwise classifier is correct (ok/ERR) and its signed decision score.")
story += [P_(
    "<b>Interpretation.</b> For all three pairs the distance between class centroids is only 0.12-0.19 times the typical within-class spread, i.e. the classes are nearly "
    "concentric blobs in this space, and 85-93% of the pair's training points are support vectors; very few are bounded (&alpha;=C, 0.1-4%), so the RBF "
    "solution is a memorising, nearly-everywhere-curved boundary rather than a soft-margin compromise. Test errors concentrate at the boundary (error rate 54% in cat/dog, 42-44% in the others "
    "among the 50 closest points versus 23-39% for the rest) but are not confined there: even far from the boundary, 24-39% of examples are wrong, which is what overlapping classes (not "
    "noise at the margin) look like. In Fig. 7 the nearest-boundary test images are visually similar across the pair (e.g. cat and dog heads in close-ups, cars and trucks, birds and deer on grass). "
    "A plausible reading (not tested here) is that the reconstruction-trained code mostly keeps colour and low-frequency layout, which these pairs share, so the "
    "errors reflect the representation more than the SVM.")]

story += [P_("5.3 Clustering versus classification (Q4c)", h2), P_(
    "Labels are never used to fit a clustering. We fit KMeans (k=5,10,15,20,30) and diagonal-covariance GMMs (k=10,20) on the 45,000 training codes, "
    "and choose method and k by <i>validation</i> ARI (labels used for selection only; NMI and train silhouette are logged too). "
    f"Chosen: {q4c['chosen']['method']} with k={q4c['chosen']['k']} (val ARI {q4c['chosen']['val_ari']:.3f}). The final model was fit on training codes and assigned test codes; "
    "test labels are used only afterwards. To compare with the SVM we map each cluster to the majority class of its <i>training</i> members.")]
cont = np.array(q4c["contingency"])
rows = [["cl."] + [c[:4] for c in CLASSES]] + [[k] + [int(v) for v in cont[k]] for k in range(cont.shape[0])]
table(rows, f"Table 9. Q4c test cluster-to-class contingency (rows: clusters; columns: true classes). Test ARI = {q4c['test_ari']:.3f}, test NMI = {q4c['test_nmi']:.3f}.",
      colw=[.25 * inch] + [.3 * inch] * 10, fs=5.6)
gap = sorted([(q4c["svm_recall"][i] - q4c["cluster_recall"][i], i) for i in range(10)], reverse=True)
rows = [["Class", "SVM recall", "Cluster-mapped recall"]] + [[CLASSES[i], pc(q4c["svm_recall"][i]), pc(q4c["cluster_recall"][i])] for i in range(10)]
table(rows, f"Table 10. Per-class test recall of the selected SVM vs the cluster-to-class mapping (overall: SVM {pc(q4s['test_acc'])}, clusters {pc(q4c['cluster_mapped_test_acc'])}).",
      colw=[.9 * inch, .8 * inch, 1.1 * inch], fs=6.2)
po = q4c["pair_overlap"]
story += [P_(
    f"<b>Result.</b> Test ARI {q4c['test_ari']:.3f}, NMI {q4c['test_nmi']:.3f}: the clusters carry little class information (chance is 0). Mapped to classes they give "
    f"{pc(q4c['cluster_mapped_test_acc'])}% test accuracy versus {pc(q4s['test_acc'])}% for the SVM. <b>Disagreement:</b> the clearest case is <i>automobile</i> (SVM recall {pc(q4c['svm_recall'][1])}% "
    f"vs {pc(q4c['cluster_recall'][1])}% for clusters: no cluster has automobile as its training majority; 298 of the 1,000 test automobiles fall in cluster 6, which trucks (349) win) and, equally, <i>deer</i> "
    f"({pc(q4c['svm_recall'][4])}% vs {pc(q4c['cluster_recall'][4])}%; deer split between clusters 2 and 4, which frog and bird win). The supervised SVM can separate classes along directions that carry little variance, while clustering follows the dominant "
    "variance (background colour/brightness). In contrast truck (56.0 vs 51.2) and airplane (59 vs 49) are similar under both methods, meaning those classes are aligned with the dominant structure. "
    f"At the pair level, cat and dog test images share clusters heavily (cluster-distribution overlap {po['cat-dog']['cluster_overlap']:.2f}) whereas the SVM separates them with a pairwise confusion rate of "
    f"{po['cat-dog']['svm_confusion_rate']:.2f} (fraction of the pair's test images misclassified into the other class): the clusters treat cat and dog as nearly one group while the classifier separates most of them. "
    "The same pattern holds for automobile/truck (overlap 0.81 vs confusion 0.18) and bird/deer (0.85 vs 0.15). The SVM is above chance everywhere, but the representation limits both.")]

# ------------------------------------------------------------------ discussion
story += [P_("6. Limitations and honest notes", h1), P_(
    "(1) <b>Compute:</b> CPU-only, 10 epochs, 2-point lr grid, no augmentation; absolute accuracies (e.g. 78.7 for the CNN) are far below tuned CIFAR10 models and the gaps would change with longer training. "
    "(2) <b>Seeds:</b> Q2 clean/permuted use 3 seeds (std about 0.3); the CNN block-shuffle runs, Q1b, Q3 and Q4 use one seed, so differences below about 1 point (e.g. DAE vs clean AE in Q3b) are not established. "
    "(3) <b>Q2c</b> block-shuffling introduces artificial block-border edges. (4) <b>Q3b/c</b> conclusions are for &sigma;=0.2 noise and d=256 only. "
    "(5) <b>Q4:</b> 10,000-image training subset; the linear-kernel collapse at C&ge;100 is an optimisation (iteration cap) artefact; the SVM sub-problem for the pair analysis is retrained on the pair "
    "rather than extracted from the one-vs-one model (the problems are identical). (6) The test set was used only for the reported numbers; test-set diagnostics (confidence, per-class recalls, hard-pair choice in Q4b which the assignment requires to be based on test errors) were not used to change any model or setting.")]
story += [P_("References", h1), P_(
    "[1] A. Krizhevsky, <i>Learning Multiple Layers of Features from Tiny Images</i>, 2009. &nbsp; [2] C. Zhang et al., <i>Understanding deep learning requires rethinking generalization</i>, ICLR 2017. "
    "&nbsp; [3] C.-C. Chang, C.-J. Lin, <i>LIBSVM</i>, ACM TIST 2011. &nbsp; [4] P. Vincent et al., <i>Extracting and composing robust features with denoising autoencoders</i>, ICML 2008.", cap)]


def page_deco(c, d):
    c.setFont("Times-Roman", 8); c.drawCentredString(W / 2, 0.45 * inch, str(d.page))


doc = BaseDocTemplate(P("report", "report.pdf"), pagesize=letter, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                      title="ML Assignment 3 - Prakash Singh (M26CSA015)", author="Prakash Singh")
colh = H - 2 * M
first = PageTemplate("first", [Frame(M, H - M - 1.0 * inch, W - 2 * M, 1.0 * inch, id="t", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                               Frame(M, M, CW, colh - 1.0 * inch, id="l1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                               Frame(M + CW + GAP, M, CW, colh - 1.0 * inch, id="r1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page_deco)
later = PageTemplate("later", [Frame(M, M, CW, colh, id="l", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                               Frame(M + CW + GAP, M, CW, colh, id="r", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page_deco)
doc.addPageTemplates([first, later])
story.insert(0, NextPageTemplate("later"))
doc.build(story)
from pypdf import PdfReader  # noqa
print("pages:", len(PdfReader(P("report", "report.pdf")).pages))
