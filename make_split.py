from common import *
d = get_data()
print("seed", SEED, "train", len(d["ytr"]), "val", len(d["yva"]), "test", len(d["yte"]))
print("val per class", np.bincount(d["yva"]).tolist())
