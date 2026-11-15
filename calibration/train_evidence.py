"""Train the learned possession-evidence model on OTHER matches and export it.

Data: Metrica Sports public sample games 1, 2 and 3 (git clone
https://github.com/metrica-sports/sample-data ../ext, or set MTAH_EXT_DIR). Their tracking is made
to look like the provided automatic tracking (drifting ball offset + player dropouts, matched on
input statistics only), and their event data gives the possession labels. The provided match is
never used for training.

    python calibration/train_evidence.py          # writes mtah/evidence_model.npz

Requires scikit-learn (only for training; the pipeline itself evaluates the exported trees in
numpy).
"""
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import external, features as F, gbtrees  # noqa: E402

SIGMA = 1.1          # ball offset matched to the provided tracking (calibration/calibrate.py)
SEEDS = (0, 1)       # two independent noise draws per game
PARAMS = dict(max_iter=200, max_depth=4, learning_rate=0.1, random_state=0)


def training_set():
    Xs, ys = [], []
    for g in (1, 2, 3):
        data, ev = external.load_any(g)
        ref = external.reference(ev, len(data["bx"]))
        for seed in SEEDS:
            d = external.drop_players(external.degrade(data, SIGMA, seed=seed), seed=seed)
            X = F.build(d)
            m = (ref < 2) & np.isfinite(X[:, 0])
            y = (ref[m] == 0).astype(int)
            # both views of every frame: home/away swapped, label flipped -> no team bias
            Xs += [X[m], F.swap(X[m])]
            ys += [y, 1 - y]
    return np.concatenate(Xs), np.concatenate(ys)


def export(model, path):
    feats, thrs, lefts, rights, leaves, values, starts = [], [], [], [], [], [], [0]
    for (pred,) in model._predictors:
        n = pred.nodes
        feats.append(n["feature_idx"].astype(np.int32))
        thrs.append(n["num_threshold"].astype(np.float64))
        lefts.append(n["left"].astype(np.int64))
        rights.append(n["right"].astype(np.int64))
        leaves.append(n["is_leaf"].astype(bool))
        values.append(n["value"].astype(np.float64))
        starts.append(starts[-1] + len(n))
    np.savez(path, feature=np.concatenate(feats), threshold=np.concatenate(thrs),
             left=np.concatenate(lefts), right=np.concatenate(rights),
             is_leaf=np.concatenate(leaves), value=np.concatenate(values),
             tree_start=np.array(starts), baseline=float(np.ravel(model._baseline_prediction)[0]),
             feature_names=np.array(F.NAMES))


def main():
    X, y = training_set()
    print("training frames:", len(y))
    model = HistGradientBoostingClassifier(**PARAMS).fit(X, y)
    export(model, gbtrees.MODEL)
    exported = gbtrees.load()
    sample = X[:: max(1, len(X) // 20000)]
    gap = np.abs(gbtrees.predict_proba(exported, sample) - model.predict_proba(sample)[:, 1]).max()
    print("trees:", len(exported["tree_start"]) - 1, " max |numpy - sklearn| =", gap)
    print("wrote", gbtrees.MODEL)


if __name__ == "__main__":
    main()
