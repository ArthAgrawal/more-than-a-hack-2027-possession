"""Plain-numpy evaluation of the exported gradient-boosted trees (no scikit-learn needed).

The model file stores, for every tree, its nodes as parallel arrays. Prediction follows
scikit-learn's HistGradientBoosting rule exactly: go left when x <= threshold (NaN never occurs
here), sum leaf values over trees, add the baseline, apply the logistic function.
"""
import os

import numpy as np

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_model.npz")


def load(path=MODEL):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def raw_score(model, X):
    feat, thr, left, right, leaf, value, start = (
        model[k] for k in ("feature", "threshold", "left", "right", "is_leaf", "value", "tree_start"))
    out = np.full(len(X), float(model["baseline"]))
    for t in range(len(start) - 1):
        base = start[t]
        node = np.full(len(X), base, np.int64)
        active = ~leaf[node]
        while active.any():
            idx = node[active]
            go_left = X[active, feat[idx]] <= thr[idx]
            node[active] = np.where(go_left, left[idx], right[idx]) + base
            active = ~leaf[node]
        out += value[node]
    return out


def predict_proba(model, X):
    return 1.0 / (1.0 + np.exp(-raw_score(model, X)))
