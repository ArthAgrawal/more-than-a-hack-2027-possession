"""Produce the submission CSV and print automatic, label-free sanity checks.

    python -m mtah.run [--out evaluation/results/mtah.csv]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from mtah.load import load, N  # noqa: E402
from mtah.possession import possession, P, FPS  # noqa: E402

LETTERS = np.array(["H", "A", "D"])


def graded_mask():
    g = np.zeros(N, bool)
    for _, a, b in half_video_blocks(GAME):
        g[a:b + 1] = True
    return g


def sanity(lab, alive, graded):
    """Statistics that need no ground truth."""
    spells = []
    for _, a, b in half_video_blocks(GAME):
        x = lab[a:b + 1]
        change = np.flatnonzero(np.diff(x)) + 1
        bounds = np.r_[0, change, len(x)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if x[s] != 2:
                spells.append((e - s, x[s]))
    d = np.array([s[0] for s in spells]) / FPS
    t = np.array([s[1] for s in spells])
    g = graded
    print("graded frames      ", g.sum())
    print("alive share        %.3f  (%.1f min)" % (alive[g].mean(), alive[g].sum() / FPS / 60))
    print("possessions        %d  (H %d, A %d)" % (len(d), (t == 0).sum(), (t == 1).sum()))
    print("duration s  mean %.1f  median %.1f  <1s %d  <2s %d" % (
        d.mean(), np.median(d), (d < 1).sum(), (d < 2).sum()))
    print("home share of alive %.3f" % ((lab[g] == 0).sum() / max(1, (lab[g] < 2).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "mtah.csv"))
    args = ap.parse_args()
    data = load()
    g = graded_mask()
    lab, alive = possession(data, g, P)
    letters = LETTERS[lab].astype(object)
    letters[~g] = "X"  # outside play there is no ground truth; abstaining there is free
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        for i, l in enumerate(letters):
            fh.write("%d,%s\n" % (i, l))
    sanity(lab, alive, g)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
