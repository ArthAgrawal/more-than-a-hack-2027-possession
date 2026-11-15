"""Reproduce how the possession parameters in mtah/possession.py were chosen.

Uses ONLY other matches: Metrica Sports' public sample games 1 and 2
(git clone https://github.com/metrica-sports/sample-data ../ext  or set MTAH_EXT_DIR).
The provided BAR-ATH match is read only for its *input* statistics (how far the tracked ball sits
from the nearest player), never for labels.

    python calibration/calibrate.py            # noise fit + grid search, prints the top settings
"""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (the official scorer's metric library)
from mtah import external, possession as pos  # noqa: E402
from mtah.load import load  # noqa: E402
from mtah.run import graded_mask  # noqa: E402

SIGMAS = (0.8, 1.0, 1.2, 1.4)
SIGMA = 1.1  # chosen below: matches the provided match's 10th-50th percentile profile
SEEDS = (0, 1)
GRID = dict(ctrl_radius=(2.0, 2.5, 3.0), ev_scale=(1.0, 2.0), fast_ball=(0.0, 10.0, 20.0),
            lam=(15.0, 30.0, 60.0))


def nearest_profile(d, mask):
    _, near = pos.nearest_distance(d["px"], d["py"], d["bx"], d["by"])
    return np.percentile(near[mask & np.isfinite(near)], [10, 25, 50])


def main():
    games = {}
    for g in (1, 2):
        data, ev = external.load_game(g)
        games[g] = (data, external.reference(ev, len(data["bx"])))

    target = load()
    print("provided match, nearest-player distance p10/25/50:",
          nearest_profile(target, graded_mask() & np.isfinite(target["bx"])).round(2))
    data1, ref1 = games[1]
    for s in SIGMAS:
        prof = nearest_profile(external.degrade(data1, s), ref1 < 2)
        print("  external game 1 degraded sigma %.1f:" % s, prof.round(2))

    deg = {(g, seed): (external.degrade(d, SIGMA, seed=seed), r)
           for g, (d, r) in games.items() for seed in SEEDS}

    def score(p):
        out = []
        for d, ref in deg.values():
            graded = ref != pm.X
            lab, _ = pos.possession(d, graded, p)
            pred = np.where(graded, lab, pm.X)
            r = pm.evaluate(ref, pred)
            out.append((r["spell_f1_mean"], r["macro_f1"]))
        return np.mean(out, 0)

    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        p = dict(pos.P, **dict(zip(keys, values)))
        rows.append((*score(p), *values))
    rows.sort(reverse=True)
    print("\nspellF1  macroF1  " + "  ".join(keys))
    for r in rows[:10]:
        print("%.4f   %.4f   " % r[:2] + "  ".join("%g" % v for v in r[2:]))
    print("\nchosen (mtah/possession.py P): spellF1 %.4f macroF1 %.4f" % tuple(score(pos.P)))


if __name__ == "__main__":
    main()
