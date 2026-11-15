"""Leave-one-game-out test of the learned evidence against the distance rule.

Train on two Metrica sample games, test on the third (all three ways), each with two unseen
noise draws. Reproduces the comparison table in TECHNICAL_REPORT.md. Needs scikit-learn and the
external sample data (see train_evidence.py). Never touches the provided match.

    python calibration/cross_validate.py
"""
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah import external, features as F, possession as pos  # noqa: E402
from train_evidence import PARAMS, SIGMA  # noqa: E402

GAMES = (1, 2, 3)
TRAIN_SEEDS, TEST_SEEDS = (0, 1), (2, 3)
LAMS = (12, 16, 24, 28, 32, 48, 64)


def main():
    raw = {}
    for g in GAMES:
        d, ev = external.load_any(g)
        raw[g] = (d, external.reference(ev, len(d["bx"])))
    sets = {}
    for g in GAMES:
        for s in TRAIN_SEEDS + TEST_SEEDS:
            d = external.drop_players(external.degrade(raw[g][0], SIGMA, seed=s), seed=s)
            sets[g, s] = dict(d=d, ref=raw[g][1], X=F.build(d), alive=pos.alive_mask(d, raw[g][1] != 3))

    def score(S, lab):
        r = pm.evaluate(S["ref"], np.where(S["ref"] != 3, lab, 3))
        return r["spell_f1_mean"], r["macro_f1"]

    rule, learned = [], {lam: [] for lam in LAMS}
    for test in GAMES:
        Xs, ys = [], []
        for g, s in [(g, s) for g in GAMES if g != test for s in TRAIN_SEEDS]:
            S = sets[g, s]
            m = (S["ref"] < 2) & np.isfinite(S["X"][:, 0])
            y = (S["ref"][m] == 0).astype(int)
            Xs += [S["X"][m], F.swap(S["X"][m])]
            ys += [y, 1 - y]
        model = HistGradientBoostingClassifier(**PARAMS).fit(np.concatenate(Xs), np.concatenate(ys))
        for s in TEST_SEEDS:
            S = sets[test, s]
            lab, _ = pos.possession(S["d"], S["ref"] != 3, dict(pos.P, evidence="distance"),
                                    alive=S["alive"])
            rule.append(score(S, lab))
            ok = np.isfinite(S["X"][:, 0])
            p = (model.predict_proba(S["X"][ok])[:, 1]
                 + 1 - model.predict_proba(F.swap(S["X"][ok]))[:, 1]) / 2
            ev = np.zeros(len(ok))
            ev[ok] = 0.5 * np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
            for lam in LAMS:
                lab = np.full(len(ev), pos.D, np.int8)
                for a, b in zip(*pos._runs(S["alive"])):
                    lab[a:b] = pos.viterbi2(ev[a:b], lam)
                learned[lam].append(score(S, lab))
    print("distance rule      spellF1 %.4f  macroF1 %.4f  per test %s"
          % (*np.mean(rule, 0), np.round([r[0] for r in rule], 3)))
    for lam in LAMS:
        print("learned, lam %3d   spellF1 %.4f  macroF1 %.4f  per test %s"
              % (lam, *np.mean(learned[lam], 0), np.round([r[0] for r in learned[lam]], 3)))


if __name__ == "__main__":
    main()
