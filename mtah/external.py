"""External calibration data: Metrica Sports public sample games 1 and 2.

https://github.com/metrica-sports/sample-data  (tracking + human event data, other matches)

We derive a reference possession timeline from the *event data of those games* and use it to
choose the control parameters. Nothing here touches the provided BAR-ATH match.

Reference construction (mirrors the challenge definition of possession):
  * on-ball events (PASS, SET PIECE, SHOT, RECOVERY, BALL LOST, CARRY) give the team in possession
    from their start frame onwards
  * the ball is dead from a BALL OUT / goal / foul until the next SET PIECE starts
"""
import os

import numpy as np
import pandas as pd

EXT = os.environ.get("MTAH_EXT_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ext")
L, W = 105.0, 68.0
H, A, D, X = 0, 1, 2, 3
ON_BALL = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "CARRY"}


def _tracking(path):
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    cols = list(raw.columns)
    players = [c for c in cols if c.startswith("Player")]
    xs = raw[players].to_numpy(float) * L
    ys = raw[[cols[cols.index(c) + 1] for c in players]].to_numpy(float) * W
    bi = cols.index("Ball")
    bx = raw.iloc[:, bi].to_numpy(float) * L
    by = raw.iloc[:, bi + 1].to_numpy(float) * W
    return raw["Period"].to_numpy(), xs, ys, bx, by


def load_game(g):
    base = os.path.join(EXT, "data", "Sample_Game_%d" % g, "Sample_Game_%d_" % g)
    per, hx, hy, bx, by = _tracking(base + "RawTrackingData_Home_Team.csv")
    _, ax, ay, _, _ = _tracking(base + "RawTrackingData_Away_Team.csv")
    px = np.concatenate([hx, ax], 1)
    py = np.concatenate([hy, ay], 1)
    team = np.r_[np.zeros(hx.shape[1], np.int8), np.ones(ax.shape[1], np.int8)]
    sp = np.full_like(px, np.nan)
    sp[1:-1] = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) * 12.5
    ev = pd.read_csv(base + "RawEventsData.csv")
    return dict(px=px, py=py, ps=sp, bx=bx, by=by, team=team, period=per), ev


def reference(ev, n):
    """Per-frame H/A/D reference from human events. Frames are 1-based in the files."""
    lab = np.full(n, X, np.int8)
    ev = ev.sort_values(["Start Frame", "End Frame"]).reset_index(drop=True)
    dead_from = None
    marks = []  # (frame, label)
    for _, r in ev.iterrows():
        t, st = r["Type"], str(r["Subtype"])
        s, e = int(r["Start Frame"]) - 1, int(r["End Frame"]) - 1
        side = H if r["Team"] == "Home" else A
        if t == "SET PIECE":
            marks.append((s, side))
        elif t in ON_BALL:
            marks.append((s, side))
        if t == "BALL OUT" or ("GOAL" in st and t == "SHOT") or ("-OUT" in st and t == "SHOT"):
            marks.append((max(e, s), D))
        if t == "FAULT RECEIVED" or ("FAULT" in st and t == "CHALLENGE"):
            marks.append((s, D))
    marks.sort()
    for (f0, l0), (f1, _) in zip(marks, marks[1:] + [(n, X)]):
        if f0 < n:
            lab[f0:min(f1, n)] = l0
    # a dead stretch ends only at a set piece: on-ball marks inside it (e.g. the BALL OUT pass
    # itself is earlier, so fine) are rare; leave them.
    return lab


def degrade(data, sigma, seed=0, tau=25):
    """Make the clean sample tracking look like automatic tracking: add a slowly drifting
    ball offset (std `sigma` metres per axis, correlation time ~`tau` frames). The amount is
    chosen to match the nearest-player-distance profile of the provided match's *inputs*;
    no labels are involved."""
    rng = np.random.default_rng(seed)
    n = len(data["bx"])
    k = np.exp(-0.5 * (np.arange(-3 * tau, 3 * tau + 1) / tau) ** 2)
    out = dict(data)
    for key in ("bx", "by"):
        z = np.convolve(rng.standard_normal(n + len(k)), k, "same")[: n]
        z = z / z.std() * sigma
        out[key] = data[key] + z
    return out
