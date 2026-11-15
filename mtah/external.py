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


def drop_players(data, keep=0.93, seed=0, mean_len=250):
    """Blank out player tracks in random stretches so that about `keep` of player-frames remain,
    like the provided tracking (about 10.2 of 11 players per team visible). No labels involved."""
    rng = np.random.default_rng(seed + 100)
    px, py = data["px"].copy(), data["py"].copy()
    n, k = px.shape
    for j in range(k):
        t = 0
        while t < n:
            t += int(rng.exponential(mean_len * keep / (1 - keep)))
            gap = int(rng.exponential(mean_len))
            px[t:t + gap, j] = np.nan
            py[t:t + gap, j] = np.nan
            t += gap
    out = dict(data)
    out["px"], out["py"] = px, py
    return out


def _subtype_string(sub):
    if sub is None:
        return "nan"
    if isinstance(sub, dict):
        return sub["name"]
    return "-".join(s["name"] for s in sub)


def load_game3():
    """Sample game 3: FIFA EPTS tracking (player set changes with substitutions) + JSON events,
    converted to the same structures as games 1 and 2."""
    import json
    import xml.etree.ElementTree as ET
    base = os.path.join(EXT, "data", "Sample_Game_3", "Sample_Game_3_")
    root = ET.parse(base + "metadata.xml").getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of = {p.get("id"): team_ids.index(p.get("teamId")) for p in root.iter("Player")}
    chan = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    players = sorted(team_of)
    slot = {pid: i for i, pid in enumerate(players)}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [slot[chan[r]] for r in refs[::2]]  # x then y per player
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), np.array(order)))
    with open(base + "tracking.txt") as fh:
        lines = fh.read().splitlines()
    n = max(int(l.split(":", 1)[0]) for l in lines)
    px = np.full((n, len(players)), np.nan)
    py = np.full((n, len(players)), np.nan)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    for line in lines:
        fr, pl, ball = line.split(":")
        f = int(fr)
        order = next(o for s, e, o in specs if s <= f <= e)
        vals = np.array([v.split(",") for v in pl.split(";")], dtype=float)
        px[f - 1, order] = vals[:, 0] * L
        py[f - 1, order] = vals[:, 1] * W
        b = ball.split(",")
        bx[f - 1], by[f - 1] = float(b[0]) * L, float(b[1]) * W
    team = np.array([team_of[p] for p in players], np.int8)
    sp = np.full_like(px, np.nan)
    sp[1:-1] = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) * 12.5
    ev = json.load(open(base + "events.json"))["data"]
    rows = [{"Team": "Home" if e["team"]["id"] == team_ids[0] else "Away",
             "Type": e["type"]["name"], "Subtype": _subtype_string(e["subtypes"]),
             "Start Frame": e["start"]["frame"], "End Frame": e["end"]["frame"] or e["start"]["frame"]}
            for e in ev]
    return dict(px=px, py=py, ps=sp, bx=bx, by=by, team=team), pd.DataFrame(rows)


def load_any(g):
    return load_game3() if g == 3 else load_game(g)
