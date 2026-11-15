"""Load the raw ATD tracking into numpy arrays (cached as .npz, never committed)."""
import os, sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, GAME, require  # noqa: E402

PITCH = GAMES[GAME]["pitch"]  # metres (105.3, 68.0)
N = GAMES[GAME]["n_video_frames"]
CACHE = os.path.join(ROOT, "cache")


def slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)


def parse_raw(raw_path, n_slots):
    px = np.full((N, n_slots), np.nan, np.float32)
    py = np.full((N, n_slots), np.nan, np.float32)
    ps = np.full((N, n_slots), np.nan, np.float32)
    bx = np.full(N, np.nan, np.float32)
    by = np.full(N, np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            groups = parts[1].split(";")
            for k, g in enumerate(groups):
                if g[0] == "N":
                    continue
                a, b, c = g.split(",")
                px[f, k] = float(a); py[f, k] = float(b)
                ps[f, k] = float(c) if c[0] != "N" else np.nan
            ball = parts[2]
            if ball[0] != "N":
                a, b = ball.split(",")[:2]
                bx[f] = float(a); by[f] = float(b)
    L, W = PITCH
    return px * L, py * W, ps, bx * L, by * W


def load(source="tactical"):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, source + "_tracking.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    teams = slot_teams(meta)
    px, py, ps, bx, by = parse_raw(raw, len(teams))
    out = dict(px=px, py=py, ps=ps, bx=bx, by=by, team=teams)
    np.savez(path, **out)
    return out
