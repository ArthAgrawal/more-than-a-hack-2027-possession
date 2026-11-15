"""Frame-level ball possession (H / A / D) from automatic tracking.

Two stages, both deterministic, no learning and no labels from the provided match:

  1. alive_mask()  -- is the ball in play? Built from the ball's visibility, the ball leaving the
                      pitch, a motionless ball awaiting a restart, and the players' running speed.
  2. possession()  -- within each stretch of live play, which team has the ball? Per-frame
                      evidence (by default learned on other matches, see learned_evidence;
                      or the hand-made "clearly closest team" rule) is decoded with a
                      two-state Viterbi that charges a fixed penalty per change of possession.
                      This keeps each possession in one piece: a deflection or a ball flying
                      past an opponent cannot flip the label, while sustained control does.

Every constant in P is set either from football / physics reasoning stated next to it or by
calibration on *other* matches (Metrica Sports public sample games, see mtah/external.py and
calibration/). None is fitted to the ground truth or the video of the provided match.
"""
import warnings

import numpy as np

FPS = 25
L, W = 105.3, 68.0  # pitch size in metres (games.py)
H, A, D = 0, 1, 2

P = dict(
    # ---------------- alive / dead ----------------
    # A ball dropout shorter than this is a tracking blip in play; longer is a stoppage. The
    # provided match has no dropout between 23 and 121 frames, so any value in that range gives
    # the same output; 2 s is a natural "brief interruption" bound.
    max_blip=2 * FPS,
    # Long dropouts during which the game visibly carries on are not stoppages: mean player
    # speed above this (m/s) both during the gap and in the `after_len` frames after it, with the
    # ball last seen at least `gap_edge` m from every line. On the external games this pattern
    # flags 2 of 156 real stoppages.
    gap_run_speed=2.0,
    after_len=2 * FPS,
    gap_edge=3.0,
    # Ball beyond a line by more than this (m) is out of play: ball radius 0.11 m plus tracking
    # noise. The reading must last `min_out` frames (single-frame glitches are ignored), and the
    # ball then stays dead until it is `back_in` m back inside the field (throw-in set-up).
    out_margin=0.30,
    min_out=3,
    back_in=0.5,
    # A motionless ball (smoothed speed below `static_speed` m/s for `static_len` frames) with
    # nobody within `static_alone` m, or with the players on average walking (below
    # `static_walk` m/s), is a ball placed for a restart. A live ball is not left lying still.
    static_speed=0.8,
    static_len=2 * FPS,
    static_alone=3.0,
    static_walk=1.2,
    # Live stretches shorter than this are flicker of the dead-ball signal. In the external
    # reference games only 1-3 of ~95 live stretches per match last under 2 s.
    min_alive=2 * FPS,
    # ---------------- possession ----------------
    # Evidence is full while the nearest player is within `ctrl_radius` m of the ball and fades
    # to zero at twice that distance (ball travelling between players: no evidence).
    ctrl_radius=2.5,
    # A distance advantage of this many metres over the nearest opponent is clear evidence.
    ev_scale=2.0,
    # A ball flying past a player is weak evidence: evidence is halved at this ball speed (m/s).
    fast_ball=20.0,
    # Cost of one change of possession, in frames of clear evidence (~1.2 s).
    lam=30.0,
    # ---------------- learned evidence (default) ----------------
    # "learned": per-frame log-odds from gradient-boosted trees trained on other matches
    # (calibration/train_evidence.py); "distance": the hand-made rule above.
    evidence="learned",
    # Switch penalty for the learned evidence, in log-likelihood units. Middle of the plateau
    # (24-32) in leave-one-game-out tests on the external games.
    lam_learned=28.0,
)
# ctrl_radius, ev_scale, fast_ball and lam were chosen by grid search on the external games
# after degrading their clean ball positions to match the noise profile of the provided
# tracking (calibration/calibrate.py).


# ---------------------------------------------------------------- helpers
def _runs(mask):
    """Start and stop indices of the True runs of a boolean array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def _smooth(a, k):
    """NaN-aware centred moving average of width k."""
    ok = np.isfinite(a)
    num = np.convolve(np.where(ok, a, 0.0), np.ones(k), "same")
    den = np.convolve(ok.astype(float), np.ones(k), "same")
    out = num / np.maximum(den, 1e-9)
    out[den == 0] = np.nan
    return out


def _velocity(x):
    """Central-difference velocity (units per second) along axis 0."""
    v = np.full_like(x, np.nan)
    v[1:-1] = (x[2:] - x[:-2]) * (FPS / 2.0)
    return v


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanmedian(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a)


def team_speed(px, py):
    """Mean speed (m/s) of all tracked players per frame, from positions."""
    v = np.full_like(px, np.nan)
    v[1:-1] = np.hypot(px[2:] - px[:-2], py[2:] - py[:-2]) * (FPS / 2.0)
    return _nanmean(v, axis=1)


def nearest_distance(px, py, bx, by):
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    near = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    near[~np.isfinite(near)] = np.nan
    return dist, near


# ---------------------------------------------------------------- stage 1
def alive_mask(data, graded, p=P):
    px, py, ps, bx, by = (data[k] for k in ("px", "py", "ps", "bx", "by"))
    n = len(bx)
    present = np.isfinite(bx)
    _, nearest = nearest_distance(px, py, bx, by)
    alive = present.copy()

    # 1. short dropouts are blips in play
    s, e = _runs(~present)
    for a, b in zip(s, e):
        if b - a <= p["max_blip"] and a > 0 and b < n:
            alive[a:b] = True

    # 2. long dropouts while the game visibly carries on
    tsp = team_speed(px, py)
    for a, b in zip(s, e):
        if b - a > p["max_blip"] and a > 0 and b + p["after_len"] < n:
            edge = min(bx[a - 1], L - bx[a - 1], by[a - 1], W - by[a - 1])
            if (_nanmean(tsp[a:b]) > p["gap_run_speed"]
                    and _nanmean(tsp[b:b + p["after_len"]]) > p["gap_run_speed"]
                    and edge > p["gap_edge"]):
                alive[a:b] = True

    # 3. ball out of the pitch, dead until clearly back in
    m, h = p["out_margin"], p["back_in"]
    out = present & ((bx < -m) | (bx > L + m) | (by < -m) | (by > W + m))
    near_line = present & ((bx < h) | (bx > L - h) | (by < h) | (by > W - h))
    ns, ne = _runs(near_line)
    s, e = _runs(out)
    for a, b in zip(s, e):
        if b - a >= p["min_out"]:
            i = np.searchsorted(ns, a, side="right") - 1
            if i >= 0 and ne[i] >= b:
                a, b = ns[i], ne[i]
            alive[a:b] = False

    # 4. motionless ball awaiting a restart
    speed = np.hypot(_velocity(_smooth(bx, 5)), _velocity(_smooth(by, 5)))
    speed = _smooth(speed, 11)
    walk = _nanmean(ps, axis=1)
    s, e = _runs(present & (speed < p["static_speed"]))
    for a, b in zip(s, e):
        if b - a >= p["static_len"]:
            alone = _nanmedian(nearest[a:b]) >= p["static_alone"]
            walking = _nanmean(walk[a:b]) < p["static_walk"]
            if alone or walking:
                alive[a:b] = False

    # 5. drop live flicker
    s, e = _runs(alive)
    for a, b in zip(s, e):
        if b - a < p["min_alive"]:
            alive[a:b] = False
    return alive & graded


# ---------------------------------------------------------------- stage 2
def evidence(data, p=P):
    """Per-frame evidence in [-1, 1]: + home has the ball, - away, 0 no information."""
    px, py, bx, by, team = (data[k] for k in ("px", "py", "bx", "by", "team"))
    dist, _ = nearest_distance(px, py, bx, by)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    dh = dist[:, team == 0].min(1)
    da = dist[:, team == 1].min(1)
    w = np.clip((2 * p["ctrl_radius"] - np.minimum(dh, da)) / p["ctrl_radius"], 0.0, 1.0)
    dh = np.where(np.isfinite(dh), dh, 50.0)
    da = np.where(np.isfinite(da), da, 50.0)
    s = np.tanh((da - dh) / p["ev_scale"]) * w
    if p["fast_ball"]:
        bsp = np.hypot(_velocity(_smooth(bx, 3)), _velocity(_smooth(by, 3)))
        s = s / (1.0 + (np.nan_to_num(bsp) / p["fast_ball"]) ** 2)
    s[~np.isfinite(bx)] = 0.0
    return s


def viterbi2(s, lam):
    """Two-state (home / away) labelling maximising sum(+-s) - lam * (number of switches)."""
    n = len(s)
    back = np.zeros((n, 2), np.int8)
    vh, va = s[0], -s[0]
    for t in range(1, n):
        h_stay, h_sw = vh, va - lam
        a_stay, a_sw = va, vh - lam
        back[t, 0] = 0 if h_stay >= h_sw else 1
        back[t, 1] = 1 if a_stay >= a_sw else 0
        vh = max(h_stay, h_sw) + s[t]
        va = max(a_stay, a_sw) - s[t]
    out = np.empty(n, np.int8)
    out[-1] = 0 if vh >= va else 1
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def learned_evidence(data):
    """Per-frame half log-odds that home has the ball, from the exported tree model.

    Each frame is scored from both sides (as is, and with home/away swapped) and the two
    probabilities averaged, so the evidence is exactly symmetric between the teams.
    """
    from mtah import features, gbtrees
    model = gbtrees.load()
    X = features.build(data)
    ok = np.isfinite(X[:, 0])
    s = np.zeros(len(X))
    if ok.any():
        p_home = gbtrees.predict_proba(model, X[ok])
        p_home_swapped = 1.0 - gbtrees.predict_proba(model, features.swap(X[ok]))
        p = np.clip((p_home + p_home_swapped) / 2, 1e-4, 1 - 1e-4)
        s[ok] = 0.5 * np.log(p / (1 - p))
    return s


def possession(data, graded, p=P, alive=None):
    """Return (labels 0/1/2 per frame, alive mask)."""
    if alive is None:
        alive = alive_mask(data, graded, p)
    if p.get("evidence") == "learned":
        s, lam = learned_evidence(data), p["lam_learned"]
    else:
        s, lam = evidence(data, p), p["lam"]
    lab = np.full(len(s), D, np.int8)
    for a, b in zip(*_runs(alive)):
        lab[a:b] = viterbi2(s[a:b], lam)
    return lab, alive
