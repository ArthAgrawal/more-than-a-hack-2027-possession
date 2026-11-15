"""Per-frame features for the learned possession evidence.

Every team feature comes in a home version and an away version, built the same way. Swapping them
(`swap`) gives the same frame seen from the other team, which lets us train on both views and
keeps the model free of any home/away bias.

Layout (29 columns): 6 proximity features per team, 2 ball features, 4 touch features per team,
1 touch-strength feature, 3 carrier-movement features per team. See NAMES.
"""
import numpy as np

from mtah.possession import _smooth, _velocity

BIG = 30.0      # stand-in distance (m) when a team has nobody tracked
CLOSE = 1.5     # "on the ball" distance (m) for the since/until-close features
CAP = 125       # cap (frames, 5 s) for the since/until-close features
TOUCH_DV = 30.0 # velocity change (m/s) that counts as a full-strength touch


def _roll_min(a, k):
    """Centred rolling minimum over 2k+1 frames."""
    out = a.copy()
    for s in range(1, k + 1):
        out[s:] = np.minimum(out[s:], a[:-s])
        out[:-s] = np.minimum(out[:-s], a[s:])
    return out


def _roll_max(a, k):
    """Centred rolling maximum over 2k+1 frames."""
    out = a.copy()
    for s in range(1, k + 1):
        out[s:] = np.maximum(out[s:], a[:-s])
        out[:-s] = np.maximum(out[:-s], a[s:])
    return out


def _since(mask, cap):
    """Frames since `mask` was last true (looking backwards), capped at `cap`."""
    out = np.full(len(mask), cap, np.float32)
    idx = np.flatnonzero(mask)
    pos = np.searchsorted(idx, np.arange(len(mask)), side="right") - 1
    ok = pos >= 0
    out[ok] = np.minimum(np.arange(len(mask))[ok] - idx[pos[ok]], cap)
    return out


def _shift(a, k):
    """a[t - k] (k > 0 looks back, k < 0 looks ahead), NaN outside."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def team_block(dist, rel, mask):
    """Nearest and 2nd-nearest distance, relative speed of the nearest, players within 3 m."""
    d = np.where(mask[None, :] & np.isfinite(dist), dist, np.inf)
    order = np.argsort(d, axis=1)[:, :2]
    rows = np.arange(len(d))
    d1 = d[rows, order[:, 0]]
    d2 = d[rows, order[:, 1]]
    r1 = rel[rows, order[:, 0]]
    d1 = np.where(np.isfinite(d1), d1, BIG)
    d2 = np.where(np.isfinite(d2), d2, BIG)
    r1 = np.where(np.isfinite(r1), r1, 10.0)
    n3 = (d < 3.0).sum(1)
    return d1, d2, r1, n3


def touch_block(d1, touch):
    """Touch features for one team.

    t5 / t15: strongest ball touch (sudden velocity change) within +-0.2 s / +-0.6 s, weighted by
    how close this team's nearest player was to the ball at that frame.
    back / fwd: frames since this team last had a player within CLOSE of the ball, and frames
    until it next does (both capped at 5 s).
    """
    near = np.exp(-d1 / CLOSE)
    t5 = _roll_max(touch * near, 5)
    t15 = _roll_max(touch * near, 15)
    close = d1 < CLOSE
    back = _since(close, CAP)
    fwd = _since(close[::-1], CAP)[::-1]
    return [t5, t15, back, fwd]


def carrier_block(px, py, pvx, pvy, bx, by, bvx, bvy, mask):
    """How the team's nearest player moves relative to the ball.

    align: cosine between his running direction and the ball's direction (a dribbler runs with
    the ball; a challenging defender usually moves against it). ahead: how far the ball is in front
    of him along his running direction (m). speed: his running speed (m/s).
    """
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(mask[None, :] & np.isfinite(dist), dist, np.inf)
    j = np.argmin(d, axis=1)
    rows = np.arange(len(bx))
    ok = np.isfinite(d[rows, j])
    vx, vy = pvx[rows, j], pvy[rows, j]
    sp = np.hypot(vx, vy)
    bsp = np.nan_to_num(np.hypot(bvx, bvy))
    moving = (sp > 0.5) & (bsp > 0.5)
    align = np.where(moving, (vx * bvx + vy * bvy) / (sp * bsp + 1e-9), 0.0)
    ux, uy = vx / (sp + 1e-9), vy / (sp + 1e-9)
    ahead = (bx - px[rows, j]) * ux + (by - py[rows, j]) * uy
    ahead = np.clip(np.nan_to_num(np.where(sp > 0.5, ahead, 0.0)), -3.0, 3.0)
    return [np.where(ok, np.nan_to_num(align), 0.0), np.where(ok, ahead, 0.0),
            np.where(ok, np.minimum(sp, 10.0), 0.0)]


def build(data):
    """(n_frames, 29) feature matrix; rows without a visible ball are NaN."""
    px, py, bx, by, team = (data[k] for k in ("px", "py", "bx", "by", "team"))
    sbx, sby = _smooth(bx, 3), _smooth(by, 3)
    bvx, bvy = _velocity(sbx), _velocity(sby)
    bsp = np.nan_to_num(np.hypot(bvx, bvy))
    bacc = np.nan_to_num(np.abs(_velocity(_smooth(np.hypot(bvx, bvy), 3))))
    spx = np.apply_along_axis(lambda c: _smooth(c, 3), 0, px)
    spy = np.apply_along_axis(lambda c: _smooth(c, 3), 0, py)
    pvx = np.nan_to_num(_velocity(spx))
    pvy = np.nan_to_num(_velocity(spy))
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    rel = np.hypot(bvx[:, None] - pvx, bvy[:, None] - pvy)

    h = team_block(dist, rel, team == 0)
    a = team_block(dist, rel, team == 1)
    cols = []
    for d1, d2, r1, n3 in (h, a):
        cols += [d1, d2, np.minimum(r1, 15.0), n3, _roll_min(d1, 6), _roll_min(d1, 25)]
    cols += [np.minimum(bsp, 40.0), np.minimum(bacc, 300.0)]

    # touches: change of ball velocity between 0.12 s before and 0.12 s after
    dv = np.hypot(_shift(bvx, -3) - _shift(bvx, 3), _shift(bvy, -3) - _shift(bvy, 3))
    touch = np.minimum(np.nan_to_num(dv), TOUCH_DV) / TOUCH_DV
    cols += touch_block(h[0], touch) + touch_block(a[0], touch) + [touch]

    # carrier movement (player velocities smoothed over 0.2 s)
    spx5 = np.apply_along_axis(lambda c: _smooth(c, 5), 0, px)
    spy5 = np.apply_along_axis(lambda c: _smooth(c, 5), 0, py)
    pvx5 = np.nan_to_num(_velocity(spx5))
    pvy5 = np.nan_to_num(_velocity(spy5))
    for side in (team == 0, team == 1):
        cols += carrier_block(px, py, pvx5, pvy5, bx, by, bvx, bvy, side)

    X = np.column_stack(cols).astype(np.float32)
    X[~np.isfinite(bx)] = np.nan
    return X


TEAM_COLS = 6
TOUCH_COLS = 4
CARRIER_COLS = 3
NAMES = (["h_d1", "h_d2", "h_rel1", "h_n3", "h_min6", "h_min25"]
         + ["a_d1", "a_d2", "a_rel1", "a_n3", "a_min6", "a_min25"]
         + ["ball_speed", "ball_acc"]
         + ["h_touch5", "h_touch15", "h_since_close", "h_until_close"]
         + ["a_touch5", "a_touch15", "a_since_close", "a_until_close"]
         + ["touch"]
         + ["h_align", "h_ball_ahead", "h_carrier_speed"]
         + ["a_align", "a_ball_ahead", "a_carrier_speed"])


def swap(X):
    """The same frames with home and away exchanged."""
    k, t, c = TEAM_COLS, TOUCH_COLS, CARRIER_COLS
    o = 2 * k + 2           # start of the touch block
    q = o + 2 * t + 1       # start of the carrier block
    return np.concatenate([X[:, k:2 * k], X[:, :k], X[:, 2 * k:o],
                           X[:, o + t:o + 2 * t], X[:, o:o + t], X[:, o + 2 * t:q],
                           X[:, q + c:q + 2 * c], X[:, q:q + c]], axis=1)
