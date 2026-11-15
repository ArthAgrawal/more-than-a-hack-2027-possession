"""Watch the model at work: a top-down replay of the tracking with the model's live call.

    python tools/replay.py --start 54000 --seconds 40          # writes cache/replay.mp4

Optional extra, not part of the submission pipeline. Needs matplotlib and ffmpeg on the PATH.
The output goes to cache/ (git-ignored): a rendered replay is derived from the licensed tracking
data, so keep it private -- do not commit or publish it.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.load import load  # noqa: E402
from mtah.possession import possession, L, W, FPS  # noqa: E402
from mtah.run import graded_mask  # noqa: E402

HOME, AWAY, DEAD = "#a50044", "#2b6cb0", "#8a8a8a"
CALLS = {0: ("BARÇA BALL", HOME), 1: ("ATHLETIC BALL", AWAY), 2: ("BALL DEAD", DEAD)}


def draw_pitch(ax):
    ax.add_patch(Rectangle((-3, -3), L + 6, W + 6, color="#1f5a2e", zorder=0))
    line = dict(color="white", lw=1.4, alpha=0.8, fill=False)
    ax.add_patch(Rectangle((0, 0), L, W, **line))
    ax.plot([L / 2, L / 2], [0, W], color="white", lw=1.4, alpha=0.8)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **line))
    for left in (True, False):
        for depth, width in ((16.5, 40.32), (5.5, 18.32)):
            x = 0 if left else L - depth
            ax.add_patch(Rectangle((x, W / 2 - width / 2), depth, width, **line))
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(W + 3, -3)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=54000, help="first video frame")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "cache", "replay.mp4"))
    args = ap.parse_args()

    data = load()
    labels, _ = possession(data, graded_mask())
    start, stop = args.start, args.start + int(args.seconds * FPS)
    team = data["team"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig = plt.figure(figsize=(12.8, 9.6), dpi=100, facecolor="#101418")
    ax = fig.add_axes([0.04, 0.17, 0.92, 0.72])
    bar = fig.add_axes([0.04, 0.07, 0.92, 0.05])
    draw_pitch(ax)
    home = ax.scatter([], [], s=150, c=HOME, edgecolors="white", linewidths=1.2, zorder=3)
    away = ax.scatter([], [], s=150, c=AWAY, edgecolors="white", linewidths=1.2, zorder=3)
    ball = ax.scatter([], [], s=130, c="#ffd400", edgecolors="black", linewidths=1, zorder=5)
    banner = fig.text(0.5, 0.935, "", ha="center", va="center", fontsize=30, weight="bold",
                      color="white", bbox=dict(boxstyle="round,pad=0.4", fc=DEAD, ec="none"))
    clock = fig.text(0.96, 0.935, "", ha="right", va="center", fontsize=16, color="#cbd5e0",
                     family="monospace")
    lost = fig.text(0.5, 0.53, "", ha="center", fontsize=14, color="#ffd400")
    fig.text(0.04, 0.935, "Model output (no human labels)", va="center", fontsize=12,
             color="#cbd5e0")

    # the whole clip's labels as a coloured bar, with a moving playhead
    seg = labels[start:stop]
    cuts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        bar.add_patch(Rectangle((a, 0), b - a, 1, color=CALLS[int(seg[a])][1]))
    bar.set_xlim(0, len(seg))
    bar.set_ylim(0, 1)
    bar.axis("off")
    head = bar.axvline(0, color="white", lw=2.5)
    for x, (name, colour) in zip((0.04, 0.10, 0.175),
                                 (("Barça", HOME), ("Athletic", AWAY), ("dead ball", DEAD))):
        fig.text(x, 0.035, name, color=colour, fontsize=12, weight="bold")

    w, h = fig.canvas.get_width_height()
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s", "%dx%d" % (w, h), "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "23", args.out],
        stdin=subprocess.PIPE)
    for f in range(start, stop):
        px, py = data["px"][f], data["py"][f]
        ok = np.isfinite(px)
        home.set_offsets(np.c_[px[ok & (team == 0)], py[ok & (team == 0)]])
        away.set_offsets(np.c_[px[ok & (team == 1)], py[ok & (team == 1)]])
        if np.isfinite(data["bx"][f]):
            ball.set_offsets([[data["bx"][f], data["by"][f]]])
            lost.set_text("")
        else:
            ball.set_offsets(np.empty((0, 2)))
            lost.set_text("ball not visible to tracking")
        text, colour = CALLS[int(labels[f])]
        banner.set_text(text)
        banner.get_bbox_patch().set_facecolor(colour)
        sec = f / FPS
        clock.set_text("video %02d:%05.2f" % (sec // 60, sec % 60))
        head.set_xdata([f - start, f - start])
        fig.canvas.draw()
        ffmpeg.stdin.write(fig.canvas.buffer_rgba().tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
