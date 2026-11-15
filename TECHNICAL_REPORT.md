# More than a Hack 2027 — Phase 1: Ball possession from automatic tracking

**Match:** FC Barcelona (H) vs Athletic Club (A), tactical camera, 199,319 frames at 25 fps
**Output:** one label per frame — `H`, `A`, `D` (dead ball), `X` outside the graded halves
**Code:** `mtah/` in this repository · `python -m mtah.run` regenerates the submission byte for byte

## 1. Idea in one paragraph

The ranking metric rewards recovering *whole possessions*, so we treat the task as segmentation,
not per-frame classification. First we decide when the ball is in play, using signals in the
automatic tracking that behave like stoppages. Then, inside each stretch of live play, we decode
which team has the ball with a two-state Viterbi. Every frame contributes evidence (which team is
clearly closest to the ball), and every change of possession costs a fixed penalty. A deflection,
a ball flying past an opponent or a few frames of tracking noise cannot flip the label. Sustained
control by the other team does, and the change lands where the evidence actually turns.

## 2. What the inputs told us

All of these are statistics of the provided *inputs*; no labels are involved.

* **Ball dropouts are bimodal.** Missing-ball gaps last either ≤ 22 frames (< 1 s) or ≥ 122 frames
  (≈ 5 s), with nothing in between. Short gaps are blips during play. Long gaps coincide with
  stoppages: mean player speed drops from 2.2 to 1.2 m/s, and the gap lengths match throw-ins,
  goal kicks and fouls.
* **Dead balls are also visible while the ball is tracked.** The ball sits 0.3–0.9 m beyond a
  touchline for several seconds (throw-in set-up), or lies motionless with nobody near it
  (free kick set-up).
* **The automatic ball is offset from the players.** The nearest player is a median 1.9 m from the
  tracked ball, against 0.25 m in hand-corrected tracking. A tight "ball at feet" radius therefore
  does not work here, which shaped the possession model and its calibration.
* **Team membership is given.** Each of the 46 track slots belongs to one team in the metadata.
  Slots are not identities, but we only ever need "nearest home" and "nearest away" player, so
  identity switches inside a team are harmless.

## 3. Method

### 3.1 Ball in play (`alive_mask`)

1. Ball missing for ≤ 2 s → still in play (tracking blip). Missing for longer → stoppage.
2. Exception: a long dropout where players average > 2 m/s during the gap *and* in the 2 s after
   it, with the ball last seen > 3 m from every line, is play continuing out of sight.
3. Ball > 0.3 m beyond a line for ≥ 3 frames → out of play. It stays dead, with hysteresis, until it
   is 0.5 m back inside the field.
4. Ball motionless (< 0.8 m/s) for ≥ 2 s with nobody within 3 m, or with players on average
   walking (< 1.2 m/s) → ball placed for a restart.
5. Live stretches under 2 s are discarded as flicker. Real matches almost never have them: 1–3 of
   ~95 per match in the external games.

### 3.2 Who has the ball (`evidence` + `viterbi2`)

For each frame, let `dH` and `dA` be the distances from the ball to the nearest home and away
player. The evidence is

    s = tanh((dA − dH) / 2 m) · w(min(dH, dA)) / (1 + (v_ball / 20 m/s)²)

* `w` is 1 while someone is within 2.5 m of the ball and fades to 0 at 5 m. A ball travelling
  between players carries no information, so the current owner simply keeps it, which matches the
  rule that in loose-ball phases the last team in possession keeps it.
* The speed term discounts a fast ball passing a player, which is not control.

Within every live stretch, a two-state Viterbi picks the H/A sequence that maximises
`Σ ±s − 30 · (number of changes)`. A possession change needs the equivalent of ~1.2 s of clear
evidence for the new team. Because decoding is global, the change is placed at the frame where
the evidence turns, not after a confirmation delay. That keeps transitions on time, which
matters for the IoU-based spell matching. The restart team after a stoppage falls out of the same
decoding.

## 4. Calibration — on other matches only

The organisers' clarification forbids training, fitting or tuning on the ground truth or the
video of the provided match. We therefore set the parameters in two ways:

* **Football and physics reasoning**, stated next to each constant in `mtah/possession.py`. Some
  of these are confirmed by the input structure itself: any dropout threshold between 23 and
  121 frames gives an identical output.
* **External data:** Metrica Sports' public sample games 1 and 2
  (github.com/metrica-sports/sample-data). These have tracking plus human event data for two
  different matches. We derive a reference possession timeline from their events
  (`mtah/external.py`): on-ball events give the team; ball out, goals and fouls start a dead
  ball that lasts until the next set piece. We then score candidate settings with the official
  metric code.

Those sample games use hand-corrected tracking, where the ball is glued to the player's feet,
so tuning on them directly would be misleading. We first **degrade** their ball positions with a
slowly drifting offset (σ = 1.1 m, correlation time ≈ 1 s). The amount is chosen so that the
nearest-player distance profile matches the provided match's inputs (p10/25/50 = 0.44/0.89/1.91 m,
against 0.53/0.94/1.64 m to 0.62/1.08/1.86 m for σ = 1.0 to 1.2). A grid search over control
radius, evidence scale, ball-speed discount and switch penalty, averaged over 2 games × 2 noise
seeds (`calibration/calibrate.py`), gives a broad plateau. We take a setting in the middle of it,
not the single best cell, because the top cells differ by less than the metric's noise band.

| Setting (external games, degraded) | spellF1 | macroF1 |
|---|---|---|
| Our method, chosen parameters | **0.497** | 0.843 |
| Earlier pipeline version: threshold rule (switch after N frames of control), best grid setting | 0.452 | 0.839 |
| Our method on clean (undegraded) tracking | 0.541 | 0.866 |
| Our method with the reference ball-in-play mask (possession error only) | 0.546 | 0.908 |

The oracle row shows that most of the remaining error is in team attribution, not in dead-ball
detection.

**Use of the public 5 minutes.** As the rules permit, we scored the public window once, after
all parameters were frozen, to check that frames are aligned and the file is well formed. Result:
spellF1 0.497 and macroF1 0.719, against 0.424 and 0.621 for the best published baselines. The
number is consistent with the external estimate. Nothing was changed after seeing it.

## 5. Automatic sanity checks on the full match (no labels)

| | Ours | External reference games |
|---|---|---|
| Possessions | 296 (H 146 / A 150) | 360–381 per match |
| Mean / median duration | 11.8 s / 7.6 s | 9.3–10.2 s mean |
| Live time | 58.4 min of 106 graded | 59–61 min of ~95 |
| Live stretches < 2 s | 0 | 1–3 |
| Barcelona share of live ball | 61.8 % | — |

## 6. Rule compliance

* No hand labels or hand corrections of any kind on the provided match.
* No use of its ground truth (beyond the single post-freeze format check above) or its video.
  The video was not used at all.
* No human-produced data about the provided match. Human event data is used only for *other*
  matches (Metrica public sample games), as RULES.md allows. Metrica asks users to credit the
  source, which we do here.
* `python -m mtah.run` regenerates the submitted file exactly from the provided data. No match
  data is committed (`data/` and `cache/` are git-ignored).

## 7. Limitations and next steps

* Team attribution in tight duels is the main error source. Signals we have not used yet: ball
  trajectory endpoints (the receiver of a pass), team shape (which side is pressing), and Smart
  Tagging phases as a weak prior.
* There is no ball height, so a high ball over a player still produces some evidence. The speed
  discount only partly covers this.
* Fouls with the ball still visible and moving are not detected as dead balls.

## 8. Reproduce

    pip install -r requirements.txt            # numpy, pandas
    # put the match data in ./data (see DATA.md)
    python -m mtah.run --out evaluation/results/submission.csv
    python evaluation/validate_submission.py evaluation/results/submission.csv
    # optional: reproduce the calibration
    git clone https://github.com/metrica-sports/sample-data ../ext
    python calibration/calibrate.py
