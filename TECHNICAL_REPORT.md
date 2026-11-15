# More than a Hack 2027 — Phase 1: Ball possession from automatic tracking

**Team:** La Masia — Arth Agrawal, Adya Goenka, Sanika Gumaste, Monish Nambirajan Sunitha
**Match:** FC Barcelona (H) vs Athletic Club (A), tactical camera, 199,319 frames at 25 fps
**Output:** one label per frame — `H`, `A`, `D` (dead ball), and `X` only outside the graded halves
**Reproduce:** `python -m mtah.run --out evaluation/results/submission.csv` (numpy + pandas only)

## 1. The approach in one paragraph

spellF1 rewards recovering *whole possessions*, so we treat the task as segmentation, not
per-frame classification. Stage 1 decides when the ball is in play, using signals in the
automatic tracking that behave like stoppages. Stage 2 scores every live frame with a learned
estimate of "which team has the ball", including who has just touched the ball. The model is
gradient-boosted trees trained only on *other* matches whose tracking has been degraded to look
like ours. Stage 3 decodes those scores
inside each stretch of live play with a two-state Viterbi that charges a fixed cost per change of
possession. A deflection, a ball flying past an opponent or a few frames of tracking noise cannot
flip the label; sustained evidence for the other team does, at the frame where it turns.

## 2. Assets: what we used, and what we dropped

| Asset | Used? | How / why not |
|---|---|---|
| Automatic tracking (ATD), players + ball | **Yes** | The only input of the pipeline: positions in metres, team of each track slot from the metadata |
| Smart Tagging | Inspected, dropped | Only ~60 set pieces and coarse multi-second phases; too coarse to place possession changes |
| Homography | Dropped | Only needed to work in video pixels; we never use the video |
| Tactical / broadcast video | Not used | Not needed by the method; also avoids any risk around the "no checking against the video" rule |
| Broadcast tracking | Dropped | Not synchronised to the submission timeline |
| Public 5-minute ground truth | Pipeline check only | Scored once per frozen version, after all choices were made (Section 6) |

**Observations on the inputs that shaped the method** (statistics of inputs only, no labels):

* **Ball dropouts are bimodal.** Missing-ball gaps last either ≤ 22 frames (< 1 s) or ≥ 122 frames
  (≈ 5 s), never anything in between. The long gaps coincide with stoppages: mean player speed falls
  from 2.2 to 1.2 m/s, and their durations match throw-ins, goal kicks and fouls.
* **Dead balls are visible while the ball is tracked too.** The ball sits 0.3–0.9 m beyond a
  touchline for seconds (throw-in set-up), or lies still with nobody near it (free-kick set-up).
* **The automatic ball is offset from the players.** The nearest player is a median 1.9 m away,
  against 0.25 m in hand-corrected tracking. Any model learned on clean data must therefore be
  trained on degraded data (Section 5).
* **Track slots switch identity but never team.** We only ever need "nearest home" and "nearest
  away" player, so identity switches inside a team are harmless.

## 3. Method

### 3.1 Ball in play (`alive_mask`)

1. Ball missing ≤ 2 s → still in play (tracking blip). Missing longer → stoppage.
2. Exception: a long dropout where players average > 2 m/s during the gap *and* for 2 s after it,
   with the ball last seen > 3 m from every line, is play continuing out of sight. On the external
   games this pattern flags only 2 of 156 real stoppages.
3. Ball > 0.3 m beyond a line for ≥ 3 frames → out. It stays dead, with hysteresis, until it is
   0.5 m back inside the field.
4. Ball motionless (< 0.8 m/s) for ≥ 2 s with nobody within 3 m, or with players on average
   walking (< 1.2 m/s) → ball placed for a restart.
5. Live stretches under 2 s are discarded. The external matches have only 1–3 per match.

### 3.2 Learned possession evidence (`features.py`, `learned_evidence`)

For every frame with a visible ball we build 29 features, mirrored for both teams:
* distance from the ball to the nearest and second-nearest player of each team
* speed of the ball relative to each team's nearest player (a ball moving *with* a player is
  being carried; one flying past him is not)
* number of each team's players within 3 m
* each team's minimum distance to the ball over ±0.25 s and ±1 s
* ball speed and acceleration
* **touches:** a kick or a trap shows up as a sudden change of ball velocity between 0.12 s
  before and 0.12 s after. For each team, the strongest touch within ±0.2 s and ±0.6 s,
  weighted by how close that team's nearest player was at the moment.
* **on-the-ball history:** for each team, the time since one of its players was last within
  1.5 m of the ball, and the time until one next is (capped at 5 s)
* **carrier movement**, for settling duels: whether each team's nearest player runs *with* the
  ball's direction (a dribbler) or against it (a challenging defender), how far the ball is in
  front of him, and his speed

A gradient-boosted tree model (200 trees, depth 4) predicts P(home has the ball). Training uses
every frame twice, once as is and once with the teams swapped and the label flipped. At
prediction time we average both views, so the model has no home/away bias by construction. The
trees are exported to a plain array file (`mtah/evidence_model.npz`, 230 KB) and evaluated in
numpy (`mtah/gbtrees.py`). The pipeline therefore needs no machine-learning library, and the
output does not depend on any library version. The export matches scikit-learn's predictions to
2·10⁻¹⁶.

### 3.3 Decoding (`viterbi2`)

Within every live stretch, the H/A sequence maximises
`Σ ±½·logit P(home) − 28 · (number of changes)`. The switch cost of 28 log-likelihood units
sits in the middle of a broad plateau (24–32) in the held-out tests. Decoding is global, so a
change lands where the evidence turns, not after a confirmation delay; transition timing matters
for the IoU-based spell matching. Frames with no visible ball carry no evidence, so the current
owner keeps the ball through them.

## 4. Possession rules we implemented

* **Loose balls:** with no evidence, for example a ball in flight or not visible, the team that last
  had the ball keeps it, as in the challenge definition. A change needs sustained evidence for the
  other team, which in practice means control.
* **Challenges and deflections:** a single touch, a deflected pass or a won header that is not
  followed by control produces only brief evidence, which costs less than a switch. No change.
* **Interceptions:** the receiving team's evidence quickly outweighs the switch cost, so the change
  is placed at the moment of control.
* **Restarts:** every stoppage ends a possession. The team of the next live stretch is decided by
  the evidence after the restart, typically the taker and his first receiver, so throw-ins, goal
  kicks, free kicks and kick-offs get their owner automatically. No rule assumes who restarts.
* **Dead ball:** always `D`. We never assign dead-ball ownership, since the ground truth discards it.
* **`X`:** written only outside the two graded halves (before kick-off, halftime, after the final
  whistle), where nothing is scored. Inside the halves every frame gets H, A or D. An X there would
  be counted as a miss under macroF1, and under spellF1 it would be filled with the previous label.

## 5. Training data and calibration — other matches only

The organisers' clarification forbids training, fitting or tuning on the ground truth or the
video of the provided match, and RULES.md allows human-labelled data from other games.

* **Source:** Metrica Sports public sample games 1, 2 and 3 (github.com/metrica-sports/sample-data).
  These are three anonymised professional matches with corrected tracking and event data produced
  by Metrica Sports. We use them as published, with credit to the source as Metrica requests.
* **Labels:** derived by our code from their event data (`mtah/external.py`). On-ball events
  (pass, carry, shot, set piece, recovery, ball lost) give the team in possession from their start
  frame; a ball out, goal or foul starts a dead ball that lasts until the next set piece. This gives
  1,144 possessions over about 286 minutes.
* **Matching the domain:** their clean tracking is degraded to resemble the provided automatic
  tracking, using input statistics only. We add a slowly drifting ball offset (σ = 1.1 m,
  correlation time ≈ 1 s), chosen so the nearest-player distance profile matches ours
  (p10/25/50 = 0.44/0.89/1.91 m), and random player dropouts so that about 93% of player-frames
  remain, like the ~10.2 of 11 players per team visible here.
* **Training set:** 3 games × 2 noise draws × both team views = 1,052,668 frames
  (`calibration/train_evidence.py`).
* **Hand-set constants:** the stage-1 constants come from football and physics reasoning stated
  in `mtah/possession.py`, and the bimodal dropout structure makes several of them insensitive (any
  gap threshold between 23 and 121 frames gives identical output).

## 6. Results

**Held-out test (`calibration/cross_validate.py`).** Train on two external games, test on the
third (all three ways), each on two noise draws never used in training; 6 tests in total:

| Method | spellF1 | macroF1 | Wins vs previous |
|---|---|---|---|
| Hand-made distance rule + decoder | 0.453 | 0.832 | — |
| Learned evidence, proximity features only | 0.504 | 0.853 | 6 of 6 |
| + touch features | 0.541 | 0.863 | 6 of 6 |
| **+ carrier-movement features (submitted)** | **0.550** | **0.866** | 5 of 6; confirmed on 9 fresh noise draws (+0.008) |

The switch cost was chosen in the middle of the plateau (16–32 all within 0.002). We also tested
these ideas under the same protocol and **dropped** them, because none beat the noise:

| Tried and dropped | Effect on held-out spellF1 |
|---|---|
| "Whistle" detection (all players slowing down together → dead ball) | +0.001 |
| Larger tree model (300–400 trees, depth 6) | −0.000 to +0.002 |
| Training on mixed noise levels (σ = 0.8 and 1.4) for robustness | ±0.007 under noise mismatch, 0.000 at the expected level |
| Team-shape features (spread, pressing toward the ball) | −0.014; fits game-specific habits |
| Longer context (±2 s minimum distance) | −0.004 |
| Learned classifier for "is this ball dropout play or a stoppage?" | +0.000; not better than calling every long dropout a stoppage |

**Public 5 minutes**, scored once per frozen version, as a pipeline check only:

| Version | spellF1 | macroF1 | Possessions (true 17) | Possession-share error |
|---|---|---|---|---|
| Distance rule | 0.497 | 0.719 | 16 | 0.049 |
| Learned, proximity only | 0.465 | 0.717 | 20 | 0.015 |
| Learned + touch | 0.480 | 0.712 | 18 | 0.018 |
| **Learned + touch + carrier movement (submitted)** | 0.491 | 0.712 | 18 | 0.020 |
| Best published baseline (smooth125) | 0.424 | 0.581 | 16 | 0.052 |

The official scorer reports that, with only 17 possessions in this window, gaps below about 0.34
spellF1 are within noise, so these versions are indistinguishable here. We chose the
submitted version on the held-out external test (about 1,140 possessions), as decided before any
public check, and changed nothing after seeing it.

**Full-match sanity checks (no labels):**

| | Submitted | External reference matches |
|---|---|---|
| Possessions | 301 (H 145 / A 156) | 360–403 |
| Mean / median duration | 11.6 s / 7.0 s | 9.3–10.2 s mean |
| Live time | 58.4 of 106 graded min | 59–61 of ~95 min |
| Barcelona share of live ball | 62.1 % | — |

On held-out external games the same model also finds about 70% of the true possession count: it
leaves out very short possessions that it cannot place confidently, which is what the IoU-based
metric rewards. The lower count here is that same behaviour, not a sign of failure.

## 7. Rule compliance

* No hand labels or hand corrections of any kind on the provided match.
* Nothing trained, fitted or tuned on its ground truth or its video. The video is not used at all.
  The public window was used only as the post-freeze pipeline check above.
* Human-produced data comes only from other matches (Metrica's public sample games), as RULES.md
  allows.
* `python -m mtah.run` regenerates the submitted file byte for byte from the provided data.
  Verified from scratch on a clean copy of the repository. No match
  data is committed.

## 8. Limitations and what we would do next

* **Domain gap:** the model learns on degraded clean tracking, not real automatic tracking. A better
  noise model (identity switches, ball false positives) should help.
* **No ball height:** a high ball over a player still produces some evidence.
* **Fouls while the ball is still visible and moving** are not detected as stoppages. On the
  external games this is only 1–7% of dead-ball time, and a "whistle" rule did not help.
* **Play continuing while the ball is out of sight** is mostly labelled dead. About 1 in 7 long
  dropouts is really play, but neither our rule nor a learned classifier can tell them apart
  reliably from player movement alone. Detecting the ball in the video would address this.
* **Our own ball detection on the video**, projected with the homography, could fill the rare
  in-play gaps.

## 9. Reproduce

    pip install -r requirements.txt                    # numpy, pandas (pinned)
    # put the match data in ./data (see DATA.md)
    python -m mtah.run --out evaluation/results/submission.csv
    python evaluation/validate_submission.py evaluation/results/submission.csv

    # optional: retrain / re-check the model on the external games
    git clone https://github.com/metrica-sports/sample-data ../ext
    pip install -r calibration/requirements.txt
    python calibration/train_evidence.py               # rebuilds mtah/evidence_model.npz
    python calibration/cross_validate.py               # 3-way held-out comparison (Section 6)
    python calibration/calibrate.py                    # noise fit + distance-rule grid

    # optional: watch the model at work
    pip install -r tools/requirements.txt
    python tools/replay.py --time 36:00 --seconds 40 --live
