# Methodology

Where every number this project publishes comes from.

[`ARCHITECTURE.md`](ARCHITECTURE.md) covers the shape of the system — the five
stages and how data moves through them. This covers the *derivations*: each
constant, the formula it sits in, and the measurement that chose it.

It exists because the justifications are correct but scattered. `K = 20` is
defended in a docstring in `features/elo.py`, `fifa_weight: 0.5` in a comment
block in `config/player_rating.yaml`, the learning rate in
`models/match_model.py`. That is the right place to *decide* a number and a poor
place to *revise* from: answering "how is a player rating built" means reading
four files in the right order.

Figures here were re-derived from the committed artifacts, not copied from the
prose. Where a number depends on a sample or a fitted parameter, the sample is
named — a bare percentage reads as a property of football when it is partly a
property of one measurement.

---

## 1. Data

| Source | Gives | Scale |
|---|---|---|
| football-data.co.uk | results, shots, cards, **odds** | 29,143 matches |
| Understat | xG, npxG, xG chain/buildup, PPDA, deep completions | team-match and player-season |
| FBref | player stats: tackles, interceptions, keeper, misc | 65 cached league-seasons |
| EA FC exports | player overall and attributes | 12 editions, one per season |

Five leagues, 2010/11 to 2026/27: `SP1` 6,149 · `E0` 6,130 · `I1` 6,130 ·
`F1` 5,802 · `D1` 4,932.

Derived: **34,695 player-seasons**, of which **23,846** clear the rating gates,
covering **9,784** players; **1,170 team-seasons**.

Odds are present in the data and **excluded from features by name**, with a test.
See §6.

**Entity handling.** Club renames go through `entities/teams.py` aliasing. A
January transfer produces two player-season rows, one per club, because the
rating is keyed to the club he played for. Promoted sides start at Elo `START`
and carry no squad rating until they have had a rated season — the strength join
is a left join, so they arrive as NaN and the booster handles missing values
natively, which is part of why that estimator was chosen.

---

## 2. Leak safety

The rule: a feature for match *M* may depend only on matches before *M*. Three
mechanisms enforce it in different places, and one test proves it.

**Rolling features shift before they roll.** `features/rolling.py`, in
`prev_window`: `grouped.shift(1)` runs *before* the rolling window, so *M*'s
five-match form excludes *M*. The shifted values are then regrouped on the same
keys, so a window never spans two teams.

**Elo records the rating carried in.** `_walk` writes
`pre[(league, date, home, away)] = (home_rating, away_rating)` and only then
applies the update, so a match never sees its own result. Unplayed placeholder
rows sort first within a date, so a European tie being predicted reads the
rating before the real played row updates it.

**Squad strength joins the previous season.** Last season's player ratings are
knowable in August; this season's are not.

**`tests/test_truncation_invariance` is the enforcement.** It picks a cutoff,
rebuilds the entire feature table from data truncated there, and requires every
overlapping row to match the full build exactly. A feature that can see the
future fails the suite without anyone having written a test for it.

That test used to run against an EPL-only file with no `League` column — 11,400
rows of one division where the model trains on five leagues — so the repo's most
load-bearing test was validating a dataset that shared neither the shape nor the
span of the real one, and could not exercise any cross-league path. It now runs
against `team_match_history_all.csv`.

---

## 3. Evaluation: walk-forward

```python
for season in test_seasons:          # pipelines/backtest.py
    train = features[features["Season"] < season]
    # fit, predict `season`, record, advance
```

Train on every prior season, predict one, advance. Twelve test seasons from
1516, **20,013 matches**. Random k-fold would train on 2024 to predict 2018 and
report a score that cannot be reproduced in production.

| Season | n | RPS | Accuracy |
|---|---|---|---|
| 1516 | 1,826 | 0.2059 | 50.4% |
| 1617 | 1,826 | 0.1952 | 56.6% |
| 1718 | 1,826 | 0.1951 | 54.2% |
| 1819 | 1,826 | 0.1964 | 52.5% |
| 1920 | 1,725 | 0.2034 | 52.6% |
| 2021 | 1,826 | 0.2023 | 51.9% |
| 2122 | 1,826 | 0.2003 | 51.9% |
| 2223 | 1,826 | 0.2019 | 53.5% |
| 2324 | 1,752 | **0.1927** | 53.6% |
| 2425 | 1,752 | 0.1973 | 53.8% |
| 2526 | 1,752 | 0.1999 | 51.9% |
| 2627 | 250 | 0.2027 | 49.6% |

1920 and 2021 are the two worst seasons and the reason is covid: matches behind
closed doors, with home advantage suppressed for most of both.

**Temperature scaling** is fitted on out-of-sample predictions accumulated by the
walk itself — free, honest, and far more stable than the most recent season
alone. Gated at `MIN_CALIBRATION_MATCHES = 700`: fitting on one season of 380
matches produced temperatures swinging between 0.89 and 1.82 that did not
transfer, leaving calibration marginally worse than doing nothing.

**Every change is gated by a paired bootstrap** over per-match RPS differences.
Pairing cancels the dominant variance — some fixtures are simply unpredictable —
which is what makes a 0.001 effect resolvable at all. A difference is claimed
only when the 95% interval excludes zero.

---

## 4. The model

Two Poisson `HistGradientBoostingRegressor`s, one per side, predicting goal
*rates*; their output becomes a Dixon-Coles scoreline matrix, and every published
number is a sum over that matrix.

Why this shape rather than classifying H/D/A:

1. **One coherent object.** 1X2, scorelines, over/under and BTTS all come from
   the same matrix, so the site cannot show an outcome probability that
   contradicts its own scoreline table.
2. **Draws.** Hard to classify directly — a classifier learns that predicting a
   draw is never optimal and the class collapses. Derived from a goal
   distribution they fall out at the right rate: predicted 24.6% against an
   actual 25.1%.
3. **Expected goals are a deliverable**, and here they are the native prediction
   rather than a bolt-on.

Poisson loss is the right objective for a count target: it models the conditional
rate, keeps predictions positive, and matches the data-generating process. The
raw model output already *is* λ.

### Hyperparameters

```python
{"loss": "poisson", "learning_rate": 0.02, "max_iter": 300,
 "max_leaf_nodes": 15, "min_samples_leaf": 40,
 "l2_regularization": 1.0, "early_stopping": False, "random_state": 7}
```

`learning_rate × max_iter` is one quantity — the total learning budget — and it
is what matters. Across a 24-point grid scored by a four-fold walk-forward:

```
budget   3.0    4.5    6.0    9.0   12.0   15.0   18.0   27.0   45.0
RPS    .19825 .19801 .19798 .19802 .19817 .19827 .19845 .19889 .19961
```

A flat bottom at 4.5–9 rising monotonically after it. The previous
`0.05 × 300` sat at budget **15**, on the rising side: the model was mildly
**over**-trained, which is the opposite of the expectation for 235 features on
29,143 rows.

`0.02 × 300` is budget 6 — the minimum of that curve, and interior to the grid on
every axis. `0.03 × 150` scored a hair better pooled but sits at the edge of the
iteration axis, which is where a selection artefact hides. The interior point was
taken over the better score deliberately.

Confirmed on the full 20,013-match walk-forward, paired bootstrap against the
previous settings: **RPS 0.2003 → 0.1991**, −0.00124 [−0.00158, −0.00090]. Log
loss 0.9847 → 0.9809, Brier 0.5867 → 0.5838, accuracy 52.69% → 52.99%. All four
moved together, which the three feature experiments that failed the same week did
not.

**`early_stopping=False` is deliberate.** Scikit-learn's early stopping holds out
a *random* validation split, which on time-ordered football data means
validating against the future. The walk-forward is the honest selector, and it is
what chose these.

### Dixon-Coles

Independent Poisson is very nearly right and wrong in one documented way: it
under-predicts low-scoring draws. The correction multiplies the four cells where
both sides score at most once:

```
tau(0,0) = 1 - lambda*mu*rho     tau(0,1) = 1 + lambda*rho
tau(1,0) = 1 + mu*rho            tau(1,1) = 1 - rho
```

Fitted **`rho_ = -0.0566`**, by maximum likelihood on training data only — it is
a model parameter like any other and fitting it on the test set leaks. Negative
rho moves mass into 0-0 and 1-1 and out of 1-0 and 0-1. Tau breaks the
sum-to-one and truncation at 10 goals loses a sliver of tail, and renormalising
fixes both at once.

Draw is the trace, home win the lower triangle, away win the upper. Over/under
sums cells where `i + j > line`; BTTS is `matrix[1:, 1:]`.

**One consequence worth stating, because it looks like a contradiction.** An
outcome sums a triangle; an exact score is one cell. Inside the 0-5 grid a draw
has 6 cells against 15 per win, so 1-1 is the single likeliest score in 67% of
matches while the home side is usually the likeliest *winner*. Even a perfectly
even match at 2.75 total goals gives the draw 0.280 against 0.360 each way — the
draw leads only below about 2.0 total goals, and these leagues run 2.7–2.8. The
highest draw probability in 20,013 backtested matches is 37.5%, and the draw was
the top pick twice; the market made it favourite 7 times in 20,007.

---

## 5. Metrics

**RPS** is the headline. Outcomes are *ordered* — a draw sits between a home and
an away win — so predicting away when home wins should cost more than predicting
a draw. RPS is the mean squared error of the cumulative distribution:

```python
cum_pred = np.cumsum(probs, axis=1)[:, :-1]
cum_obs  = np.cumsum(obs,   axis=1)[:, :-1]
np.mean(np.sum((cum_pred - cum_obs) ** 2, axis=1) / (len(OUTCOMES) - 1))
```

The last cumulative term is dropped because it is 1 for both; dividing by
`r - 1 = 2` normalises to [0, 1].

| | RPS | Log loss | Brier | Accuracy |
|---|---|---|---|---|
| base rate | 0.2299 | 1.0708 | 0.6477 | 44.06% |
| **this model** | **0.1992** | **0.9809** | **0.5839** | **52.94%** |
| bookmakers | 0.1950 | 0.9673 | 0.5747 | 53.85% |

Exact scoreline right **12.8%** of the time.

**Why not accuracy.** Model A says H 51 / D 25 / A 24; model B says H 90 / D 5 /
A 5. Both pick home, so accuracy scores them identically — but if home loses, B
was catastrophically wrong and A was mildly wrong. Accuracy is blind to
confidence, blind to the ordering, and gameable by never predicting a draw.

**Calibration is a separate claim from accuracy** and matters as much for the
site's credibility. A model that says 60% and is right 60% of the time is useful
even if its hit rate is unremarkable; one that says 90% and is right 60% of the
time is actively misleading however good its hit rate looks. Note that the base
rate is perfectly calibrated and has zero discrimination — both words are needed.

Draw probability against outcome, which is the bin most likely to hide a fudge:

| predicted | observed | n |
|---|---|---|
| 12.1% | 11.6% | 1,020 |
| 17.8% | 17.2% | 2,035 |
| 21.3% | 21.3% | 1,915 |
| 23.9% | 25.3% | 3,659 |
| 26.3% | 27.8% | 5,746 |
| 28.6% | 28.1% | 4,581 |
| 30.8% | 31.5% | 1,057 |

### Baselines

1. **Base rate** — the training-set frequencies, 44.1 / 25.1 / 30.8, on every
   match. Anything that cannot beat this has learned nothing.

   Two conventions for this number appear in the project and differ in the
   fourth decimal. **0.2299** is one global base rate fitted over all 20,013
   matches, which is what the README headline quotes. **0.2301** is what
   `run_backtest` records: each season's baseline comes from *that season's*
   training set, pooled by match count, so it is the honest walk-forward
   version and is what `EXPERIMENTS.md` reports. Neither is wrong; they answer
   "what is the base rate" and "what would a base-rate forecaster have scored
   walking forward".
2. **Always home** — accuracy 44.06%. An accuracy baseline only; it has no
   distribution to score.
3. **Bookmaker closing odds**, de-vigged by proportional normalisation of
   reciprocals. Raw `1/odds` sums to about 1.04–1.06 — the margin. Proportional
   removal slightly over-corrects favourites and is more than good enough as a
   benchmark.

The model covers **88.0%** of the RPS distance from the base rate to the market:
`(0.2299 - 0.1992) / (0.2299 - 0.1950)`.

The market is a strong benchmark, **not a theoretical ceiling.** It stays 0.0042
ahead because it prices information this model does not have — team news,
lineups, injuries — and the aggregated opinion of everyone betting into it.

---

## 6. Odds are never features

They are the strongest single predictor available and using them would inflate
every metric, but a model that predicts the market by reading the market has
learned nothing.

This is enforced in `feature_columns`, not merely intended. It used to rely on
odds never being in the frame, which held only because nothing joined them — and
the moment the API joined them for display, every odds column silently became a
model input. There is now an explicit exclusion and a test. Worth remembering
that the failure was loud by luck: the quiet version of that bug is a model that
reads the market and looks excellent.

---

## 7. Elo

```python
START = 1500.0          # arbitrary scale; only differences matter
K = 20.0
HOME_ADVANTAGE = 60.0   # ~0.55 expected score at equal ratings
SEASON_REGRESSION = 0.25
```

Expected score, logistic, base 10 over 400 — that spacing is what makes +400
rating points mean 10:1 odds:

```
E_home = 1 / (1 + 10 ** ((R_away - R_home - 60) / 400))
```

Update, zero-sum, with the away side taking the exact negative:

```
R' = R + K * w * (S - E),     w = max(1, ln(1 + |goal difference|))
S  = 1 win / 0.5 draw / 0 loss
```

Four details, each of which is a decision:

- **`log1p` on the margin.** A 4-0 says more than a 1-0 but not four times as
  much; blowouts are noisy and often happen against ten men.
- **The multiplier floors at 1.** `log1p(0) = 0`, so without the floor a draw
  would never move a rating and an unbeaten run of draws would leave a side
  looking exactly as it did in August.
- **Season regression is toward `START`, not the league mean.** Anchoring to the
  league looks right, since squad turnover is a club property. It was tried and
  rejected: removing the global anchor removes the only restoring force, so
  league means become a random walk. It drifted the five leagues 38 points apart
  on pure churn and **inverted** the ordering against measured European strength
  (Spearman -0.20), against a gain of 0.0011 Brier on 788 cross-league ties.
- **Neutral venues** get no home term, and the check is `is True` explicitly:
  concatenating a frame carrying that column onto one without it fills NaN, and
  `if NaN` is *truthy*, which once silently stripped home advantage from all
  29,143 domestic matches.

**Why Elo at all, alongside the model.** Every other feature is a window, and
windows forget. A side that has been excellent for three years and has just drawn
two looks, to a 19-match window, much like a mid-table side on a good run. An Elo
rating only moves when a result disagrees with what the rating expected, so it
accumulates across seasons instead of expiring. That is why `elo_diff` is the
most important feature in the model by a factor of ten.

**The constants were swept and are already right.** 150 combinations scored on
Elo's own Brier from 2016/17, with 2024/25–2025/26 held out:

```
K    10=.15684  15=.15534  20=.15498  25=.15516  30=.15563  40=.15703
HA   40=.15558  50=.15510  60=.15498  70=.15521  80=.15577
SR   0.0=.15491 0.15=.15467 0.25=.15498 0.35=.15554 0.5=.15673
```

`K` and `HOME_ADVANTAGE` sit exactly on the grid minimum. Only
`SEASON_REGRESSION` moved — 0.25 to 0.15, worth -0.0003 Brier and confirmed
held-out — and it **did not survive the backtest**: pooled RPS 0.2003 → 0.2005.

Keep that lesson: *a better rating is not automatically a better feature when the
model has 234 others to lean on.*

**UEFA ties feed the walk but are not returned as rows** (the `extra` argument).
They are evidence about the teams, not part of the domestic record — putting them
in the main frame would pull them into every rolling window, and they carry no
xG, so the model would be fed something different from what it was trained on.
Elo is the only feature that can carry strength across a league boundary.

That works, and is not enough. On 788 cross-league ties Elo's own Brier improved
0.1772 → 0.1735 and its correlation with the result +0.260 → +0.295, at no
domestic cost. But the model barely uses it: regressing predicted home-win
probability on the league-rating gap gives slope +0.0034 (p 0.22) where the truth
does +0.0402 — about 8% of the right response. **The limit is the training set,
not the feature.** Every match the model trains on is domestic, so the league gap
in training is always zero; the booster has never seen a row where `elo_diff`
carries league strength.

---

## 8. Player ratings

Seven steps. `config/player_rating.yaml` holds every debatable number.

**1. Per-90.** Each counting stat becomes `value / (minutes / 90)`.

**2. Percentile within (role, season).** A centre-back is only ever ranked
against centre-backs in the same season. This buys cross-position comparability
and immunity to season inflation at once — a high-scoring season moves everyone's
percentile together, so it inflates nobody.

Gates: `min_minutes = 450`, and `min_group_size = 40` peers before a pool is
rated at all. Five matchweeks in, whoever has cleared the minutes floor would
otherwise land in the 99th percentile of a pool of twelve.

**3. Sub-ratings.** Weighted means of percentiles; weights are normalised so only
ratios matter, and a negative weight means lower is better. Seven roles — GK,
CB, FB, DM, MID, AMW, FWD. Centre-back, as an example:

```
defending 0.62   involvement 0.24   creation 0.07   finishing 0.07
  defending = interceptions_padj 3.0, tackles_won_padj 2.5, fouls -0.5,
              clean_challenge_rate 1.0, ball_won_per_foul 1.0
```

Defensive volume is **possession-adjusted only**. Raw per-90 counts reward a
defender for playing on a side that has to defend a lot: Brentford sit deep, so
Nathan Collins scored 36 for defending while Damsgaard, a winger who tracks back,
scored 95. Measured over 1,052 completed team-seasons against the *next* season's
goals conceded: possession-adjusted only -0.489, raw plus adjusted -0.483, EA
rating alone -0.478.

DM uses **the same metrics and weights as MID, deliberately.** What was wrong was
the *pool*, not the scoring: a holder ranked for chance creation against number
eights lost 5.2 rating points on average and took 18% of each season's top fifty
while being 40% of the players. Compared with his own kind on identical terms the
gap closes to 0.1 and his share reaches 38%. Re-weighting toward defending and
build-up was measured and rejected — the midfield line's correlation with next
season's points fell 0.646 → 0.634. *Separate the players, not the yardstick.*

**4. Peak blend, `peak_weight = 0.15`.** The composite is pulled toward the
player's single best sub-rating, because a weighted mean of percentiles punishes
specialists from both ends. Mané 2018/19 scored 22 with one assist — 97th
percentile finishing, 36th creation — and winger weights put creation above
finishing, so a 22-goal season rated below an 11-goal one.

0.30 was tried first, improved all three outcome correlations, and **failed the
gate**: walk-forward RPS regressed 0.2003 → 0.2008. That is the whole reason this
is swept against a held-out metric rather than against the correlation table.
0.15 holds the RPS and still moves the ratings the complaint was about: Mané
76 → 78, Van Dijk 1920 63 → 66.

**5. Empirical-Bayes shrinkage, `shrinkage_nineties = 8`.**

```
c_hat = (n / (n + k)) * c_player + (k / (n + k)) * c_role
```

Eight full matches sits halfway between a player's own numbers and the role
average; by thirty he is ~79% himself. This is the regression-to-the-mean
control: a striker with 500 minutes and 7 goals is a small sample, not one of
Europe's best forwards.

**6. Scale.** The shrunk composite's percentile interpolates onto 0–99 through
nine anchors — `0.50 → 68`, `0.90 → 79`, `0.97 → 85`, `1.00 → 94`. A crowded
middle, a long thin top tail, nobody at 99.

**7. The EA blend, `fifa_weight = 0.5`.** EA's overall is first mapped onto this
scale by within-role percentile — raw scales differ, and averaging directly would
let whichever is wider dominate — then

```
rating = (1 - w) * performance + w * EA_on_our_scale
```

Swept 0.7 → 0.0 over 32,078 player-seasons, scored by Spearman against team
outcomes across 1,542 completed team-seasons:

| w | atk > GF | def > GA | ovr > pts |
|---|---|---|---|
| 0.7 | 0.696 | -0.492 | 0.697 |
| 0.6 | 0.707 | -0.487 | 0.703 |
| **0.5** | **0.716** | **-0.480** | **0.707** |
| 0.4 | 0.718 | -0.479 | 0.710 |
| 0.2 | 0.705 | -0.434 | 0.699 |
| 0.0 | 0.677 | -0.379 | 0.663 |

Those are the *next*-season columns, and only they decide anything: the model
joins strength from the previous season, so a same-season correlation flatters a
rating that is partly a consequence of the season it describes. Prediction peaks
on a plateau at 0.4–0.5 and collapses below it; 0 is the worst setting measured
on every metric. 0.5 over 0.4 because the 0.002 between them is noise while the
face validity is not.

**The honesty correction, `unmeasured_shift = 1.0`.** Where a role could not be
measured, lean on EA in proportion to what is missing:

```
w_eff = fifa_weight + (1 - fifa_weight) * unmeasured_shift * (1 - measured)
```

A 2014/15 centre-back has `measured = 0.38` — FBref served no `misc` table, so
`defending`, 62% of the composite, was refused — giving `w_eff = 0.81`.
Renormalising over the hole is right; presenting the result as half the rating is
not. Everything from 1617 on is unchanged bit for bit.

**Unmatched players**, with no EA entry, get their own shrinkage at
`unmatched_shrinkage_nineties = 4`, chosen for distribution parity rather than to
empty the leaderboard: it brings the unmatched p95 (84) level with the matched
p95 (81). Without it they keep untempered percentiles — about 16% of rated
player-seasons but 11–18 of each season's top fifty.

---

## 9. Team strength

```
1. role -> line      GK -> goalkeeper, CB/FB -> defence,
                     DM/MID -> midfield, AMW/FWD -> attack
2. pick the XI       4-3-3; within each line, top N by MINUTES
3. per line          minutes-weighted mean of those players' ratings
4. overall           lines weighted 1 / 4 / 3 / 3, normalised by weights present
5. gate              MIN_RATED_PLAYERS = 11
```

**Slots are filled by minutes, not by rating.** The question is who *will* play,
and a manager's team selection is better evidence of that than our opinion of who
deserves to.

State this precisely: it is **not a predicted lineup**. It is historical minutes
used as a proxy for expected selection, and no external predicted lineup is used
for unplayed matches.

`MIN_RATED_PLAYERS = 11` exists because five matchweeks into 2026/27 a fresh
build produced 72 team ratings from a *median of two* rated players — numbers
that would have been published and then fed back to the model as a
previous-season feature.

Goalkeepers are included, and the margin is narrow. The first justification
written here was wrong: it cited -0.62 for the keeper against -0.49 for the back
line, both measured against the keeper's *own* season, which flatters him because
save percentage is partly a consequence of that season's goals. Re-measured the
way the model uses strength — joined from the previous season — the keeper
predicts next season's goals conceded at -0.474, slightly *worse* than the back
line's -0.483. What keeps him in is the overall: +0.694 with him against +0.690
for the outfield ten.

---

## 10. What actually drives predictions

Mean |SHAP| over the last 2,000 complete rows of the full multi-league feature
build, both goal models averaged. **235 features.**

| # | Feature | Share |
|---|---|---|
| 1 | **`elo_diff`** | **24.3%** |
| 2 | `home_shots_on_target_for_last_19_venue` | 2.5% |
| 3 | `home_elo` | 2.2% |
| 4 | `away_shots_on_target_for_last_19` | 1.9% |
| 5 | `home_npxg_against_last_5` | 1.6% |

| Family | Share |
|---|---|
| Elo (3 columns) | 26.9% |
| xG family | 21.3% |
| Squad strength (12 columns) | 5.9% |
| Pressing / PPDA | 1.8% |
| Rest / congestion | 0.9% |
| Weakest 100 features combined | 6.2% |

`elo_diff` carries ten times the next feature. The share moves with the sample —
an earlier measurement over a different period put it at 18% — so quote the
sample with the number.

### What this table does and does not say

SHAP measures **how much the fitted model's output moves** with a feature. It
does not measure how much *predictive performance* that feature is responsible
for, and under redundancy the two come apart. Both are worth having and they
answer different questions; see the opening section of
[`EXPERIMENTS.md`](../EXPERIMENTS.md) for the sufficiency / attribution /
necessity distinction, which is the thing most easily got wrong here.

Against the family-level Shapley attribution, which retrains for all sixteen
coalitions and measures gain rather than output sensitivity:

| family | SHAP (this table) | Shapley (performance) |
|---|---|---|
| Elo | 26.9% | 26.5% |
| xG | 21.3% | 25.3% |
| squad | **5.9%** | **19.8%** |

Elo agreeing to within half a point is reassuring — SHAP is itself a Shapley
method, over features instead of families — though with four largely
interchangeable families it should not be over-read.

**The squad gap is the informative one, and it is expected rather than a
contradiction.** SHAP is conditioned on *one* fitted model, the full one, in
which Elo, form and xG already supply what squad strength would have said; a
booster handed four correlated views will lean on whichever it splits on first,
and the others look idle. The Shapley figure refits without them, and squad then
reaches 76% of the full model's gain from 18 columns on its own.

So a low SHAP share means "the fitted model does not lean on this", **not**
"this carries no information". That distinction is why the player-level
what-if is declined on the leave-one-out number (§12) rather than on this
table.

**Windows are 5 and 19 matches.** Five alone made the model timid: it could not
separate a genuinely elite side from one that had won three of five, so it shrank
every prediction toward 50%. Where the market said 80%+, it said 78% and the home
side won 87%; simulated tables spread 9.7 points against a real 17.4, with the
champion on 72 where reality averages 89. Nineteen is half a season — long enough
to express persistent quality, short enough to move when a side genuinely
changes. Windows roll across the summer deliberately, or August fixtures would
have no history at all.

**Pruning was measured and rejected.** Dropping the 34 features with a near-twin
above |r| 0.95 — all xG against non-penalty xG — moved RPS +0.00025 [-0.00011,
+0.00059]. Gradient boosting is robust to redundant inputs; removing them costs a
little and gains nothing.

**An ablation ladder is the measurement this document is missing.** Three
*negative* ablations are recorded here — goals-based Elo, feature pruning, the
one-token EA tier — but no positive ladder (Elo only → plus form → plus xG → plus
strength → full). The walk-forward harness can produce it.

---

## 11. Limits

- **No lineups, no team news, no injuries.** This is the whole remaining gap to
  the market, and it is demonstrably *not* a calibration problem: sharpening the
  output probabilities was swept from 0.6 to 1.2, the optimum is ~0.95, and RPS
  does not improve. The model is not uniformly timid — it is timid exactly where
  the market is confident, which is a shortage of information.
- **Where the market says 80–90%, the truth is 85.3% and this model says 79.2%.**
  The one-line summary of its weakness: well calibrated, under-discriminating.
- **Feature engineering on results has hit its ceiling.** A goals-based
  attack/defence Elo was built, properly tuned, and taken up enthusiastically by
  the booster — ranks 2, 3, 4 and 8 of 241, 9.4% of attributed movement, with
  `elo_diff` falling from 17.9% to 16.0% to make room — and predicted exactly as
  well as before: RPS 0.2004 against 0.2003, +0.00003 [-0.00036, +0.00041]. It
  reorganised a tenth of the model's reasoning and gained nothing, because it is
  built from the same match results as everything else.
- **Defending is measured by volume, not by quality or chances prevented.** Van
  Dijk is the clearest case and it looks like a bug: 3rd–11th percentile on
  tackle volume in every Liverpool season, because he does not need to tackle.
  Two fixes were measured and both made the rating *worse* against next-season
  goals conceded — dropping tackles raises him fifteen points and costs the only
  thing the number is for. Aerial duels would settle it and FBref no longer
  publishes them.
- **The rating is better at attackers because the data is.** Correlated against
  EA's next-edition move: forwards 0.15, goalkeepers 0.23, centre-backs,
  full-backs and midfielders **0.04–0.06**, in every season.
- **No event-level data**, so no xThreat, no VAEP, no carries or progressive
  actions, no duels, no post-shot xG, no open-play/set-piece split. Gaps in the
  sources, not the method: Understat and FBref publish season aggregates, and no
  weighting of aggregates reconstructs them.
- **Leicester 2015/16 is not a defect.** They are 15th in team strength because
  EA's own FC16 ratings put that squad 18th of 20, below Watford and Aston Villa,
  and the bookmakers said 5000-1. A squad-quality rating that liked Leicester in
  2015/16 would be the broken one.

---

## 12. Why there is no player-level what-if

The most-requested feature, and the one to keep refusing. "Barcelona without
Yamal: 84 → 82.1, win probability 61% → 56%" is computable and would mean
nothing. Four reasons, which compound:

1. **Removing squad strength from the full model costs nothing measurable:**
   +0.00015 RPS, 95% interval [−0.00001, +0.00032], which does not clear zero
   over 20,013 matches. So perturbing one player's contribution to a
   minutes-weighted mean of eleven moves an input the model has shown it can
   lose entirely without the output changing detectably.

   Note this argument deliberately rests on the **leave-one-out** figure, not on
   squad's 5.9% SHAP share and not on its 19.8% Shapley share. Squad quality
   does carry real information — alone it reaches 76% of the full model's gain
   — but it is redundant with Elo, form and xG, and *redundant* is exactly what
   makes a per-player perturbation meaningless here. See §10 and
   [`EXPERIMENTS.md`](../EXPERIMENTS.md).
2. **It enters as a previous-season team aggregate.** The model has never seen a
   lineup. There is no mechanism by which it could know who is playing.
3. **The counterfactual is not identified.** A player appears in stronger
   lineups, when fit, against opponents chosen by rotation. "Barcelona win more
   with Yamal" is confounded by all three. It would be a *model* counterfactual,
   never a causal estimate.
4. **It would look meaningful.** That is the actual failure mode — a confident
   number with nothing behind it, on a site whose credibility rests on not doing
   that.

The honest version needs per-match lineups — roughly 57 hours of scraping at the
source's rate limit — and then player-level features the model actually trains
on. That is a separate project, and the shortcut is not a smaller version of it.

The same discipline is why `What If?` is labelled a sandbox rather than a
forecast, and why head-to-head records are not shown: the model uses no
head-to-head feature, so displaying one would imply a factor that does not exist
in the prediction.
