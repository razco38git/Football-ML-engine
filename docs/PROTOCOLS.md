# Pre-registered protocols

Experiments that have been specified but **not yet run**. Criteria are fixed here
before any result exists, and move into the harness source as asserted constants
before execution.

Completed work lives in [`../EXPERIMENTS.md`](../EXPERIMENTS.md). A protocol
leaves this file when it has been executed and written up there.

**Status: awaiting approval. Neither experiment has been run. No production code
or configuration has been changed.**

---

# Direction 2 — does a defender's rating travel with him?

## The question

A centre-back's rating correlates **+0.67** with his own club's defensive record.
`config/player_rating.yaml` gives **CB `team_defence: 0.30`** and **FB
`team_defence: 0.0`**, so by construction 30% of a centre-back's composite *is*
his club's expected goals against, goals against and clean-sheet rate.

The config states the limitation plainly: "this is a *club* number, so Saliba and
Gabriel both take Arsenal's 98 and both land on 85." What is unknown is whether
that makes the rating **wrong about the player** — whether the number describes a
footballer or the side he happens to play in.

A transfer separates the two. If a defender's rating is about him, it should
predict his next assessment equally well whether or not he changes club. If it is
about his old side, it should attenuate when he leaves.

## What this is not

`config/player_rating.yaml` already records a sweep of FB `team_defence` at
0.00 / 0.15 / 0.30, with 0.30 winning on aggregates and being declined on
editorial grounds. **This protocol does not re-run that.** That experiment asked
whether a weight improves club-level validity; this one asks whether the number
describes the player. The answer feeds that decision rather than repeating it.

---

## 1. Sample

### Inclusion, and the conditioning problem it fixes

The obvious sample — players with a *rated* season at `t+1` — **conditions on
post-transfer information**: a player is rated at `t+1` only if he cleared the
450-minute gate at his new club. That selects for successful transfers, which is
precisely the thing under test.

EA's raw exports are the whole worldwide database (16,161 rows in `fifa_2425`),
not just the big five, so the outcome can be recovered without that condition.
Inclusion is therefore:

> rated at `t`, EA-matched at `t`, and locatable in EA edition `t+1` by
> normalised name with `age` within ±1.5 of `age_t + 1`.

Measured recovery on the 6,209 departed player-seasons: **77.9% recovered
uniquely, 3 ambiguous (0.05%)**. Age consistency is a strong independent
disambiguator that EA supplies for free.

| | rated-at-t+1 sample | EA-recovery sample |
|---|---|---|
| CB movers / stayers | 630 / 2,305 | **1,168 / 2,229** |
| FB movers / stayers | 524 / 1,901 | **956 / 1,933** |
| conditions on post-transfer minutes | **yes** | no |

Mid-season transfers (245 player-seasons with two club rows) are excluded and
reported separately — they are a different treatment.

**Mover status is defined within EA's own club naming** (`fifa_club` at `t`
against `club` in edition `t+1`), so no alias map is needed and departures
outside the big five are classified correctly. A "stayer" who fell below the
minutes gate through injury is correctly a stayer.

### Residual selection, stated not solved

~6% of eligible player-seasons are unrecoverable — retirements, players dropping
out of EA entirely. This correlates with age and decline. Mitigation: an age-cap
sensitivity analysis (≤30 at `t`), reported alongside.

---

## 2. Outcome and scale

### EA edition drift is real and must be removed

| edition | mean | sd | max |
|---|---|---|---|
| 1415 | 73.41 | 5.43 | 94 |
| 1920 | 75.98 | 5.24 | 94 |
| 2526 | 75.33 | 5.36 | 91 |

The mean drifts **2.5 points** and the sd varies 5.14–5.60, so a raw
`fifa_overall_{t+1} − fifa_overall_t` pools a systematic edition effect with the
player's actual move.

**Outcome:** `Δ_EA = z(fifa_overall_{t+1}) − z(fifa_overall_t)`, each z-scored
**within its own edition** over the full worldwide export. Edition fixed effects
are included in addition, so the specification is robust to the standardisation
rather than reliant on it.

`2627` is excluded as an outcome edition — it carries 504 big-five players
against ~1,900 in complete editions and is still being populated.

### Ceiling effects

Only 0.3–0.8% of players reach ≥90, and the observed maximum falls 94 → 91 over
the window. The ceiling is thin but it bites exactly where it matters: a
high-rated defender can only move down, which induces a mechanical negative
relationship between level and move.

Three responses, all pre-declared:

1. `fifa_overall_t` (z-scored within edition) enters as a covariate — this is the
   principal control for both the ceiling and mean reversion;
2. a sensitivity analysis excluding `fifa_overall_t ≥ 88`;
3. a censored-regression (Tobit) check with the upper limit at the edition
   maximum, reported as a robustness column rather than the headline.

### Regression to the mean

`Δ_EA` regressed on `fifa_overall_t` is negative by construction, and
`performance_rating` is correlated with `fifa_overall_t`. **Omitting
`fifa_overall_t` would therefore bias the coefficient of interest**, and would do
so differentially if movers and stayers differ in prior level — which they do.
Including it is not optional.

### The predictor

**`performance_rating`**, not `rating`. `rating` is a 50/50 blend with
`fifa_overall_t`, so using it against EA's next edition would be substantially
EA predicting itself. `performance_rating` is what `rating.py` computes before
`_blend_with_fifa`.

Standardised within (role, season): `z_perf`.

---

## 3. Specification

### Primary estimand: the interaction, not the ratio

One pooled regression per role:

```
Δ_EA  =  α
       + β · z_perf
       + δ · (z_perf × mover)          <-- the estimand
       + η · mover
       + θ · z(fifa_overall_t)
       + controls
       + season FE + origin-league FE
```

- **β** = stayer slope, **β + δ** = mover slope. Both reported separately with
  intervals, as required.
- **δ is the primary estimand.** It is a difference, it is estimated in one
  model, and it is stable regardless of how small β is.

**Controls:** age and age², minutes at `t`, `measured_share`,
`fifa_weight_used`. All from season `t`.

### The ratio is reported, and its instability is reported with it

The travel ratio `(β + δ) / β` is reported because it is interpretable, but it is
**not the decision variable**. `METHODOLOGY.md` records the defender EA-move
correlation at **0.04–0.06**, so β is expected to be small and the ratio has a
near-zero denominator.

Three things are therefore reported with it:

1. a **Fieller** interval, which is the correct construction for a ratio of
   normals, rather than a naive percentile interval on the bootstrap ratios;
2. the **proportion of bootstrap replicates in which β changes sign** — above
   ~5% the ratio is not interpretable at all and is reported as undefined;
3. the bootstrap ratio distribution's tail behaviour, so heavy tails are visible
   rather than summarised away.

### Power gate, declared in advance

**If the 95% CI for β (stayers) includes zero, the primary analysis is declared
uninformative** and no contamination or equivalence claim is made from it. The
attenuation of a null is undefined. Given the documented 0.04–0.06 baseline this
is a realistic outcome, and pre-declaring it is what stops it being rationalised
afterwards.

In that case the decision falls to the decomposition (§5) and the result is
written up as "this instrument lacks the resolution", which is itself worth
recording.

### Uncertainty

**Cluster bootstrap by player**, 10,000 resamples, seed 7. A player contributes
up to 11 consecutive-season pairs; an observation-level bootstrap would
substantially understate every interval. This is the same effective-sample point
`player_rating.yaml` already makes for the FB sweep ("the paired bootstrap's
effective sample is nearer 12 than 20,060").

### Matching

Regression adjustment is primary. **Coarsened exact matching** is the robustness
check: age band (≤21 / 22–25 / 26–29 / 30+) × prior-rating tercile × minutes
tercile, within season and origin league, movers matched to stayers.

All matching variables are from season `t`. If the two approaches disagree, the
question is reported unresolved.

---

## 4. Decision rule: three outcomes, not two

A confidence interval containing 1.0 is **not** evidence of no effect. The
decision is three-way, on the ratio, with equivalence bounds fixed here.

### Equivalence interval: **[0.85, 1.18]**

That is "attenuation of less than 15%". The justification is mechanistic rather
than conventional:

`team_defence` is **30% of a centre-back's composite weight**. If club
contamination were proportional to that weight, removing the club's contribution
by transfer should attenuate the slope on the order of 30%. An attenuation below
**half** of what the mechanism predicts cannot be attributed to it with any
confidence, and could not justify a weight change given the editorial cost the
config already records for exactly this trade ("a full back rated 8% on
ball-winning and 30% on his club's clean sheets — the variant where a player's
published number is least about the player").

1.18 is the reciprocal of 0.85, so the interval is symmetric in log-ratio.

### The three verdicts

| verdict | condition |
|---|---|
| **Contamination** | 95% CI for δ excludes 0, the ratio point estimate is < 0.85, and the decomposition (§5) agrees in sign |
| **Practical equivalence** | 90% CI for the ratio lies **entirely within** [0.85, 1.18] — a TOST at α = 0.05 |
| **Inconclusive** | anything else, including a CI that includes 1.0 but extends beyond the equivalence bounds, and any case failing the power gate |

Inconclusive is a real and likely outcome. It is written up as such and does
**not** close the question.

### If contamination fires

A candidate reweighting is **measured and not shipped**, and reported against the
existing guards, reusing `next_season_validity()` from
`pipelines/squad_size_sweep.py` verbatim:

| guard | bar |
|---|---|
| `def_vs_ga` | must not worsen by more than **0.010** |
| `ovr_vs_pts` | must not worsen by more than **0.005** |
| walk-forward RPS | upper CI bound **< +0.0005** |

Same bars as the positional-quota experiment, so the two are directly
comparable. **Production weights are not changed by this experiment under any
outcome.**

---

## 5. CB versus FB: a direct test, and what it can and cannot support

### The test

CB and FB are pooled into one regression with a **three-way interaction**:

```
Δ_EA  ~  z_perf × mover × is_CB  +  all lower-order terms  +  controls
```

The quantity reported is **Δδ = δ_CB − δ_FB**, with a player-clustered bootstrap
confidence interval. Not overlapping point estimates — a direct estimate of the
difference with its own uncertainty.

### Assumptions required to read Δδ as evidence about `team_defence`

Stated because most of them are only partly satisfied.

**Holds, and is documented.** The two roles share the defending metric definition
exactly — the config says so: "Full backs share the centre backs' defending
definition exactly, so they share its defect." So Δδ is not driven by different
defensive measurement.

**Holds, and is measured.** Transfer selection is similar: attrition into the
follow-up sample is 33.5% for CB and 34.2% for FB, and mover shares are
comparable.

**Does not hold.** `team_defence` is not the only weight that differs:

| | defending | creation | involvement | finishing | **team_defence** |
|---|---|---|---|---|---|
| CB | 0.32 | 0.07 | 0.24 | 0.07 | **0.30** |
| FB | 0.38 | 0.32 | 0.25 | 0.05 | **0.00** |

FB leans **0.32 on creation against CB's 0.07**. Creation is an individual
attacking measurement and is plausibly *less* club-contaminated, so it predicts
FB attenuating less — **confounded with the `team_defence` difference and in the
same direction**. Δδ cannot separate them.

**Doubtful.** The roles are not equally stable across a transfer. Full-backs are
converted to wing-backs and to midfield more often than centre-backs change
function, which adds genuine role-change noise to the FB arm.

### Consequence, pre-declared

**Δδ is corroborating evidence, not identification.** It is reported as
"consistent with / inconsistent with a `team_defence` mechanism", never as
"attributable to `team_defence`".

**The identifying test is the within-role decomposition**, which needs no
cross-role comparison:

> Project the club-level defensive aggregates (`team_xga_per_match`,
> `team_ga_per_match`, `team_clean_sheet_pct`, all at season `t`) out of
> `performance_rating`, giving a club-orthogonal residual rating. Re-estimate δ
> using the residual. **If contamination is the mechanism, δ should shrink toward
> zero for CB and change little for FB.**

Everything in that projection is season-`t` information. This is the test the
conclusion rests on; the CB/FB contrast supports it.

---

## 6. EA-free secondary outcome

Kept because EA's own ratings are nudged by club success — the same contamination
being tested — so the primary outcome cannot stand alone. **It is corroborating
evidence and explicitly not a clean causal instrument.**

### Design

Difference-in-differences at club level.

- **Unit:** receiving club-season.
- **Pre window:** club's goals conceded per match in season `t` (the season
  before the signing), over all league matches.
- **Post window:** the same in season `t+1`.
- **Outcome:** `Δ GA/match = GA_{t+1} − GA_t`, which is why both windows are full
  seasons — a shorter post window would be contaminated by fixture difficulty.
- **Treatment:** the standardised `performance_rating` of the arriving defender,
  measured at his **old** club in season `t`.
- **Comparison baseline:** all other clubs in the same league-season that
  received no rated defender arrival, via **league-season fixed effects**. The
  estimand is whether a *higher-rated* arrival predicts a *larger* defensive
  improvement than the league-season norm.

### Adjustments

- **League-season fixed effects** absorb league scoring level, season effects and
  rule changes.
- **Opponent strength is close to self-adjusting here** — a domestic league is a
  double round-robin, so every club faces nearly the same opposition within a
  season. This is a real advantage of the setting and is why opponent strength
  enters as a residual control (the club's own previous-season
  `strength_overall`) rather than a fixture-weighted correction.
- **Mean reversion** is the dominant confounder: clubs that concede heavily
  improve regardless. Controlled by including `GA_t` on the right-hand side, and
  by a sensitivity analysis restricted to mid-table clubs.

### Minutes, and the conditioning it forces

A signing who plays 200 minutes cannot move a defence, so the arriving player's
minutes at the new club must enter — as an exposure weight, with a reported
sensitivity at thresholds of 900 / 1,500 / 2,000 minutes.

**Those minutes are post-transfer information.** This outcome therefore
*necessarily* conditions on a post-treatment variable, and the coefficient is a
conditional-on-playing effect rather than an intention-to-treat one. **This is
the main reason it is corroborating only**, and it is why the primary outcome was
redesigned to avoid exactly this.

### Attribution limits

A club's goals conceded is produced by eleven players, a goalkeeper, a manager,
a tactical system and the rest of that summer's business. Attributing its change
to one arrival is weak by construction.

Mitigations and their cost in sample:

| restriction | CB | FB |
|---|---|---|
| arrivals playing ≥1,500 min at the new club | 421 | 268 |
| receiving club-seasons | 355 | 246 |
| **with exactly one such arrival** | **294** | **224** |

The single-arrival restriction is the primary specification. It does not control
for managerial change, other signings below the minutes threshold, or departures
— and no available data would let it. **Underpowered and confounded; read as
corroboration of direction, never as an effect size.**

---

## 7. Post-transfer conditioning audit

Every element checked against the rule that nothing after the transfer may enter
except the outcome and the treatment definition.

| element | source | verdict |
|---|---|---|
| inclusion in sample | EA match at `t`, EA presence at `t+1` | **clean** — no minutes gate at the new club |
| predictor `z_perf` | season `t` | clean |
| covariates (age, minutes, `measured_share`, `fifa_overall_t`) | season `t` | clean |
| matching variables | season `t` | clean |
| mover/stayer status | EA club at `t` vs `t+1` | treatment definition, not conditioning |
| decomposition projection | club aggregates at `t` | clean |
| bootstrap clustering | player identity | clean |
| **secondary outcome minutes weight** | season `t+1` | **post-treatment — declared, corroborating only** |
| residual attrition (~6%) | retirement, EA exit | **selection, not removable** — age-cap sensitivity reported |

---

## 8. Limitations to carry into the write-up

1. **Movers are not randomly assigned.** Clubs buy on information we do not
   have. Matching on age, rating and minutes cannot remove that, and no design
   here is causal.
2. **EA is an external proxy, not ground truth.** It is lagged, noisy, and
   partly driven by club prestige and team success — the same contamination being
   tested. A positive result on EA alone is weak.
3. **Low baseline power.** The defender EA-move correlation is 0.04–0.06. The
   power gate exists because this experiment may simply not resolve the question.
4. **Δδ cannot isolate `team_defence`** — the CB/FB weight vectors differ on
   creation as well, in the same direction.
5. **The secondary outcome conditions on post-treatment minutes** and attributes
   a team outcome to one player.
6. **Hyperparameters and all production weights fixed**, as everywhere else.

---

# Direction 3 — conditional Dixon-Coles ρ

Unchanged in substance from the approved outline; three points tightened.
**Fully isolated from Direction 2**: separate harness, separate data path,
nothing shared but the backtest file, and neither may be run in the same process.

## Motivation

ρ is a single scalar fitted in `MatchPredictor.fit` (`match_model.py:157`).
Profiling the Dixon-Coles likelihood per league on the committed backtest:

| | ρ̂ | 95% CI |
|---|---|---|
| pooled | −0.0567 | [−0.0740, −0.0367] |
| **D1** | **−0.1297** | **[−0.1784, −0.0787]** |
| SP1 | −0.0326 | [−0.0743, +0.0038] |

The Bundesliga's interval excludes the shipped −0.0566. ρ also trends with
scoring level, −0.038 in the lowest λ quartile to −0.070 in the highest.

## Arms

| arm | ρ |
|---|---|
| production | one global ρ |
| per-league | ρ per league |
| shrunk | per-league ρ, James–Stein toward the pooled value |
| per-λ | ρ linear in predicted total goals |

## Fitting rule

**Every ρ in every arm is fitted strictly within the training fold.** No arm may
see the scored fold's scorelines. This is the leak this experiment is most likely
to commit and it is asserted in the harness, not merely intended.

## Reporting

Out-of-sample scoreline log-likelihood, **pooled and per league, each with a
bootstrap interval** — not point estimates. Per-season win/loss counts alongside,
as everywhere else in the project.

## Metrics

Scoreline log-likelihood is primary; exact-score hit rate and BTTS / over-under
calibration are secondary.

**RPS is reported as a full outcome with its own interval, and is not assumed
insensitive to ρ.** ρ moves mass between the 0-0, 0-1, 1-0 and 1-1 cells, and
those cells sit on both sides of the draw/win boundary, so an RPS effect is
possible and will be measured rather than asserted away.

## Acceptance criteria

- out-of-sample scoreline log-likelihood improves in **≥ 4 of 5 leagues**, and
- pooled log-likelihood improvement has a bootstrap interval excluding zero, and
- exact-score hit rate does not fall, and
- RPS does not regress beyond **+0.0005** on the upper CI bound.

## Kill condition

Per-league ρ fitted on training folds fails to beat the global ρ out of sample —
the heterogeneity is in-sample noise. D1's advantage is **4.99 log-likelihood
over 3,402 matches** (~0.0015 nats/match), so this is a live possibility.

## Expected value, stated honestly

Small, and mostly confined to scoreline surfaces. It earns a slot because it is
hours rather than days, and because one ρ across five leagues is currently an
undocumented assumption rather than a measured choice.

---

# Assertions and tests enforcing the constraints

Written into `tests/test_rating_travel_ab.py` and `tests/test_conditional_rho_ab.py`
**before** either harness runs, following the pattern of
`tests/test_positional_quota_ab.py`.

## No-write isolation (both experiments)

| assertion | what it stops |
|---|---|
| source contains none of `to_csv`, `to_pickle`, `to_parquet`, `to_json`, `to_hdf`, `shutil`, `os.replace`, `os.remove`, `os.rename`, `unlink(`, `write_text`, `write_bytes`, `mkdir(`, `savetxt`, `np.save` | a killed run leaving artefacts behind — a `finally` does not survive `SIGKILL`, and one experiment once left `team_strength.csv` holding its own numbers with the API serving them |
| no `open(...)` with a `w` / `a` / `x` mode | the same, via the one path a string scan would miss |
| AST import check: nothing importing `build_players` | reaching the module that writes production ratings |
| `git status` clean under `data/processed/` and `config/` after the run | the catch-all, verified rather than assumed |
| SHA-256 of `player_ratings.csv`, `team_strength.csv` and `models/latest.txt` identical before and after | production artefacts provably untouched |

## Direction 2 — post-transfer conditioning

| assertion | what it stops |
|---|---|
| every covariate and matching column resolves to a season-`t` row; a column drawn from the `t+1` frame raises | the whole class of bugs this revision exists to fix |
| sample inclusion never references `rated`, `minutes` or `measured_share` at `t+1` | re-introducing the minutes-gate selection |
| the only `t+1` quantities reachable by the primary model are `fifa_overall_{t+1}` and EA club at `t+1` | silent leakage through a merge |
| `2627` is absent from outcome editions | scoring against a half-populated EA export |
| mid-season transfer rows (two club rows in one player-season) are excluded, count asserted at 245 | mixing two different treatments |
| the secondary outcome is tagged `post_treatment=True` and the report refuses to print it as a headline | the corroborating outcome being promoted to the conclusion |

## Direction 2 — estimand integrity

| assertion | what it stops |
|---|---|
| the predictor column is `performance_rating`, never `rating` | EA predicting itself through the 50/50 blend |
| `fifa_overall_t` is present in the design matrix | the mean-reversion and ceiling bias |
| `Δ_EA` is built from within-edition z-scores | EA's 2.5-point edition drift entering the outcome |
| bootstrap resampling draws **players**, not rows; asserted by checking a resample contains whole player blocks | intervals that are too narrow by construction |
| equivalence bounds asserted as the declared constants `0.85` / `1.18`, and `MARGIN`, `DEF_GAIN`, `OTHER_TOLERANCE` as in the quota experiment | a bar moved after seeing the answer |
| the power gate is computed from the declared constants and fires before any verdict is printed | an attenuation claim on top of a null stayer slope |
| a group in which no arm can move is not counted as a pass | the counting bug from the extended-Elo run, pinned once and kept pinned |
| the ratio is reported as undefined when >5% of bootstrap replicates flip the sign of β | a ratio built on a near-zero denominator being read as a finding |

## Direction 3 — fold discipline

| assertion | what it stops |
|---|---|
| every ρ is fitted from training-fold scorelines only; the harness raises if a fit sees a scored-fold index | the leak most likely to be committed here |
| all arms are asserted to have scored the same fixture set | intervals comparing different populations |
| arms are built from a declared constant list, with `production` first and equal to the shipped global ρ | measuring something nobody runs |
| RPS is computed and reported for every arm | assuming ρ cannot move RPS |
| the two harnesses cannot be imported into one process | Direction 3 picking up Direction 2 state |
