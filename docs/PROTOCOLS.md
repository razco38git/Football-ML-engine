# Pre-registered protocols

Experiments that have been specified but **not yet run**. Criteria are fixed here
before any result exists, and move into the harness source as asserted constants
before execution.

Completed work lives in [`../EXPERIMENTS.md`](../EXPERIMENTS.md). A protocol
leaves this file when it has been executed and written up there.

**Status.** Both directions have been **executed**; results in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md).

> **The figures in Direction 2 below are the pre-run feasibility audit's**, and
> are deliberately left as they were written. The executed run's counts differ
> slightly: it recovers through two tiers rather than one raw lookup, so it
> finds **more** pairs (CB 1,347 movers / 2,379 stayers against the 1,168 /
> 2,229 projected here) at the same overall recovery rate of 92.7%. Where the
> two disagree, the write-up in `EXPERIMENTS.md` is the executed figure.
>
> That 92.7% is also what caught a bug: the first execution reported 88.9%, the
> protocol said 92.7%, and the protocol was right. See *"The invalid first run:
> a filter that destroyed a fold"*. Direction 3: no-ship on all three
arms. Direction 2: **inconclusive** on two pre-declared triggers, with no
contamination detected and nothing to ship. Both protocols are kept below as
the pre-registrations they were, unedited after the fact. No production code,
configuration or artifact was changed by either.

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

### Recovery is two-tier, and the audit of what it misses

| tier | definition | n | share |
|---|---|---|---|
| 1 | already EA-matched at `t+1` by the production matcher | 14,149 | 69.0% |
| 2 | raw-edition lookup: normalised name, `age` within ±1.5 of `age_t + 1` | 4,852 | 23.7% |
| | **recovered** | **19,001** | **92.7%** |
| | unrecovered | 1,490 | **7.3%** |

**The unrecovered are not missing at random.** Standardised differences against
the recovered, on pre-transfer variables only:

| variable | recovered | unrecovered | std. diff |
|---|---|---|---|
| `fifa_age` | 25.79 | **28.26** | **+0.57** |
| `minutes` | 1,804 | **1,366** | **−0.58** |
| `performance_rating` | 68.74 | 65.61 | −0.37 |
| `fifa_overall` | 75.32 | 74.28 | −0.20 |
| `measured_share` | 0.96 | 0.97 | +0.14 |

Unrecovered players are **older, played less, and were rated lower** — the
retirement-and-decline signature. By age band, the unrecovered share runs 3.7%
(≤21), 4.7%, 6.8%, 12.2%, **25.5% (34+)**.

**Recovery is balanced across the two roles under test**: CB 93.0%, FB 93.3%.
The CB-versus-FB contrast is therefore not threatened by differential
missingness, which is the part of the design most exposed to it.

### The treatment is missing too, not just the outcome

Mover status is defined by EA club at `t+1` against `t`. An unrecovered player
has **no `t+1` row at all**, so for those 1,490 player-seasons **both the
outcome `Δ_EA` and the treatment `mover` are unobserved.** This is jointly
missing treatment and outcome, which is a materially harder problem than missing
outcome alone, and the previous draft understated it.

Mover share is observable only among the recovered: **CB 34.4%** (1,168 of
3,397), **FB 33.1%** (956 of 2,889).

### What IPW identifies, and what it cannot

IPW identifies the target estimand under

> `P(R = 1 | X, A, Y) = P(R = 1 | X)`

where `R` is recovery, `X` the pre-transfer covariates, `A` mover status and `Y`
the EA move. That is **recovery independent of both treatment and outcome given
the observed pre-transfer covariates.**

The recovery model can only be fitted on `X`, because `A` is unobserved exactly
when `R = 0`. So:

- **IPW corrects differential missingness by observed pre-transfer
  characteristics** — age, minutes, rating, `measured_share`, role, season,
  league. On those it does real work: the imbalances are large (+0.57 on age,
  −0.58 on minutes) and reweighting removes them.
- **IPW cannot address differential missingness by mover status**, which is the
  threat that matters. If a player who moves and fails leaves EA's database at a
  different rate from a player who stays and fails, *conditional on X*, no
  reweighting on `X` repairs it, and the data contain no information with which
  to estimate that difference.

**Consequence, stated plainly: the primary effect is not point-identified
without an untestable assumption.** IPW is reported as the estimate under MAR
in `X`; it is not a solution to selection on `A` or `Y`, and will not be
described as one.

### The direction of bias is a conjecture, not a result

An earlier draft of this protocol asserted that missingness biases δ toward zero,
on the reasoning that a player who drops out of professional football leaves EA's
database while a declining stayer at a big-five club remains in it.

**That reasoning is a conjecture and is withdrawn as a stated direction.** It is
not supported by any analysis here and cannot be, because the quantity it is
about — recovery rate by mover status — is unobservable. It is plausible; so is
the reverse, since a declining stayer who loses his place is also invisible to a
rating gate while remaining at his club, and EA retains squad players who no
longer play.

No claim about the direction of bias will be made in the write-up unless the
sensitivity grid below supports one. The grid is what replaces the conjecture.

### Pattern-mixture analysis over both unknowns

Because `A` and `Y` are jointly missing, the sensitivity analysis is a **grid in
two dimensions**, not a single imputation.

**Dimension 1 — mover share among the unrecovered, `π`.** Scenarios:

| π | reading |
|---|---|
| **0.33** | same as observed — missingness unrelated to mover status |
| 0.50 | movers ~2× over-represented among the vanished |
| 0.67 | movers ~4× over-represented |
| 0.00 / 1.00 | the extreme anchors, reported as bounds rather than scenarios |

**Dimension 2 — imputed `Δ_EA`, possibly differing by assigned status.** At each
`π`, the unrecovered receive `Δ_EA` drawn from the observed distribution at the
**10th percentile** (decline), the **median** (MAR), and separately with movers
and stayers given *different* values so that the imputation can itself be
adverse to the hypothesis.

δ is re-estimated at every cell. Two things are reported:

1. **the range of δ across the grid**, which is the honest statement of how much
   the missingness could be doing;
2. a **tipping-point**: the combination of `π` and imputed decline at which the
   verdict first flips. If the verdict survives the whole plausible region
   (`π ≤ 0.67`, decline down to the 10th percentile) the conclusion is robust to
   this; if it flips at `π = 0.40`, it is not, and that is reported as the
   headline rather than buried.

### If the grid does not resolve it

Then the protocol's answer is that **the primary effect cannot be identified from
these data without an assumption the data cannot check**, and the experiment
reports that rather than a number. That is a legitimate and pre-declared outcome,
and it is why the EA-free corroborating outcome and the within-role decomposition
exist — both are affected by missingness differently, and neither requires mover
status to be known for the unrecovered.

### Missingness sensitivity analyses, pre-declared

1. **Inverse-probability weighting.** A logistic recovery model on pre-transfer
   covariates only — age, age², minutes, `performance_rating`, `measured_share`,
   role, season, origin league — with the analysis reweighted by the inverse
   predicted probability. Weights trimmed at the 1st and 99th percentiles.
   This is the MAR-based correction and is reported beside every
   complete-case estimate.
2. **Pattern-mixture departure from MAR.** Unrecovered rows imputed at the
   observed **10th percentile** of `Δ_EA` (a decline assumption), then at the
   observed median (MAR), and δ re-estimated under each. The reported quantity
   is the **range of δ across the two**, which is a bound on how much the
   missingness could be doing.
3. **Age cap.** Restricted to `fifa_age ≤ 30` at `t`, where unrecovered is
   3.7–6.8% rather than 25.5%.

A conclusion that does not survive all three is reported as missingness-sensitive
and does not support a weight change.

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

**Tobit was specified in the previous draft and is withdrawn as a sensitivity
analysis.** It assumes a continuous latent variable censored at a known,
structural threshold, and neither condition holds here:

- the per-edition maximum (91–94) is an **observed maximum, not a censoring
  point** — EA's scale permits 99 and nobody reaches it;
- there is **no mass point at the top**, which is the defining signature of
  censoring. Counts approaching the 2025/26 maximum run 45, 30, 16, 22, 7, 13,
  5, 2 — a smooth decline, not a pile-up;
- the outcome is a **difference of two standardised scores**, so any bound
  applies to the components rather than to the quantity being modelled.

A Tobit here would invent a latent tail above 91 that nothing in the data
supports. It is **labelled exploratory**, reported only if a reader asks for it,
and no conclusion rests on it.

Three defensible responses replace it, all pre-declared:

1. **`z(fifa_overall_t)` as a covariate** — the principal control for both the
   ceiling and mean reversion;
2. **Top-band exclusion** — a sensitivity excluding `fifa_overall_t ≥ 88`;
3. **Rank outcome, run in parallel to the primary.** `Δ_rank` built from
   **within-edition percentile ranks** instead of z-scores. This respects the
   bounded, ordinal nature of an EA rating by construction, is immune to scale
   compression at the top, and makes no distributional assumption. If the z-score
   and rank specifications disagree in sign, the result is reported as
   scale-dependent and inconclusive.

A median (quantile) regression on the z-score outcome is reported as a fourth
robustness column, since it is insensitive to the asymmetric upper tail. A
cumulative-link ordinal model on the `t+1` *level* with the `t` level as
covariate is available and is not pre-declared, because it answers a slightly
different question and would invite being selected after the fact.

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

### Equivalence interval: **[0.85, 1.18]**, a declared threshold rather than a derivation

The bounds are a **pre-declared practical threshold**. They are *informed* by the
weight structure and do **not** follow from it mathematically — nothing implies
that a 30% composite weight produces a 30% slope attenuation, and the previous
draft overstated this.

What the 30% does is establish an **order of magnitude**: `team_defence` is 30%
of a centre-back's composite, so if club contamination operates through it at
all, the attenuation is plausibly in the tens of percent rather than the single
digits. Setting the threshold at 15% — roughly half that scale — marks the point
below which an effect is too small to attribute to the mechanism or to justify a
weight change, given the editorial cost the config already records for this
trade.

That is a judgement about what would change a decision, declared in advance so it
cannot be adjusted afterwards. 1.18 is the reciprocal of 0.85, so the interval is
symmetric in log-ratio.

### The continuous estimates are the real output

Verdicts are a convenience for the write-up. **What gets reported in every case,
whatever the verdict:**

- β (stayers) and β + δ (movers), each with a player-clustered 95% CI;
- δ with its CI, on both the z-score and rank specifications;
- the ratio with its Fieller interval and sign-flip diagnostic;
- the same quantities under IPW, under the pattern-mixture bounds, and under the
  age cap;
- Δδ for the CB-versus-FB contrast with its CI.

No conclusion is stated as a verdict without the interval beside it.

### Alternative-bound sensitivity

The equivalence decision is re-run at **[0.80, 1.25]** (a looser 20% threshold)
and **[0.90, 1.11]** (a stricter 10%), and all three are tabulated. If the verdict
flips across those bounds, the write-up says so plainly: the conclusion then
depends on the threshold rather than on the data, which is itself the finding
and is reported as inconclusive.

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

**The within-role decomposition carries more weight**, because it needs no
cross-role comparison. It is specified fully below.

---

## 5a. The residualisation procedure

### Variables projected out

Exactly the three club aggregates that constitute the `team_defence` sub-rating
in `config/player_rating.yaml`, each at **season `t`**, for the club the player
played for in season `t`:

```
team_xga_per_match      team_ga_per_match      team_clean_sheet_pct
```

No others. Adding further club covariates would project out legitimate player
signal that correlates with club quality, and the question is specifically about
*these* inputs.

### Where the projection is estimated

OLS of `performance_rating` on the three aggregates, with an intercept, fitted
**within each (role, season) stratum** — matching the stratum the rating itself
is percentile-ranked in.

The projection is fitted on **all rated players in that (role, season)
stratum**, not on the analysis sample. That superset is deliberate: it uses only
season-`t` information, it does not condition on a player having an EA outcome at
`t+1`, and it keeps the residual definition independent of which rows survive
into the regression.

Fitted on the full stratum, the projection involves **no outcome data of any
kind**, so there is no path by which it can leak. This is asserted rather than
argued — see the test table.

### Scaling, so the comparison is fair

Residualising necessarily shrinks variance, and a smaller-variance predictor
produces a smaller coefficient **for that reason alone**. Comparing a raw δ to a
residual δ without correcting for this would manufacture the result.

Both predictors are therefore **re-standardised to unit variance within (role,
season)** before entering any regression:

```
z_perf      = standardise( performance_rating ,  within (role, season) )
z_perf_res  = standardise( residual           ,  within (role, season) )
```

so δ and δ_res are both "per standard deviation of the predictor as used" and are
directly comparable.

**The projection's R² is reported per role and season**, because it is the
quantity that says how much of the rating these three club numbers account for
in the first place. If R² is near zero for a role, the decomposition has nothing
to remove there and δ_res ≈ δ is uninformative rather than reassuring — that case
is pre-declared as such.

### Prediction and interpretation

> **If club contamination is the mechanism, δ_res should shrink toward zero for
> CB and change little for FB.**

The comparison reported is **δ − δ_res**, per role, with a player-clustered
bootstrap CI, and the two roles' changes contrasted.

**This is mechanism-consistent evidence, not causal proof.** The residual is the
part of the rating orthogonal to three particular club measurements in one
season; it is not "the player's true ability". Club quality and player quality
are genuinely correlated, so projecting one out removes real signal along with
any contamination, and the procedure cannot distinguish the two. A shrinking δ_res
is consistent with contamination and also consistent with the residual simply
being noisier. The write-up must say both.

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
**2b.** **7.3% of cases are unrecoverable, and both the outcome and the
   treatment are missing for them** — the unrecovered are older, played less and
   were rated lower. IPW addresses selection on observed pre-transfer
   covariates only; selection on mover status is untestable, so **the primary
   effect is not point-identified**. No direction of bias is claimed. Recovery is
   balanced across CB and FB (93.0% / 93.3%), so the role contrast is protected.
**2c.** **The residual is not "the player".** Club and player quality are genuinely
   correlated, so the projection removes real signal alongside any contamination
   and cannot separate them.
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

## Exact-score hit rate: a non-inferiority margin, not a zero-tolerance rule

"Does not fall" would reject an arm on a decline of one match in twenty
thousand, which is noise. The margin is set the same way the RPS floor was — at
the resolution of the measurement.

> **Margin: 0.25 percentage points.** The criterion is that the **upper bound of
> the 95% paired-bootstrap CI on the decline** sits below 0.25pp — not that the
> point estimate is non-negative.

**This is a declared practical tolerance, not a quantity derived from a
standard deviation.** The unpaired sampling sd of a 12.8% rate over 20,038
fixtures — `sqrt(0.128 × 0.872 / 20,038) = 0.236pp` — is cited only as an
**order-of-magnitude anchor**, and it is the wrong sd for this comparison: the
arms are paired, sharing λ and every feature, and differ only in ρ, so the
paired sd is much smaller. Deriving the margin from the unpaired figure would be
a category error dressed as arithmetic.

What justifies 0.25pp is practical: it is about 2% of the hit rate in relative
terms, below the precision at which the site quotes the number, so a decline
inside it would not change anything the project publishes. The margin is
deliberately permissive — it exists to stop a noise-level decline vetoing a real
log-likelihood gain, not to wave through a real one.

**The paired-bootstrap point estimate and its full interval are reported
explicitly for every arm**, pooled and per league, so a reader can apply a
different tolerance without re-running anything.

Reported with uncertainty, pooled and per league, like everything else.

## Acceptance criteria

- out-of-sample scoreline log-likelihood improves in **≥ 4 of 5 leagues**, and
- pooled log-likelihood improvement has a bootstrap interval excluding zero, and
- exact-score hit rate decline has a **95% CI upper bound below 0.25pp**, and
- RPS does not regress beyond **+0.0005** on the upper CI bound.

All four are computed from constants declared in the harness source, asserted by
tests, and reported with intervals whether or not they pass.

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
| equivalence is evaluated at all three bound pairs ([0.85,1.18], [0.80,1.25], [0.90,1.11]) and the harness refuses to print one alone | a threshold chosen to suit the answer |
| the rank specification runs whenever the z-score one does, and a sign disagreement forces an inconclusive verdict | a scale-dependent result presented as a finding |
| Tobit, if run at all, is tagged `exploratory=True` and cannot appear in the verdict | a mis-specified censoring model carrying a conclusion |
| IPW weights, pattern-mixture bounds and the age-cap estimate are computed for every reported δ | a complete-case result standing alone |
| the recovery model uses only season-`t` covariates | missingness correction reaching into the outcome |
| the residual projection is fitted within (role, season) on the **full rated stratum**, and the harness raises if the fitting frame is the analysis sample | conditioning the residual on EA follow-up |
| the projection uses exactly the three `team_defence` aggregates, asserted against the config | quietly widening what is projected out |
| both `z_perf` and `z_perf_res` are asserted to have unit variance within (role, season) before entering any model | a smaller δ_res manufactured by a shrunken predictor |
| the projection R² is reported per role and season | a decomposition with nothing to remove being read as reassurance |

## Direction 3 — fold discipline

| assertion | what it stops |
|---|---|
| every ρ is fitted from training-fold scorelines only; the harness raises if a fit sees a scored-fold index | the leak most likely to be committed here |
| all arms are asserted to have scored the same fixture set | intervals comparing different populations |
| arms are built from a declared constant list, with `production` first and equal to the shipped global ρ | measuring something nobody runs |
| RPS is computed and reported for every arm | assuming ρ cannot move RPS |
| the exact-score margin is the declared constant `0.25` pp and the criterion reads the CI upper bound, not the point estimate | a noise-level decline vetoing a real gain, or a real decline passing |
| log-likelihood, RPS and exact-score are each reported pooled **and** per league, with intervals | a pooled number hiding a league that got worse |
| the two harnesses cannot be imported into one process | Direction 3 picking up Direction 2 state |
