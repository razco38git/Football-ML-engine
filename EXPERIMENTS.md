# Experiments

Measurements that answer a question once and write down the answer. Nothing here
runs on a schedule and nothing the site serves depends on it.

Results that changed the production model live in the docstring of the thing they
changed — that is deliberate, so a constant and its justification cannot drift
apart. This file is for experiments whose output is a *finding* rather than a
code change.

---

# Ablation ladder

**Purpose.** The project had three ablations on record and all three were
negative: a goals-based attack/defence Elo that gained nothing, dropping 34
near-duplicate features, and a name-matching tier. Each says "this change does
not help". None says what the model's predictive power actually rests on.

This measures the other direction: start from almost nothing and add one feature
family at a time, so every family's contribution is a number with an interval
around it rather than an assumption.

## Feature families

Defined in [`src/footballml/experiments/ablation.py`](src/footballml/experiments/ablation.py),
and grouped by **where the data comes from**. That is the only boundary that is
not an opinion — "attacking features" would be a judgement call with an arguable
answer for every borderline case, while provenance has a fact behind it.

| family | provenance | columns |
|---|---|---|
| `base` | the fixture list alone — which league, how much rest | 6 |
| `elo` | the Elo walk over results | 3 |
| `form` | football-data.co.uk results and match stats, rolled over 5 and 19 matches | 120 |
| `xg` | Understat: xG, npxG, deep completions, PPDA | 94 |
| `squad` | FBref + EA player ratings aggregated to a team | 12 |
| | **total** | **235** |

Exactly the production model's 235 inputs. Two allocations have any give and
both are stated rather than buried:

- **`xg_overperformance`** (goals minus xG) mixes two sources and could argue for
  either. It cannot be computed without Understat, so it is `xg`.
- **`matches_used_last_{w}`** is window *metadata* — how many matches the window
  found — not a measurement of a team. It goes to `form`, which produces it.

`league_code` sits in `base` because it is knowable before kick-off and is what
lets the two goal models learn that Serie A and the Bundesliga score at different
rates. It is why `base` is not exactly the base rate.

**The partition is enforced, not assumed.** `family_of` raises on any column it
cannot place, so a new feature breaks the experiment instead of silently falling
outside it. `tests/test_ablation.py` asserts exhaustiveness and disjointness
against the real built feature table.

## The ladder

| rung | families |
|---|---|
| 1 | `base` |
| 2 | `base` + `elo` |
| 3 | `base` + `elo` + `form` |
| 4 | `base` + `elo` + `form` + `xg` |
| 5 | `base` + `elo` + `form` + `xg` + `squad` |
| 6 | **`full`** — every production column, verbatim |

Rung 6 is deliberately *not* defined as the union of the five families. It is the
production column list as `feature_columns` returns it, so if the partition ever
stops being exhaustive the two rungs disagree and that shows up as a non-zero
difference — rather than the experiment quietly measuring a subset of the real
model while claiming to cover it.

### Secondary: leave-one-family-out

A nested ladder measures *marginal contribution given everything before it*,
which credits whichever family arrives first with information a later one would
also have carried. `elo` and `form` are both built from match results, so they
overlap heavily, and the ladder alone cannot separate them.

Leave-one-out measures the opposite bound: drop one family from the full model
and see what is lost that nothing else replaces. Reading the two together is the
only honest way to talk about an individual family's worth — a family can look
large in the ladder and near-zero on removal, which means its information was
available elsewhere.

## Methodology

Identical to production in every respect except which columns the model may see.
The experiment calls `pipelines.backtest.run_backtest` — the same function that
produces the committed backtest — rather than reimplementing it.

| | |
|---|---|
| evaluation | walk-forward: train on `Season < s`, predict `s`, advance |
| test seasons | 12, from 1516 |
| matches | 20,013 (identical fixtures for every rung, asserted) |
| model | two Poisson `HistGradientBoostingRegressor`s, home and away |
| hyperparameters | production `DEFAULT_GBM_PARAMS`, **fixed across rungs** |
| `random_state` | 7, as in production |
| post-processing | Dixon-Coles, `rho` refitted per rung on its own training data |
| calibration | temperature, fitted on accumulated out-of-sample predictions |
| missing values | untouched — handled natively by the booster |
| benchmarks | base-rate forecast and de-vigged B365 closing odds |

**Hyperparameters are fixed to production deliberately, and this shapes the
interpretation.** They were tuned for the full 235-feature model, so a
six-feature rung runs on a learning budget chosen for something much larger.
Retuning each rung would answer a different question — "the best model obtainable
from this family set" — which is not what a marginal-contribution experiment is
for. The question here is what each family adds *to the model we ship*, so the
budget stays fixed and the lower rungs should be read as a floor rather than as
the best those features could do.

### Paired bootstrap

`footballml.models.evaluate.paired_bootstrap`, the same method used for every
other accept/reject decision in the project.

Per-match RPS is computed for both models on the **same fixtures in the same
order** — checked by `align`, which fails rather than silently mispairing — and
the per-match differences are resampled 10,000 times with a fixed seed. The
reported interval is the 2.5th to 97.5th percentile of the resampled mean
difference.

Pairing is what makes an effect this small measurable at all. Football matches
differ enormously in how predictable they are, and that between-match variance
swamps the difference between two models; differencing within a fixture cancels
the shared difficulty. **A difference is only called real when the interval
excludes zero.** A point estimate with an interval straddling zero is reported as
no difference, however suggestive its sign.

Sign convention throughout: `delta = variant − baseline` on RPS, so **negative
means the variant is better**.

## Metrics

Reported per rung: RPS (calibrated and pre-calibration), log loss, Brier,
accuracy, match count, and the difference in each against both the previous rung
and `base`.

All four are means over fixtures, so pooled values are computed from the pooled
predictions rather than by averaging per-season scores — which would weight a
250-match season like an 1,826-match one.

## Running it

```bash
python -m pipelines.ablation_ladder
```

Roughly 45 minutes: ten walk-forward runs of 12 seasons each. `--quick` starts at
2324 for a smoke test (not a result), `--no-leave-one-out` skips the secondary
pass, `--n-boot` sets the resample count.

Outputs to `experiments/ablation_ladder/`: `results.csv`, `bootstrap_results.csv`,
`per_season.csv`, `config.json` (git revision, dataset size, model config, the
exact column list per family), `run.log`, `rps_ladder.png`,
`marginal_contribution.png`.

Also `predictions.csv.gz` — every rung's per-match probabilities — so a subgroup
question can be answered with a paired bootstrap instead of another 45 minutes of
walk-forward. It is gitignored at ~7MB and is exactly reproducible by re-running,
the whole pipeline being deterministic given `random_state=7`.

Plots need `pip install -e ".[experiments]"`; without matplotlib the run writes
the CSVs and skips them.

## Results

20,013 matches, 12 walk-forward seasons from 1516. Benchmarks on the same
fixtures: base-rate forecast **0.2301**, bookmakers **0.194984**.

| rung | features | RPS | RPS (pre-cal.) | log loss | Brier | accuracy |
|---|---|---|---|---|---|---|
| `base` | 6 | 0.229878 | 0.230060 | 1.071651 | 0.648091 | 44.02% |
| `+elo` | 9 | 0.201212 | 0.201148 | 0.987199 | 0.588366 | 52.49% |
| `+form` | 129 | 0.199467 | 0.199427 | 0.981867 | 0.584596 | 52.90% |
| `+xg` | 223 | 0.199330 | 0.199252 | 0.981409 | 0.584327 | 52.89% |
| `+squad` | 235 | 0.199176 | 0.199098 | 0.980907 | 0.583937 | 52.94% |
| **`full`** | 235 | **0.199176** | 0.199098 | 0.980907 | 0.583937 | 52.94% |

Calibration moves RPS by less than 0.0002 at every rung, and never changes a
conclusion, so everything below is on the calibrated numbers.

### Marginal contribution

Paired bootstrap, 10,000 resamples. Negative favours the larger model.

| step | ΔRPS | 95% CI | share of total gain | |
|---|---|---|---|---|
| `base` → `+elo` | **−0.02867** | [−0.03041, −0.02691] | **93.4%** | significant |
| `+elo` → `+form` | **−0.00175** | [−0.00239, −0.00109] | 5.7% | significant |
| `+form` → `+xg` | −0.00014 | [−0.00049, +0.00023] | 0.4% | — |
| `+xg` → `+squad` | −0.00015 | [−0.00032, +0.00001] | 0.5% | — |
| `+squad` → `full` | +0.00000 | [+0.00000, +0.00000] | 0.0% | — |

Total `base` → `full`: **−0.03070** [−0.03248, −0.02885].

### Leave-one-family-out

Dropping one family from the full model. Positive means removal hurt.

| dropped | features left | RPS | ΔRPS | 95% CI | |
|---|---|---|---|---|---|
| `elo` | 232 | 0.199696 | **+0.00052** | [+0.00018, +0.00086] | significant |
| `form` | 115 | 0.199816 | **+0.00064** | [+0.00018, +0.00109] | significant |
| `xg` | 141 | **0.199075** | −0.00010 | [−0.00045, +0.00023] | — |
| `squad` | 223 | 0.199330 | +0.00015 | [−0.00001, +0.00032] | — |

### Does `full` reproduce production?

Yes, exactly. The `full` rung scores **0.199176** against **0.199176** for the
committed `data/processed/backtest_predictions.csv` — a difference of
**8.3e-17**, which is floating-point identity rather than agreement. The ladder
runs the production pipeline, not a copy of it.

`full` is also bit-identical to `base+elo+form+xg+squad` (Δ exactly 0.00000),
which is the partition-completeness check passing: the five families really are
all 235 features.

## Interpretation

> **Superseded in part — read the Shapley section below before quoting this.**
> The 93.4% is real but is a property of Elo being added *first*, not of Elo.
> Every family recovers ~90% of the gain on its own, so whichever led the ladder
> was always going to post that number. Averaged over all orderings the four
> families are roughly equal and `form` is marginally the largest.

One feature family of three columns carries 93.4% of the total improvement over
a base-rate forecast when it is added first. The remaining 232 features split
the last 6.6% between them.

The tempting reading — "this model is essentially a well-tuned Elo" — does not
survive the Shapley attribution. What the ladder does establish is weaker and
still useful: **by the time the model has Elo, almost nothing else moves it**,
which is a statement about redundancy among all four families rather than about
Elo's primacy.

**`form` is small but real.** +5.7% on the ladder, and it survives removal from
the full model (+0.00064, interval excludes zero) — so it carries something no
other family replaces, which Elo's own docstring predicted: Elo is slow by
construction and cannot express that a side has changed recently.

**`xg` looks worthless pooled, and that number is an artefact.** 94 features —
40% of the model — with an interval spanning zero in both directions. Removing
all of them even gives the *lowest RPS of any rung in the experiment* (0.199075
against 0.199176).

The per-season breakdown says this is not a flat nothing but a sign flip:

| season | `+form` | `+form+xg` | Δ |
|---|---|---|---|
| 1516 | 0.2038 | 0.2059 | **+0.00211** |
| 1617 | 0.1935 | 0.1955 | **+0.00201** |
| 1718 | 0.1954 | 0.1954 | +0.00003 |
| 1819 | 0.1965 | 0.1966 | +0.00010 |
| 1920 | 0.2039 | 0.2039 | +0.00007 |
| 2021 | 0.2041 | 0.2027 | −0.00134 |
| 2122 | 0.1999 | 0.2001 | +0.00016 |
| 2223 | 0.2024 | 0.2016 | −0.00072 |
| 2324 | 0.1939 | 0.1930 | −0.00088 |
| 2425 | 0.1997 | 0.1971 | **−0.00260** |
| 2526 | 0.2008 | 0.2002 | −0.00061 |

xG hurts in the earliest test seasons and helps in the most recent ones, and the
cause is in the data rather than the model: **Understat coverage begins in
1415.** Predicting 1516 means training on 1011–1415, where four of five seasons
carry no xG at all, so the features are overwhelmingly missing in training and
the model fits noise. By 2425 there are ten seasons of xG behind it.

So the pooled verdict averages over test seasons in which the family barely
existed. Splitting on how many xG seasons were available *in training* — five or
fewer against six or more, which is the median and also the midpoint of the
backtest — both halves are significant, in opposite directions:

| | n | adding xG | | removing xG | |
|---|---|---|---|---|---|
| 1516–1920 (≤5 xG seasons in training) | 9,029 | **+0.00087** [+0.00029, +0.00145] | worse | **−0.00086** [−0.00142, −0.00030] | better |
| 2021–2627 (≥6) | 10,984 | **−0.00097** [−0.00141, −0.00049] | better | **+0.00052** [+0.00009, +0.00093] | worse |
| pooled | 20,013 | −0.00014 [−0.00049, +0.00023] | — | −0.00010 [−0.00045, +0.00023] | — |

The two analyses agree, which they need not have: adding xG helps in the late
era by about as much as removing it hurts there, and the early era says the
reverse. **The pooled "no effect" is two real effects of opposite sign
cancelling**, not an absence of one.

*This split is post-hoc* — it was run after seeing the per-season table, and a
subgroup chosen that way is a hypothesis rather than a confirmed result. Three
things argue it is not an artefact of looking: the split rule comes from a fact
about the data (Understat begins in 1415) that is independent of any outcome;
both halves clear zero individually; and the two independent comparisons agree in
sign and rough magnitude. It should still be confirmed on data this experiment
has not touched before it is leaned on.

**The practical reading: xG earns its place in the only era that matters.** In
the recent half, adding it is worth −0.00097 — comparable to the entire `form`
family's pooled marginal contribution (−0.00175) and *larger* than the GBM
retune that was adopted on a −0.00124 interval. The live model trains on twelve
seasons of xG and so sits firmly in that regime.

It also means the headline backtest number **understates the model as it is
shipped today**: 0.1992 is pooled across seasons in which the model was
handicapped by data that did not yet exist.

**`squad` is not significant in either direction**, which is consistent with the
5.9% of attributed SHAP movement it carries and with the decision recorded in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §12 not to build a player-level
what-if. Its leave-one-out interval ([−0.00001, +0.00032]) only just includes
zero, so "too small to resolve at this sample size" is a fairer reading than
"worthless" — but either way it cannot carry a feature that claims to price
individual players.

### Unexpected

1. **A pooled null that was two significant effects cancelling.** On the full
   backtest xG does nothing measurable; split by how much xG the model had to
   train on, both halves clear zero in opposite directions. This is the single
   most useful thing the experiment found, and it is invisible to the headline
   number. It is also a warning about the other null results here: `squad`'s
   interval is tight enough to make a hidden sign flip unlikely, but nothing in
   this experiment rules one out.
2. **The last two rungs are not significant, yet the model is better with
   them.** `+xg` and `+squad` both improve the point estimate while straddling
   zero. Accumulated, they are 0.9% of the total gain. This is what
   diminishing returns looks like when measured honestly rather than asserted.
3. **RPS and accuracy disagree at the xG step.** Adding xG improves RPS
   (0.199467 → 0.199330) while accuracy falls very slightly (52.90% → 52.89%) —
   a clean demonstration of why the project leads on RPS. The distribution got
   better while the argmax did not.

### Follow-ups this raises

1. **Does training before 1415 help or hurt?** — **measured and answered below.
   It helps.** The lead did not pan out.
2. **Should the headline backtest start later?** Weakened by that result. The
   early seasons are not an artefact of feeding the model data it cannot use;
   they are seasons in which xG genuinely had not accumulated yet. Leaving the
   headline at the full 12 seasons is the conservative and correct choice.
3. **A Shapley attribution over family subsets** — **done, see below.** It took
   16 runs rather than 32, because `base` is a floor rather than a player, and it
   changed the headline: the families are roughly equal, not Elo-dominated.

## Limitations

1. **Marginal contribution is order-dependent, and here that dominates the
   result.** The ladder's numbers are "what this family adds given the ones
   before it", not a unique contribution. All four families are different views
   of the same match results and overlap heavily, so whichever is added first
   absorbs the shared information — which is exactly what happened. Leave-one-out
   is run alongside for the opposite bound, and the Shapley section below
   resolves it properly. **Do not quote a ladder share as a family's value.**
2. **Hyperparameters are fixed to the production model**, so lower rungs are not
   independently optimised. See above.
3. **Families are groups of columns, not of information.** Dropping `xg` leaves
   `form`, which still carries shots and results — a correlated, coarser view of
   the same matches. No ablation here isolates a *concept*, only a source.
4. **One data vintage, one seed.** Boosting is deterministic given
   `random_state`, so re-running reproduces these numbers exactly, but the
   experiment does not estimate seed-to-seed variation. Differences comparable to
   seed noise should be treated with the same caution the intervals already
   impose.
5. **The test fixtures are shared with every other decision in this project.**
   These 20,013 matches have now been used to accept or reject many changes, so
   the usual multiple-comparisons caveat applies to the collection of findings,
   not to any single interval.
6. **The era split for xG is post-hoc.** It was run after the per-season table
   suggested it. The split rule is principled and independent of the outcome, and
   both halves clear zero in opposite directions, but it is a hypothesis to
   confirm on untouched data rather than a settled result. Every other number in
   this document was specified before the run.
7. **A null here means "too small to resolve at 20,013 matches", not "zero".**
   `squad`'s leave-one-out interval is [−0.00001, +0.00032]; the true effect
   could be a third of `form`'s and this experiment could not tell.

---

# Training window A/B

**Purpose.** The ablation ladder found that xG hurts in test seasons 1516–1920
and helps from 2021 on, because Understat begins in 1415 and the early
walk-forward trains on seasons where 94 of the 235 features are missing
outright. That suggested an obvious production change: the model trains from
1011, so **are those four pre-xG seasons helping, because more rows is more rows
— or hurting, because the rows are missing 40% of the feature set?**

| arm | trains on |
|---|---|
| **A** | every season before the test season — production, unchanged |
| **B** | seasons in `[1415, test)` — the xG era only |

Identical in every other respect, predicting the same 20,013 fixtures across the
same 12 test seasons, so the comparison is paired.

## Pre-specified endpoints

Written into the module docstring *before the run*, because this experiment has
an obvious way to fool itself: the ladder had already shown the two eras differ,
so picking the favourable era afterwards would be choosing the answer.

- **Primary** — pooled RPS, paired bootstrap, adopted only if the interval
  excludes zero in arm B's favour. The same gate every other change has faced.
- **Secondary, reported either way** — the ladder's era split. Expected to move,
  since arm B discards most of the early seasons' training data and almost none
  of the late seasons'. For understanding, not for deciding.

`run_backtest` gained a `train_from` argument for this. It defaults to `None`,
and `tests/test_ablation.py` asserts that the default path is byte-identical to
production — otherwise the A/B would be measuring its own change.

## Results

| arm | train_from | median train rows | RPS | log loss | Brier | accuracy |
|---|---|---|---|---|---|---|
| **A** | all | 19,072 | **0.199176** | 0.980907 | 0.583937 | **52.94%** |
| B | 1415 | 11,768 | 0.200075 | 0.983729 | 0.585895 | 52.55% |

Paired bootstrap, 10,000 resamples; **positive favours arm A**:

| subset | n | ΔRPS | 95% CI | |
|---|---|---|---|---|
| **PRIMARY pooled** | 20,013 | **+0.00090** | [+0.00047, +0.00133] | significant |
| secondary early 1516–1920 | 9,029 | +0.00141 | [+0.00058, +0.00223] | significant |
| secondary late 2021–2627 | 10,984 | +0.00048 | [+0.00009, +0.00087] | significant |

**Verdict: rejected. Production keeps all history.**

## Interpretation

**The hypothesis was wrong, and not marginally.** Arm B is worse on every metric
and in every subset, including the late era where xG is fully mature and arm B
discards almost nothing. That last row is the one that settles it: if pre-xG rows
were harmful, dropping them should have helped most where their absence costs
least, and instead it still hurt by +0.00048 with an interval clear of zero.

The reason is that **missing is not the same as harmful.** The booster handles
NaN natively, so a 2011 row with no xG still teaches it everything else — Elo,
form, shots, rest — and simply abstains on the xG splits. Those rows are
partial, not poisoned. Removing them costs 7,304 training rows, 38% of the
median training set, and that loss comfortably outweighs any benefit from a
cleaner feature matrix.

It also sharpens what the ladder actually found. xG underperforming in early
test seasons is **not** a fixable data-curation problem. It is that there was not
yet enough xG history to learn from, and the only remedy for that is time, which
has already happened: the live model has twelve seasons of it.

So both experiments point the same way on the production model — **change
nothing** — while explaining a result that looked like it demanded a change.

## Running it

```bash
python -m pipelines.training_window_ab
```

About nine minutes. Outputs to `experiments/training_window_ab/`: `results.csv`,
`bootstrap_results.csv`, `per_season.csv`, `config.json`.

## Limitations

1. **One cutoff tested.** 1415 is the principled one — it is exactly where
   Understat begins — but the result does not establish that *no* cutoff helps.
   A weighting scheme that keeps old rows at reduced influence was not tried.
2. **Fewer rows and fewer xG-less rows are confounded.** Arm B changes both at
   once, so this measures the net effect of the change a practitioner would
   actually make, not the isolated effect of either. Separating them would need
   an arm that drops a matched number of *recent* rows instead.
3. **Same shared test set** as everything else in this project; the
   multiple-comparisons caveat under the ladder applies here too.

---

# Shapley attribution over feature families

**Purpose.** The ladder and leave-one-out disagreed, and neither is a family's
value. This runs **every** coalition of the four families — 16 walk-forward
backtests, `base` present throughout as the floor — and averages each family's
marginal contribution over all orderings. That is the unique attribution
satisfying efficiency, symmetry, dummy and linearity.

Value function is *gain*: `v(S) = RPS(base) − RPS(S)`, so `v(∅) = 0` and the
values sum to the full model's improvement over `base`.

Intervals are exact, not approximate. A Shapley value is a linear combination of
mean per-match differences, so it rewrites as the mean of a per-match quantity
and the project's usual paired bootstrap applies to that directly.

## Results

Efficiency holds: the four values sum to **0.03070**, the measured total gain.

| family | Shapley value | 95% CI | share | |
|---|---|---|---|---|
| **`form`** | **+0.00873** | [+0.00807, +0.00938] | **28.4%** | significant |
| `elo` | +0.00815 | [+0.00755, +0.00874] | 26.5% | significant |
| `xg` | +0.00776 | [+0.00714, +0.00836] | 25.3% | significant |
| `squad` | +0.00607 | [+0.00559, +0.00653] | 19.8% | significant |

All sixteen coalitions:

| coalition | features | RPS |
|---|---|---|
| `(base only)` | 6 | 0.229878 |
| `elo` | 9 | 0.201212 |
| `form` | 126 | 0.200359 |
| `xg` | 100 | 0.201585 |
| `squad` | 18 | 0.206690 |
| `elo+form` | 129 | 0.199467 |
| `elo+xg` | 103 | 0.200063 |
| `elo+squad` | 21 | 0.201103 |
| `form+xg` | 220 | 0.200118 |
| `form+squad` | 138 | 0.199544 |
| `xg+squad` | 112 | 0.200799 |
| `elo+form+xg` | 223 | 0.199330 |
| `elo+form+squad` | 141 | **0.199075** |
| `elo+xg+squad` | 115 | 0.199816 |
| `form+xg+squad` | 232 | 0.199696 |
| `elo+form+xg+squad` | 235 | 0.199176 |

## Interpretation

**This overturns the ladder's headline, and the ladder was the misleading one.**

| family | ladder share | Shapley share | ladder significant | Shapley significant |
|---|---|---|---|---|
| `elo` | **93.4%** | 26.5% | yes | yes |
| `form` | 5.7% | **28.4%** | yes | yes |
| `xg` | 0.4% | 25.3% | **no** | **yes** |
| `squad` | 0.5% | 19.8% | **no** | **yes** |

Averaged over all orderings the four families are **roughly equal**, every one
of them significant, and `form` is marginally the largest — not `elo`.

The singleton coalitions say why. Each family, **on its own**, gets most of the
way to the full model:

| family alone | RPS | gain | % of the full model's gain |
|---|---|---|---|
| `form` (126) | 0.200359 | 0.029520 | **96%** |
| `elo` (9) | 0.201212 | 0.028666 | **93%** |
| `xg` (100) | 0.201585 | 0.028294 | **92%** |
| `squad` (18) | 0.206690 | 0.023189 | 76% |

So whichever family the ladder happened to add first was always going to show
~93%, and everything after it was always going to look negligible. **Elo was not
special; it was first.** Had `form` led the ladder it would have scored 96%.

The redundancy can be put as one number. The solo gains sum to **0.109669**; the
four families combined deliver **0.030702**. Only **28% survives combination** —
about **72% of what each family carries is already present in the others**.

That is not surprising once stated: Elo, rolling form, xG rates and squad
ratings are four different views of the same match results. They are substitutes
far more than complements, which is exactly the structure a nested ladder cannot
express and a Shapley value is built for.

**`squad` is rehabilitated, with a caveat.** The ladder could not distinguish it
from zero; here it is 19.8% of the attributed gain with an interval comfortably
clear of zero, and alone it reaches 76% of the full model's gain from 18
columns. Both results are correct and they answer different questions: squad
quality carries real predictive information, *and* by the time the model has
Elo, form and xG it adds almost nothing on top. The first fact does not license
a feature that prices individual players; the second is still why
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §12 declines one.

**The best coalition is not the full model.** `elo+form+squad` scores 0.199075
on 141 features against 0.199176 for all 235 — consistent with the ladder's
finding that `full-minus-xg` was the best rung there. Pooled across the whole
backtest, the xG family is carrying its weight only in the recent era.

### What this means for how the model is described

"This model is essentially a well-tuned Elo" was the ladder's reading and it is
**not supported**. The defensible statement is:

> Four largely redundant views of match results, any one of which recovers
> ~90% of the model's edge on its own, combined for a last ~7% that none of
> them reaches alone.

### Unexpected

1. **A 9-column family and a 126-column family are worth the same.** `elo` and
   `form` differ by a factor of fourteen in width and by 2 percentage points in
   Shapley share. Feature count is not a proxy for contribution anywhere in this
   table.
2. **Every family is significant under Shapley; two were not under the ladder.**
   The ladder's nulls were an artefact of position, not evidence of absence —
   which is the general warning: a marginal contribution measured at the end of
   a nested sequence is a lower bound on a family's worth, never its value.

## Running it

```bash
python -m pipelines.shapley_families
```

About 30 minutes for 16 walk-forward runs. Outputs to
`experiments/shapley_families/`: `coalitions.csv`, `shapley.csv`, `config.json`,
`shapley_values.png`.

## Limitations

1. **`base` is a floor, not a player.** The attribution is over the four
   quality-bearing families; league identity and rest are in every coalition.
   Including them would be 32 runs and a degenerate `v(∅)`.
2. **Shapley answers "value averaged over orderings", which is not "value to
   the model as built".** For a deletion decision, leave-one-out is the relevant
   number and it is far smaller. These do not conflict; they answer different
   questions, and quoting whichever flatters a family would be the error.
3. **Hyperparameters fixed to production** for all 16 coalitions, as in the
   ladder. A 9-feature coalition runs a budget tuned for 235.
4. **Same shared test set** as every other decision here.
