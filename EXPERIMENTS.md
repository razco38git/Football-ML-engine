# Experiments

Measurements that answer a question once and write down the answer. Nothing here
runs on a schedule and nothing the site serves depends on it.

Results that changed the production model live in the docstring of the thing they
changed — that is deliberate, so a constant and its justification cannot drift
apart. This file is for experiments whose output is a *finding* rather than a
code change.

---

# Reading an attribution number

**This section exists because the first version of the ablation ladder below was
reported wrongly, and the mistake is one anybody would make.** It governs how to
read everything in this file.

The ladder measured Elo at **93.4%** of the model's gain. That number is
correct, and it is not the answer to "how much of the model's strength comes
from Elo" — which is what it was presented as. It is the answer to "what does
Elo add when it is the *first* thing added". Every family recovers ~90% of the
gain on its own, so whichever went first was always going to post that figure.

Three different questions, three different measurements, all true at once:

| question | measurement | Elo's answer |
|---|---|---|
| Is it **sufficient**? | the family alone, against `base` | **93%** of the total gain |
| What is its **fair share**? | Shapley, over all orderings | **26.5%** |
| Is it **necessary**? | leave-one-out from the full model | **+0.00052**, about 1.7% |

Elo is nearly sufficient, moderately credited, and barely necessary. Those do not
conflict: `form`, `xg` and `squad` can each stand in for most of what it does, so
removing it costs little while having it alone gets you most of the way.

**The general rule: a marginal contribution measured at the end of a nested
sequence is a lower bound on a family's worth, never its value.** Quoting a
ladder share as an attribution is the error. This is not a quirk of this project
— it is the same reason SHAP exists rather than "drop a feature and re-measure".

### Then why run a ladder at all

It answers a question Shapley does not, and the two should not be swapped:

1. **The engineering question.** "I already have Elo and form — is it worth
   building and maintaining 94 xG columns?" That is marginal contribution, and
   the ladder's +0.00014 is the right answer. Shapley's 25.3% is the *wrong*
   answer to it, and would justify work that buys nothing.
2. **Cost.** The ladder is six runs, leave-one-out four more. Shapley is 2^n —
   sixteen here, but 256 at eight families. The ladder is the cheap scout.
3. **Its shape is itself diagnostic.** A steep first step and a flat tail *is* a
   redundancy signature; complementary families would descend steadily. The
   right reaction to "rung one takes 93% and the rest are flat" is *"would any
   first family do that?"* — which is the question Shapley answers, and which
   should have been asked before the 93.4% was written down as a headline.
4. **Diminishing returns are real.** The ladder correctly shows the model is
   saturated after the first family, which is what decides where to stop.

So: the ladder for *what to build next*, leave-one-out for *what to delete*,
Shapley for *what to credit*. Reaching for whichever flatters a family is the
only way to get this wrong.

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

Both measurements were subsequently rerun on the late era alone — see the final
section — where xG passes the ladder *and* the leave-one-out test, both clear of
zero. That is a stronger result than the pooled one, and it is **still the same
matches**, so it does not discharge this caveat.

**The practical reading: xG earns its place in the only era that matters.** In
the recent half, adding it is worth −0.00097 — comparable to the entire `form`
family's pooled marginal contribution (−0.00175) and *larger* than the GBM
retune that was adopted on a −0.00124 interval. The live model trains on twelve
seasons of xG and so sits firmly in that regime.

It also means the headline backtest number **understates the model as it is
shipped today**: 0.1992 is pooled across seasons in which the model was
handicapped by data that did not yet exist.

**`squad` is not significant in either direction here**, consistent with the 5.9%
of attributed SHAP movement it carries and with the decision in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §12 not to build a player-level
what-if. Its leave-one-out interval ([−0.00001, +0.00032]) only just includes
zero, so "too small to resolve at this sample size" is a fairer reading than
"worthless".

Read that alongside the Shapley section, which puts `squad` at 19.8% with an
interval well clear of zero: it carries real information and is redundant with
the other three. Both facts hold, and it is the *redundancy* — not an absence of
signal — that makes a per-player perturbation meaningless.

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
4. **One data vintage — but there is no seed variance to worry about.**
   `random_state` is *inert* at this data size: sklearn uses it only to
   subsample when choosing bin thresholds, and only above 200,000 rows, where
   training has 25–29k. Seeds 7, 123 and 999 give bit-identical predictions.
   So re-running reproduces every number here exactly rather than approximately,
   and a multi-seed robustness check would print the same figure five times. The
   remaining vintage caveat is the data, not the estimator.
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

## Per season

The pooled figure above is a weighted mean of these, and reproduces from them to
5.3e-17 — so the table and the headline cannot disagree.

| season | n | A (all) | B (1415) | Δ | A train | B train | B keeps |
|---|---|---|---|---|---|---|---|
| 1516 | 1826 | 0.2059 | 0.2079 | **+0.00196** | 9,130 | 1,826 | 20% |
| 1617 | 1826 | 0.1952 | 0.1972 | **+0.00201** | 10,956 | 3,652 | 33% |
| 1718 | 1826 | 0.1951 | 0.1954 | +0.00025 | 12,782 | 5,478 | 43% |
| 1819 | 1826 | 0.1964 | 0.1975 | +0.00109 | 14,608 | 7,304 | 50% |
| 1920 | 1725 | 0.2034 | 0.2052 | **+0.00175** | 16,434 | 9,130 | 56% |
| 2021 | 1826 | 0.2023 | 0.2031 | +0.00080 | 18,159 | 10,855 | 60% |
| 2122 | 1826 | 0.2003 | 0.2003 | −0.00001 | 19,985 | 12,681 | 63% |
| 2223 | 1826 | 0.2019 | 0.2021 | +0.00016 | 21,811 | 14,507 | 67% |
| 2324 | 1752 | 0.1927 | 0.1931 | +0.00048 | 23,637 | 16,333 | 69% |
| 2425 | 1752 | 0.1973 | 0.1975 | +0.00017 | 25,389 | 18,085 | 71% |
| 2526 | 1752 | 0.1999 | 0.2012 | +0.00128 | 27,141 | 19,837 | 73% |
| 2627 | 250 | 0.2027 | 0.2033 | +0.00059 | 28,893 | 21,589 | 75% |

**Eleven of twelve seasons are worse under arm B**, and the twelfth is tied at
−0.00001. Sign test, two-sided: **p = 0.0063**.

Note what the last three columns show. Arm B always discards exactly the same
**7,304 rows** — seasons 1011–1314, four times 1,826 — so the absolute loss is
constant while the *share* it represents falls from 80% of the training set in
1516 to 25% by 2627.

The harm falls roughly in step: mean Δ is **+0.00131** across the first six
seasons against **+0.00045** across the last six. Spearman between the harm and
the share of rows kept is **−0.47**, the direction the row-loss explanation
predicts — but **p = 0.124**, so with twelve seasons this is suggestive and not
established, and it is reported as such.

## Interpretation

**The hypothesis was wrong, and not marginally.** Arm B is worse on every metric
and in every subset, including the late era where xG is fully mature and arm B
discards almost nothing. The per-season table makes that a stronger claim than a
single interval can: eleven of twelve seasons move the same way, so the verdict
cannot be the work of one unusual season. That last row is the one that settles it: if pre-xG rows
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

---

# What a frozen holdout could and could not do

The obvious objection to everything in this file is that these 20,013 matches
have gated many decisions, so the headline is not an untouched estimate. That is
correct, and it is the project's real remaining evaluation weakness. The usual
remedy — freeze the last two seasons, never experiment on them — does not work
here, and the arithmetic is worth writing down so it is not proposed again.

2025/26 and 2026/27 are **2,002 matches, 10% of the backtest**. Intervals scale
with 1/sqrt(n), so they widen by sqrt(20013/2002) = **3.16x**. Applying that to
the effects this project has actually measured:

| change | on 20,013 | on a 2,002-match holdout | |
|---|---|---|---|
| GBM retune, the largest ever adopted | −0.00124 [−0.00158, −0.00090] | [−0.00231, −0.00017] | still clears |
| `form`, ladder step | −0.00175 [−0.00239, −0.00109] | [−0.00381, +0.00031] | **does not clear** |
| xG, late era | −0.00097 [−0.00141, −0.00050] | [−0.00241, +0.00047] | **does not clear** |

So a frozen historical holdout could confirm the headline RPS and almost nothing
else. It could not have adjudicated two of this project's three largest
findings, and it would cost two seasons of training data to buy that.

**The project already has the better instrument, and it is prospective rather
than merely unused.** `store.py` is append-only, stamped with the model version
and never rewritten, so a prediction cannot be regenerated after the fact. That
is a stronger guarantee than an untouched historical slice, because the matches
had not been played when the forecast was recorded.

Its limitation is sample, not design: **16 settled predictions**, first recorded
2026-09-15. At roughly 50–60 settled matches a week in season it needs most of a
season to say anything useful about the headline, and it will never resolve a
0.0001-level change. That is a reason to keep recording and wait, not a reason
to carve up history.

---

# The same two measurements, on the xG era only

**Purpose.** The ladder found xG's contribution is not constant across the
backtest — significantly harmful in 1516–1920, significantly helpful from 2021.
If that is right, both the ladder and the Shapley attribution should look
different when the test set is restricted to the modern era, and the ladder
should change more than the Shapley.

**Design.** Training is left exactly as production: all history from 1011. The
training-window A/B already showed that restricting it is significantly worse,
so the only change here is which seasons are *tested* — 2021 onward, 10,984
matches, the same split used for the era analysis above. Everything else is
untouched.

```bash
python -m pipelines.ablation_ladder   --start-season 2021 --out-dir experiments/ablation_ladder_xg_era
python -m pipelines.shapley_families  --start-season 2021 --out-dir experiments/shapley_families_xg_era
```

## Ladder

| rung | features | RPS | Δ vs previous | |
|---|---|---|---|---|
| `base` | 6 | 0.231043 | — | |
| `+elo` | 9 | 0.202262 | **−0.02878** [−0.03101, −0.02657] | significant |
| `+form` | 129 | 0.200230 | **−0.00203** [−0.00280, −0.00125] | significant |
| `+xg` | 223 | 0.199260 | **−0.00097** [−0.00141, −0.00050] | **significant** |
| `+squad` | 235 | 0.199212 | −0.00005 [−0.00028, +0.00018] | — |

Leave-one-out: `elo` **+0.00055** [+0.00012, +0.00097] significant, `xg`
**+0.00052** [+0.00010, +0.00093] significant, `form` +0.00039 [−0.00008,
+0.00087] not significant, `squad` +0.00005 [−0.00018, +0.00028] not
significant.

## xG, the two eras side by side

| | full backtest (20,013) | xG era (10,984) |
|---|---|---|
| ladder `+xg` step | −0.00014 [−0.00049, +0.00023] — | **−0.00097** [−0.00141, −0.00050] **significant** |
| leave-one-out `xg` | −0.00010 [−0.00045, +0.00023] — | **+0.00052** [+0.00010, +0.00093] **significant** |
| `full-minus-xg` RPS | **0.199075** (best rung in the experiment) | 0.199731 (against 0.199212 for full) |

Pooled, dropping all 94 xG columns gave the *best* number in the experiment. On
the modern era the same deletion is clearly harmful, and both tests now agree in
the same direction. Same features, same code, different test seasons.

## Shapley, the two eras side by side

Efficiency holds in the xG era: the values sum to **0.03183**, the measured total
gain. All four families significant in both eras.

| family | full backtest | xG era | change |
|---|---|---|---|
| `elo` | 26.5% | 25.1% | −1.5 pts |
| `form` | 28.4% | 26.6% | −1.8 pts |
| **`xg`** | 25.3% | **27.4%** | **+2.1 pts** |
| `squad` | 19.8% | 21.0% | +1.2 pts |

Each family alone, as a share of its era's total gain:

| era | `elo` | `form` | `xg` | `squad` |
|---|---|---|---|---|
| full backtest | 93% | 96% | 92% | 76% |
| xG era | 90% | 94% | **95%** | 80% |

## Interpretation

**The era finding holds, and on the measurement that matters it is now
unambiguous.** In the modern era xG passes both tests — adding it helps,
removing it hurts, both intervals clear of zero — where pooled it passed
neither.

**The ladder moved a lot and the Shapley barely moved, which is the point of
having both.** xG's ladder step went from indistinguishable from zero to
significant, a roughly sevenfold change in the point estimate; its Shapley share
moved 2.1 points. They are measuring different things, exactly as the opening
section says:

- the **ladder** measures marginal value *in the model as built*, which depends
  on what else is available and on the era — so it is the number that should
  move, and does;
- the **Shapley** measures a family's share of credit given the redundancy
  structure, and that structure has not changed, so it should be stable — and is.

**The redundancy is unchanged, if anything slightly stronger.** Every family
still recovers 80–95% of the gain alone, with xG now essentially tied with form
as the best single family. So the summary stands: four largely substitutable
views of match results. What the era changes is *which* of them carries the
marginal load, not that they overlap.

`+squad` remains non-significant on the ladder in both eras, and significant in
both Shapley attributions. Nothing about the era split touches the reasoning in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §12.

## Limitations

1. **This is not independent confirmation of the era effect.** The split was
   chosen post-hoc from the pooled run, and this rerun uses the same matches, the
   same features and the same model. It confirms that the effect is large enough
   to dominate a measurement restricted to that era — which was not guaranteed —
   but it cannot confirm the split itself. Only seasons this project has not
   tested can do that, and by construction they do not exist yet.
2. **Smaller sample.** 10,984 matches against 20,013, so every interval here is
   wider than its pooled counterpart. The `+squad` step and `form`'s
   leave-one-out are both non-significant in this era and significant or
   near-significant pooled, which is at least partly sample size rather than a
   finding.
3. **Hyperparameters remain fixed to production**, as in both parent
   experiments.
