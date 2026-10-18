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

---

# The defending sub-rating

## Why

A reader asked why Arsenal, the best defence in Europe, rate 79.6, and why
William Saliba reads 32 for "Defending". Measuring it produced something worse
than the question. Across 859 team-seasons, the minutes-weighted back-line
aggregate against that club's own defensive record:

| back-line aggregate of | vs goals conceded | vs xG conceded |
|------------------------|-------------------|----------------|
| full `rating`          | **-0.533**        | -0.531         |
| `fifa_overall` (EA)    | -0.516            | -0.492         |
| `sub_involvement`      | -0.499            | -0.499         |
| **`sub_defending`**    | **+0.097**        | **+0.102**     |

`sub_defending` is not weak, it is **backwards**: teams whose defenders rank
higher on "defending" concede more. It was 62% of a centre-back's composite, and
every one of its five inputs reduces to three raw counts — tackles won,
interceptions, fouls — because `clean_challenge_rate` and `ball_won_per_foul`
are derived from those same three. It counts defensive *actions*, and a defender
makes more of them when his side defends worse. `_add_possession_adjusted` is
meant to remove that bias and under-corrects.

New inputs are not available. FBref now serves tackles, clearances, blocks and
errors with every cell blank in every season checked (the numbers are recorded
beside `DEFENSIVE_VOLUME_METRICS` in `src/footballml/players/ingest.py`), and a
paid provider was declined.

## What was run

`pipelines/defending_sweep.py`, two independent ways to fix it, over the same
20,074 walk-forward fixtures (1516-2627):

- **`share`** — cut the weight on a backwards signal. CB `defending` from 0.62
  down to 0, the freed weight redistributed across involvement, creation and
  finishing in their existing proportions.
- **`outcome`** — give defenders what the goalkeeper already has. `GK` is rated
  97% on outcomes (`goals_against_per90: -2.0`, `clean_sheet_pct`) while the
  centre-backs in front of him were rated 0% on them. Adds a `team_defence`
  sub-rating built from the club's own `xGAgainst`, `GoalsAgainst` and clean
  sheet rate, taken out of `defending`.

Judged on both: walk-forward RPS with a paired bootstrap (10,000 resamples,
seed 7), and the back line against the **next** season's goals conceded — next,
not same, because that is the only direction the model reads strength, and
because a same-season figure would mark the `outcome` variant's own homework.

### Two runs were measured and thrown away

**Run 1 — the club-name join.** `add_team_defence` joined the player frame's
Understat club spellings ("Manchester City", "Atletico Madrid", "Paris Saint
Germain") against the match history's football-data spellings ("Man City",
"Ath Madrid", "Paris SG"). A left join does not fail when it matches nothing,
it fills NaN, so **26.6% of defenders silently had no team-defence value** —
worst at Milan, Atletico, PSG, both Manchester clubs and Athletic, which is the
worst possible place for it to fail. `add_team_strength` has carried
`load_aliases("understat")` for exactly this reason all along. That run chose
`team_defence: 0.30` at delta -0.00045, CI [-0.00063, -0.00027], comfortably
significant. **Superseded in full; none of its numbers appear below.**

**Run 2 — the peak was ignoring zero weights.** `peak_weight: 0.15` blends in a
player's best sub-rating, and it took the maximum over *every* defined
sub-rating, including ones weighted zero. So a weight of zero kept a group out
of the composite but not out of the published rating. The tell was that
rebuilding moved **807 of 3,904 full backs** by up to 4 points with no
full-back weight and no full-back sub-rating changed: a left back at a
mean-defence club was being credited for being outstanding at his club's clean
sheets. It meant run 2's `baseline` was never production behaviour — it already
carried part of the effect being measured — and it reversed the full-back
conclusion entirely. **Superseded in full.**

The fix narrows the peak to the sub-ratings a role is actually weighted on
(`_peak_over_rated_sub_ratings`). In isolation it leaves every outfield role
bit-identical to production and moves only keepers, through their zero-weighted
`workload`: 459 of 1,799 move, mean delta -0.001, worst case 5 points.

Everything below is run 3, on a join at **100%** of 34,739 player-seasons and a
peak that respects a zero weight.

## Results

| variant | CB defending | team_defence | RPS | delta | 95% CI | defence vs next GA |
|---------|--------------|--------------|-----|-------|--------|--------------------|
| share-0.62 *(the setting this replaced)* | 0.62 | 0.00 | 0.19911 | -0.00001 | [-0.00014, +0.00013] | -0.440 |
| share-0.45 | 0.45 | 0.00 | 0.19917 | +0.00005 | [-0.00008, +0.00019] | -0.458 |
| share-0.30 | 0.30 | 0.00 | 0.19906 | -0.00006 | [-0.00020, +0.00008] | -0.468 |
| share-0.15 | 0.15 | 0.00 | 0.19918 | +0.00006 | [-0.00008, +0.00020] | -0.478 |
| share-0.00 | 0.00 | 0.00 | 0.19914 | +0.00003 | [-0.00011, +0.00017] | -0.486 |
| outcome-0.15 | 0.47 | 0.15 | 0.19913 | +0.00002 | [-0.00012, +0.00015] | -0.490 |
| **outcome-0.30** *(shipped; the baseline here)* | 0.32 | 0.30 | 0.19912 | — | — | **-0.523** |
| outcome-0.45 | 0.17 | 0.45 | 0.19908 | -0.00003 | [-0.00017, +0.00010] | -0.544 |

**Not one variant is significant on RPS, and the whole table spans 0.00012.**
The baseline here is the shipped setting, so `outcome-0.30` is the same
configuration measured twice and returns a delta of exactly 0.00000 — the
determinism check — as do `baseline` and `cb-0.30-only` when the second stage
re-runs them in a separate process. `share-0.62` is the configuration this
change replaced, which is the comparison that matters: +0.00001 on RPS,
-0.440 to -0.523 on defensive validity.

The defence correlation is monotone in both families, which is the one result
that has held across all three runs and both defects.

## Defensive validity, split by role

Full-back weights are identical in the first three rows, so `FB only` acts as a
control and should not move — and does not.

| variant | CB only | FB only | CB+FB+D | EA agreement (CB) | EA agreement (FB) |
|---|---|---|---|---|---|
| baseline | -0.381 | -0.396 | -0.433 | +0.778 | +0.815 |
| share-0.00 | -0.463 | -0.396 | -0.477 | +0.803 | +0.815 |
| **outcome-0.30** | **-0.544** | -0.396 | -0.526 | **+0.833** | +0.815 |
| cb-0.30+fb-0.30 | -0.544 | -0.509 | -0.559 | +0.833 | +0.829 |

xGA and npxGA agree: `outcome-0.30` reaches -0.558 and -0.531 on the CB line.
It improves in **all five leagues on all three outcome measures** — Bundesliga
-0.645 to -0.661, Premier League -0.433 to -0.517, Ligue 1 -0.325 to -0.395,
Serie A -0.594 to -0.665, La Liga -0.531 to -0.621.

That EA's independently scouted `fifa_overall` also moves closer, +0.778 to
+0.833, is the part hardest to explain as fitting our own second metric.

## The RPS numbers this section used to carry were not RPS

Everything above was re-measured on 2026-10-18. Until then this section
reported scores near **0.365**, against the figure `evaluate.py` documents for a
naive base rate — **0.22**. That gap was noticed twice and explained away as "a
different normalisation". It was not.

`pipelines/defending_sweep.py` scored through `rps_per_match` with integer
outcome labels:

```python
truth = preds["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
```

`_one_hot` encodes outcomes by comparing against the strings `'H'`, `'D'`, `'A'`.
Integers match none of them, so every indicator row came back zero and RPS
reduced to

    (p_H**2 + (p_H + p_D)**2) / 2

— a function of the forecast that never reads the result. Measured on the 827
scored European ties: 0.36316 with integer labels, **0.36316 after randomly
shuffling every outcome**, and 0.36316 from the closed form above. Correctly
scored, the same forecasts give 0.21691, and 0.24877 when shuffled.

So the sweep ranked its arms by how little probability they placed on a home
win. Every RPS figure, delta, confidence interval and per-season breakdown this
section previously carried measured that and nothing else, including a
`cb-0.30+fb-0.30` result reported as significant at -0.00046.

**What survived.** The decision did. It was taken on the defensive-validity
correlations in the section above, which come from `pandas.corr` and never
touched the broken metric — and it was taken *because* RPS could not separate
the arms, which the corrected numbers confirm far more emphatically than the
broken ones did. Nothing else in the repo was affected: `ablation_ladder`,
`training_window_ab`, `shapley_families`, `league_adjust` and `evaluate.py`
itself all pass the raw `FTR` column, so the site's published backtest figure,
the cross-league holdout and the accuracy page were never wrong.

**What changed on the corrected numbers.** Two claims this section made are
withdrawn:

- *"`outcome-0.45` predicts significantly worse."* It does not. It is -0.00003
  against the shipped setting, nominally the best RPS in the table, and not
  significant. It remains unshipped on the ceiling argument alone.
- *"The full-back extension's gain fails to replicate season by season."* There
  was no gain to replicate: fb-0.30 is +0.00001, not significant. The
  per-season table that carried this argument has been deleted rather than
  recomputed, because the argument no longer needs it.

**What the corrected table says instead.** Nine arms spanning
**0.19906 to 0.19918** — a total spread of 0.00012 — with not one interval
excluding zero. Against the setting it replaced, the shipped change is
+0.00001. The measurement floor recorded elsewhere in this file was, if
anything, generous: RPS cannot distinguish any of these configurations, which is
why the choice rests entirely on defensive validity.

`_one_hot` now raises on any label that is not one of `OUTCOMES`, naming the
offenders, and `tests/test_ablation.py` pins both that and the property the bug
removed — that shuffling the outcomes must change the score.

## Decision

**Ship `CB team_defence: 0.30` with `defending: 0.32`. Leave full backs at
zero.** The centre-back case rests on defensive validity, which replicates
across every league and every outcome measure, with RPS neutral — not a gain,
and no evidence of harm.

**Why not the full-back extension**, despite better aggregates on every line.
Its only advantage over the centre-back change was the RPS gain, and that is the
claim that failed the consistency check. What is left is a validity gain bought
at a steep editorial price: the full-back pool is 0.38, so taking 0.30 leaves a
full back rated **8% on ball-winning and 30% on his club's clean sheets**. It
was also the worst variant before the peak fix, so its case depends entirely on
that fix, and 0.30 is the edge of the tested grid. It is a reasonable product
decision, not a statistical conclusion, and it should be taken as one.

**Why not `share-0.00`.** It is the worst of the four on RPS and clearly worse
on validity (-0.463 against -0.544), and it leaves a centre-back's composite
made entirely of involvement, creation and finishing — **no defensive content at
all** — behind a bar the page labels "Ball-winning". It also moves the wrong
clubs: Werder Bremen's back line *improves* from 66.5 to 69.0, rank 51st to
36th. What it does establish, and it is the more important finding, is that
deleting 62% of a centre-back's composite does not make the model worse. That is
what a backwards signal looks like from the other side.

**Why not 0.45.** It buys the best correlation of the `outcome` arm, -0.544, and
predicts no better. Past some point the club's record crowds out what the rating
knows about the player.

## What it does to the published ratings

2025/26, the last completed season:

| defender | club `team_defence` | before | after |
|---|---|---|---|
| William Saliba (Arsenal) | 98 | 79 | **85** |
| Gabriel (Arsenal) | 98 | 80 | **85** |
| Rúben Dias (Manchester City) | 90 | 77 | **83** |
| Alessandro Bastoni (Inter) | 95 | 84 | **87** |
| Marquinhos (PSG) | 96 | 77 | **81** |
| Virgil van Dijk (Liverpool) | 59 | 79 | **83** |
| Marc Guehi (Crystal Palace) | 53 | 83 | **81** |
| Tiago Gabriel (Lecce) | 41 | 65 | **62** |

Arsenal's back line goes 77.0 to 79.6, 7th to 4th of 96, and their
`strength_defence` 79.6 to 82.8, second only to Bayern. Every full back is
unchanged. The distribution barely moves: CB mean 68.46 to 68.43, sd 6.74 to
7.23, p95 80 to 82.

## Limitations

1. **It is not a prediction gain, and is not claimed as one.** The RPS delta is
   -0.00010 with a CI including zero, and the variant that did reach
   significance failed to replicate season by season. `strength_defence` ranks
   93rd of 125 features in the match model; the case is that the published
   ratings stop being indefensible.
2. **`team_defence` is a club number and cannot be read as an individual
   contribution.** Saliba and Gabriel both take Arsenal's 98 and both land on
   85; it separates Arsenal's back line from Werder Bremen's and cannot separate
   teammates. It makes a player's rating depend partly on who he plays with,
   which is a real change to what the number means and is not how EA's ratings
   work.
3. **Marc Guehi pays for it**, 83 down to 81 — a well-regarded defender charged
   for a leaky side, the mirror image of the Van Dijk complaint that started
   this. That is the honest cost of a club-level component, not a bug.
4. **The weight was chosen post-hoc from thirteen variants across three runs**,
   so any winning margin is optimistically biased. The parts that do not depend
   on the choice are the monotone correlation trend, the 15-of-15 league and
   metric consistency, and the `share-0.00` result.
5. **The second metric is the one that decided it**, and it is a correlation
   against an outcome the component is partly built from, one season removed.
   Next-season framing and the full-back control guard against the obvious
   failure; they do not make it causal evidence about individual defenders.
6. **Hyperparameters fixed to production**, as in every other experiment here.
7. **The baseline levels are not interchangeable with older notes.** The -0.489
   in `config/player_rating.yaml` came from 1,052 completed team-seasons with a
   different aggregation. Every number above comes from one function applied
   identically across variants, so the comparisons hold; the levels do not.

## Second stage: do full backs want the same thing?

They share the centre backs' defending definition exactly, so they share its
defect. Measured on top of the chosen setting, via `--stage fb --cb-share 0.30`:

| variant | CB team_defence | FB team_defence | RPS | delta | 95% CI | defence vs next GA |
|---------|-----------------|-----------------|-----|-------|--------|--------------------|
| **cb-0.30 only** *(shipped; the baseline here)* | 0.30 | 0.00 | 0.19912 | — | — | -0.523 |
| cb-0.30 + fb-0.15 | 0.30 | 0.15 | 0.19915 | +0.00003 | [-0.00009, +0.00016] | -0.541 |
| cb-0.30 + fb-0.30 | 0.30 | 0.30 | 0.19913 | +0.00001 | [-0.00012, +0.00014] | **-0.554** |

Before the peak fix this table ran the other way: every step toward full backs
improved the correlation and made RPS worse, monotonically. That was an artefact
of the leak — FB `team_defence` was already reaching every variant including
baseline, so adding weight was partly double-counting.

With the leak closed and the metric fixed, the full-back extension is
**RPS-neutral** — +0.00001 against the shipped setting, not significant — while
the correlation climbs to **-0.554**, which is exactly the ceiling a team's own
previous conceding reaches.

That is the whole case against it, and it is stronger than the one this section
used to make. An earlier draft rejected fb-0.30 because a *significant* RPS gain
failed to replicate season by season; that gain was an artefact of the integer
labels and never existed. Another draft argued that reaching the ceiling was
evidence the rating was decaying into "last season's goals conceded", and
treated the RPS gain as evidence against that reading. With the gain gone,
nothing opposes it: the rating converges on the club's own record and buys
no prediction for doing so.

The honest cost is also unchanged and is the easier argument anyway. The
full-back pool is 0.38, so taking 0.30 would leave a full back rated **8% on
ball-winning and 30% on his club's clean sheets** — the variant in which a
published number is least about the player it names.

The weight is left at zero and the definition kept in the `FB` block, so the
measurement can be repeated rather than re-derived.

---

# How many players should a team rating be built from?

## Why

`team_strength(method="best_n")` aggregates each club's top 15 players by
rating, and the twelve `squad` features the match model reads come out of it —
as do the team ratings the site publishes. **That 15 had never been measured.**
No table, no RPS figure, no correlation anywhere in this file or in
`config/player_rating.yaml`, unlike `peak_weight`, `rho_`, the training window
and the defending weight, which all carry their evidence beside them. The
library default said `14` while production passed `15`: two unexplained numbers
that disagreed, and nothing called the 14.

## What was run

`pipelines/squad_size_sweep.py`, over the same 20,074 walk-forward fixtures.
`best_n ∈ {13, 14, 15, 16, 17}` — bracketing the shipped value rather than
extending one way, because a monotone trend either side of it is believable
where a single better neighbour is not. Players are ingested and **rated once**
for every arm: `best_n` selects from the ratings and changes nothing upstream of
them.

A second arm at 15 runs as a determinism control and came back **bit-identical**
to baseline on every metric, which is what makes the rest of the table readable.

The gate was declared before the run, because two earlier sweeps here produced
"significant" winners that did not survive: ship only on beating 15 across the
next-season correlations, without regressing RPS beyond its interval, and
improving in most of the eleven seasons that can move.

## Results

| best_n | RPS | delta | 95% CI | atk>GF | def>GA | ovr>pts |
|--------|-----|-------|--------|--------|--------|---------|
| 13 | 0.19913 | +0.00001 | [-0.00013, +0.00015] | 0.7325 | -0.5196 | 0.7546 |
| 14 | 0.19922 | +0.00010 | [-0.00003, +0.00024] | 0.7353 | -0.5231 | **0.7563** |
| **15** *(shipped)* | 0.19912 | — | — | **0.7408** | -0.5232 | 0.7562 |
| 16 | 0.19904 | -0.00007 | [-0.00021, +0.00007] | 0.7384 | -0.5280 | 0.7543 |
| 17 | 0.19905 | -0.00007 | [-0.00021, +0.00008] | 0.7331 | **-0.5316** | 0.7555 |

Attack and overall are correlations against the next season's goals for and
points, so higher is better; defence is against goals conceded, so lower is.

## Decision: nothing ships, and that is the result

No setting beats 15 on the correlations, so the gate is not met. RPS spans
**0.00018** across the whole sweep with not one interval excluding zero, and the
two arms with a nominal edge improve in 6 of the 11 seasons that can move — a
coin flip.

**But the number is better founded than "nobody measured it" suggested.** The
three correlations disagree, and 15 sits at a real optimum on one of them:

- **Attack peaks exactly at 15** and falls away in both directions, 0.7325 →
  0.7353 → **0.7408** → 0.7384 → 0.7331.
- **Defence improves monotonically with squad size**, -0.5196 at 13 through
  -0.5316 at 17, and had not levelled off at the edge of the sweep.
- **Overall is flat**, spanning 0.002, with 14 ahead of 15 by 0.0002.

So adding players keeps helping the back line while starting to hurt the attack,
and 15 is where the attacking line is sharpest.

## What this opens

The defence trend is the loose end, and it points at something the sweep was
deliberately scoped to exclude. `best_n` takes the top N **by rating with no
positional quota** — a club could in principle contribute fifteen attackers, and
`strength_defence` would be computed from whoever was left. A larger N dilutes a
thin attacking line more than a defensive one, which is exactly the shape of the
two trends above.

Testing a per-line minimum is therefore the next experiment, and it now has a
reason rather than a hunch behind it.

## Limitations

1. **RPS decides nothing here**, as in the defending sweep: 0.00018 across nine
   arms. The correlations carried the gate, and they declined to pick a winner.
2. **Five points, chosen in advance.** A finer grid around 15 would be fitting
   noise; the bracket exists to show a trend, not to find a maximum.
3. **The correlations are not independent of each other** — all three are
   computed from the same strength frame against the same club-seasons — so
   "attack peaks at 15" and "defence keeps rising" are two views of one
   selection rule, not two experiments.
4. **`mean_players` tops out below the cap**: 16.95 at `best_n=17`, because some
   clubs have fewer rated players than that. The largest arms are therefore
   slightly weaker treatments than their label suggests.
5. **Hyperparameters fixed to production**, as everywhere else in this file.

---

# Why the Champions League is not projected forward

## The question

The site simulates domestic seasons to a projected table but stops at single
Champions League ties. The reason usually offered is the knockout draw. That is
not the obstacle — a draw is an ordinary Monte Carlo, and
`src/footballml/projection.py` already simulates fixtures vectorised across
10,000 seasons.

The obstacle is coverage. A tie is predictable only when both clubs are in the
big five, because `big_five_only` (`src/footballml/ingest/european.py`) drops the
rest: a club outside England, Spain, Germany, Italy or France has no squad
rating, no form and no xG here, and `config/team_aliases.yaml` cannot even name
it, since `derive_aliases` works from domestic fixtures.

| | |
|---|---|
| European results on disk | 4,007 |
| kept (both clubs rateable) | **829** |
| dropped | **3,178** |
| distinct clubs that cannot be rated | **180** |
| dropped ties with *no* rateable side | **1,409** |

So a forward simulation would be inventing two thirds of its own inputs. Before
building one, the prerequisite was measured: **is Elo, which needs only results,
good enough to stand in where the other features are missing?**

## What was run

`pipelines/european_elo_reach.py`. `european_dropped.csv` has kept `FTHG`/`FTAG`
all along and `european_elo_rows` converts whatever frame it is handed, so the
results needed were already on disk.

Two Elo walks over the same domestic history, differing only in which European
results they learn from:

- **production** — domestic + `european_matches.csv`, which is exactly what
  `add_elo(extra=...)` is given today.
- **extended** — domestic + kept + `european_dropped.csv`.

Both are scored by a one-feature logistic head mapping the Elo gap to H/D/A,
**fitted per season on domestic matches strictly earlier than that season's
first tie**. An earlier draft fitted one head on all 29,190 domestic matches and
applied it to every season, so a 2017 tie was scored by a calibration that had
seen 2024; the ratings were always pre-match, the calibration was not. Removing
that changed the headline by 0.00014, so the look-ahead was negligible — but it
was not knowable in advance.

## Result 1: Elo reaches all 180 clubs, and ranks them recognisably

| | n | mean | sd | spread | more than 50 from 1500 |
|---|---|---|---|---|---|
| big five | 72 | 1538.7 | **96.7** | 465.5 | 61.1% |
| everyone else | 180 | 1481.9 | **31.5** | 200.8 | 17.8% |

Strongest outside the big five: Sporting CP 1583, Benfica 1554, Olympiacos 1549,
Bodø/Glimt 1543. Weakest: Malmö 1382, Bordeaux 1389, Rosenborg 1414.

**But covered is not known.** Their spread is three times tighter, because
`SEASON_REGRESSION` pulls every rating a quarter of the way back to 1500 each
summer and these clubs play six to eight European matches a year. Only 17.8% of
them are more than 50 points from the start value, against 61.1% of the big five.

## Result 2: the extended walk is better where it can be checked

On the 827 ties the full model also predicts — the same fixtures in both arms,
asserted rather than assumed:

| predictor | RPS |
|---|---|
| full model (form, xG, squad strength, Elo) | 0.21691 |
| Elo alone, **production** walk | 0.21760 |
| Elo alone, **extended** walk | **0.21621** |
| base rate | 0.23282 |

| comparison | delta | 95% CI | |
|---|---|---|---|
| extended vs production | **-0.00139** | [-0.00274, -0.00008] | significant |
| extended vs full model | -0.00071 | [-0.00391, +0.00248] | not significant |

Per season, the extension improves **8 of 10**, and one of the two exceptions is
+0.00005 — a tie. The gains are largest in the most recent seasons
(2425 -0.00271, 2526 -0.00330), which fits the mechanism: the longer the record,
the better the extra ties anchor the ratings.

Two readings, and the second is the one to hold:

1. **Elo alone is not worse than the whole model on European ties.** That is
   consistent with what this file already records — the model keeps about half
   its usual edge across a league boundary, because Elo is the only feature that
   crosses one.
2. **The result is suggestive, not established.** 827 ties is a fortieth of the
   walk-forward backtest; the interval's upper bound is -0.00008, and the season
   record gives p ≈ 0.055. This project has twice been misled by a bootstrap at
   smaller effect sizes.

## Result 3: the ties production cannot score at all

Production's walk never meets these clubs, so it has no rating for them — not a
poor prediction, no prediction. The extended walk scores all 3,178:

| | Elo (extended) | base rate |
|---|---|---|
| at least one unrateable club (3,178) | **0.20554** | 0.23188 |
| *neither* side rateable (1,417) | **0.22024** | 0.22789 |

So Elo carries real information when one side is a big-five club, and very
little when neither is — exactly where the compressed spread in Result 1
predicts it would fail.

## Decision: possible, and not being built

A league-phase projection is **feasible** — more so than the coverage numbers
suggest, because the features we lack were not buying much on these ties anyway.
It is not being built, for reasons that are about the product rather than the
modelling:

- Two thirds of its matches would rest on a predictor with **no edge worth the
  name** where neither club is rateable (0.22024 against 0.22789).
- Form would be frozen from the day of the projection through to May, a
  limitation `pipelines/project_season.py` already documents for a domestic
  season and which is far worse over nine months of a European campaign.
- The machinery it needs does not exist: `remaining_fixtures` reconstructs a
  fixture list by assuming a double round-robin, which a 36-team league phase is
  not; nothing encodes seeding, bracket topology, two-legged aggregates or
  penalties; and `fetch_schedule` discards the `round` column on the one path
  that returns unplayed fixtures.

A number on the page that says "Arsenal 23% to reach the quarter-finals" would
be indefensible in a way the measurement above is not.

## The finding worth keeping is about Elo, not the Champions League

**Production's Elo should probably learn from every European result, not only
the big-five ones.** That is a one-line change to what `add_elo` receives as
`extra=`, and it is worth -0.00139 on cross-league ties.

It is **not shipped here**, because it would change the `elo` feature for every
*domestic* match too — Elo is a single zero-sum pool — and that needs a full
walk-forward backtest, which this experiment did not run. The gain measured here
is on 827 European ties; the risk is on 20,074 domestic ones. That is the next
experiment, and it has a clear hypothesis and a clear way to be wrong.

## Limitations

1. **827 ties.** A fortieth of the backtest. The interval nearly touches zero.
2. **The season record is 8 of 10 with p ≈ 0.055**, which is the honest summary:
   consistent direction, not a settled result.
3. **Elo-only is not the model.** Result 2 says Elo matches the full model *on
   European ties*, not in general; the same comparison domestically would be
   nothing like this.
4. **The base rate is estimated in-sample** on each population it scores, which
   flatters it — deliberately, so that beating it means something.
5. **Nothing here touches the domestic backtest**, so no claim about the live
   model's accuracy follows from any of it.
