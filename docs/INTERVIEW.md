# Interview questions

The questions this project should expect, and the answers it can defend. Every
number here is reproducible from the repository; where a claim rests on a small
sample or a post-hoc choice, that is said rather than hidden.

Ordered from "tell me about the project" to the questions designed to find out
whether you understand your own work.

**Related reading:** [`METHODOLOGY.md`](METHODOLOGY.md) is the derivation,
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) is the evidence,
[`PROTOCOLS.md`](PROTOCOLS.md) holds the pre-registrations the two most recent
experiments were run under, and [`ARCHITECTURE.md`](ARCHITECTURE.md) is the code
map.

---

## 1. The project

### Q. What is it, in two minutes?

A match predictor for the top five European leagues and the Champions League,
plus the player and team rating systems that feed it. It forecasts **how many
goals each side will score** rather than classifying home/draw/away, and
everything the site shows — win probabilities, scorelines, over/under,
both-teams-to-score — is a sum over the resulting goal matrix.

Four sources: football-data.co.uk for results and odds (29,143 matches,
2010/11–2026/27), Understat for xG, FBref for player defensive and keeper stats,
and twelve EA FC editions for player ratings. Those produce 34,695
player-seasons (23,846 clear the rating gates, covering 9,784 players) and 1,170
team-seasons, which become 235 model features.

Measured on a walk-forward backtest of ~20,000 matches:

| | RPS | Accuracy |
|---|---|---|
| base rate | 0.2299 | 44.1% |
| **this model** | **0.1992** | **52.9%** |
| bookmakers | 0.1950 | 53.9% |

It covers 88% of the distance from guessing to the bookmaker benchmark. The
remaining 0.0042 is, as far as I can measure, information rather than method —
the argument for that is question 7.1.

### Q. Why predict goals instead of classifying the outcome?

Three reasons, in order of how much they mattered.

**One coherent object.** 1X2, scorelines, over/under and BTTS all come from the
same Dixon-Coles matrix, so the site physically cannot display a win probability
that contradicts its own scoreline table. A classifier would need a separate
model per surface and nothing would reconcile them.

**Draws.** A direct three-class classifier learns that predicting a draw is
almost never the argmax, and the class collapses. Derived from a goal
distribution, draws fall out at the right rate without being forced: predicted
24.6% against an actual 25.1% over the full backtest.

**Expected goals are a deliverable in their own right**, and here they are the
native prediction rather than something reverse-engineered from class
probabilities.

The cost is that I own the dependence structure between the two scorelines
explicitly instead of letting a classifier absorb it implicitly — which is what
the Dixon-Coles correction is for.

---

## 2. The metric

### Q. Why RPS and not accuracy or log loss?

Accuracy only asks whether the top pick came in, which throws away everything
interesting: a 34/33/33 forecast and a 90/5/5 forecast that both pick the winner
score identically. For a three-outcome problem where the best achievable
accuracy is around 54%, that is most of the signal discarded.

RPS is the ranked probability score — the squared error of the *cumulative*
distribution. It is the right metric here because the outcomes are **ordered**:
home, draw, away sit on a line, and being wrong by predicting a draw when the
away side won should cost less than predicting a home win. Log loss treats the
three classes as unordered and would charge both equally.

It is also the standard in the football forecasting literature, which makes the
number comparable to published work rather than only to itself.

Log loss, Brier and accuracy are all computed and reported alongside. They have
never disagreed with RPS on a shipping decision, and when they do disagree it is
informative: adding the xG family improves RPS while very slightly *reducing*
accuracy, which is the distribution getting better while the argmax does not.

### Q. What does 0.1992 actually mean? Is that good?

On its own, nothing — RPS has no absolute scale. It means something against
three reference points measured on the identical fixtures:

- **0.2299** — the base-rate forecast, the long-run 44/25/31 split applied to
  every match. This is the real "do nothing" baseline.
- **0.1950** — bookmaker closing odds, de-vigged.
- **0.1992** — this model.

So the model closes 88% of the gap between knowing nothing and the market.

Two caveats I would raise before an interviewer does. First, the bookmaker line
is a **strong benchmark, not a theoretical ceiling** — football has irreducible
noise and nobody knows where the floor is. Second, the headline figure
**understates the shipped model**: it pools twelve test seasons, and Understat's
xG only begins in 2014/15, so the earliest folds train with almost no xG. The
gap to the market was 0.0074 in 2016/17 and 0.0023 in 2025/26.

### Q. Tell me about a time a metric lied to you.

The best question to be asked about this project, because it happened and it was
mine.

`rps_per_match` one-hot-encodes outcomes by comparing against the string labels
`'H'`, `'D'`, `'A'`. A sweep harness passed `FTR.map({"H": 0, "D": 1, "A": 2})`
instead — an integer encoding. Every indicator compared a string against an
integer, every comparison was `False`, and the observed matrix was all zeros. RPS
silently degenerated into a function of the forecast alone:
`(p_H² + (p_H + p_D)²) / 2`. It **never read the result at all.**

It did not crash, did not warn, and returned a plausible-looking 0.363. I noticed
twice that 0.363 did not match the documented ~0.22 and twice talked myself out
of it as "a different normalisation."

What proved it: shuffling every outcome in the dataset left the score **bit for
bit identical**. A metric that cannot tell a correct forecast from a randomised
one is not a metric.

Three things changed. `_one_hot` now **raises** on any label that is not one of
`('H','D','A')` rather than returning silent zeros — fail loudly at the boundary,
because the caller cannot be trusted and neither could I. Every affected call
site was fixed and every published table recomputed. And there is a regression
test that shuffles the outcomes and asserts the score *moves*.

The lesson I would actually draw is not "write more tests." It is that **I had
the disconfirming evidence twice and explained it away.** The documented scale
was right there. Now a number that disagrees with a documented expectation is
treated as a bug until proven otherwise.

---

## 3. Leakage

### Q. How do you know the model has not seen the future?

This is the question the whole evaluation design exists to answer, and there are
four independent mechanisms.

**Walk-forward evaluation.** Twelve test seasons from 2015/16. Each is predicted
by a model trained only on strictly earlier seasons, retrained per fold. No
random splits anywhere — a random split on time-ordered football data trains on
May to predict August.

**Rolling features shift before they roll.** In `features/rolling.py`,
`grouped.shift(1)` runs *before* the rolling window, so match *M*'s five-match
form window excludes *M*. The shifted values are regrouped on the same keys so a
window can never span two clubs.

**Elo records the rating carried in.** `_walk` writes the pre-match ratings into
a lookup and only then applies the update, so a match never sees its own result.

**Squad strength joins the previous season.** Last season's player ratings are
knowable in August; this season's are not. This is the single most important
guard in the project, because a team rating built from the season being predicted
would be spectacular and worthless.

### Q. Those are four conventions. What enforces them?

`tests/test_truncation_invariance`. It picks a cutoff date, rebuilds the **entire
feature table** from data truncated there, and requires every overlapping row to
match the full build exactly. Any feature that can see the future produces a
different value in the truncated build and fails the suite — including features
nobody wrote a test for, and features that do not exist yet.

That is the difference between a convention and an invariant.

Worth admitting: that test used to run against an England-only file with no
`League` column — 11,400 rows of one division, where the model trains on five
leagues. The repository's most load-bearing test was validating a dataset that
shared neither the shape nor the span of the real one and could not exercise a
single cross-league path. It now runs against the full history.

### Q. Odds are in your data. Why are they not features?

Because a model that predicts the market by reading the market has learned
nothing. Odds are the strongest single predictor available and would inflate
every metric in the project while destroying its only interesting claim.

They are kept strictly as an evaluation benchmark, and the exclusion is
**enforced in `feature_columns`, not merely intended.** That changed after a
near-miss: the rule used to rely on odds never being in the frame, which held
only because nothing joined them. Then the API joined them for display — the
market price next to our number on a match card — and every odds column silently
became a model feature. The next prediction died with `KeyError: ['B365H',
'B365D', 'B365A']` because an unplayed fixture has no price yet.

That failure was loud and therefore lucky. The quiet version of the same mistake
is a model that has been reading the market for months and looks excellent for
it.

---

## 4. The model

### Q. Walk me through the architecture.

Two Poisson `HistGradientBoostingRegressor`s, one per side, predicting goal
rates. Their outputs λ_home and λ_away become a Dixon-Coles scoreline matrix
truncated at 10 goals and renormalised. Draw is the trace, home win the lower
triangle, away win the upper; over/under sums cells where `i + j > line`; BTTS is
`matrix[1:, 1:]`.

Poisson loss is the right objective for a count target: it models the conditional
rate, keeps predictions positive, and the raw model output already *is* λ.

### Q. Why gradient boosting rather than a neural network, or plain Poisson regression?

Tabular data, ~29,000 rows, 235 heterogeneous features, and **substantial
structured missingness** — promoted sides have no previous-season squad rating,
and xG does not exist before 2014/15. `HistGradientBoostingRegressor` handles
missing values natively, which is part of why it was chosen: those NaNs are
informative, not defects, and imputing them would destroy that.

A neural network on 29,000 rows of tabular data would be a worse model and a
harder one to defend.

A plain Poisson GLM was the honest alternative and would be more interpretable.
The boosting is doing real work on interactions — the learning-budget retune
alone moved RPS 0.2003 → 0.1991 — but I would not claim the architecture is the
source of the model's quality. The ablation says it is not: 93% of the gain over
a base rate arrives with three Elo columns.

### Q. How did you tune it, and how do you know you did not overfit the tuning?

The key insight is that `learning_rate × max_iter` is **one quantity** — the
total learning budget — and that is what matters. A 24-point grid scored by
four-fold walk-forward:

```
budget   3.0    4.5    6.0    9.0   12.0   15.0   18.0   27.0   45.0
RPS    .19825 .19801 .19798 .19802 .19817 .19827 .19845 .19889 .19961
```

A flat bottom at 4.5–9 rising monotonically after. The previous setting
(`0.05 × 300`, budget 15) sat on the rising side — the model was mildly
**over**-trained, which is the opposite of what you would expect for 235 features
on 29,143 rows.

Shipped `0.02 × 300` is budget 6, the minimum of that curve. `0.03 × 150` scored
a hair better but sits at the **edge** of the iteration axis, which is exactly
where a selection artefact hides. **The interior point was taken over the better
score deliberately.**

Confirmed on the full backtest against the previous settings: RPS 0.2003 →
0.1991, −0.00124 [−0.00158, −0.00090]. Log loss, Brier and accuracy all moved
with it — which three feature experiments that failed the same week did not.

`early_stopping=False` is deliberate: scikit-learn holds out a *random*
validation split, which on time-ordered data means validating against the future.

### Q. What is Dixon-Coles and why do you need it?

Independent Poisson is very nearly right and wrong in one documented way: it
under-predicts low-scoring draws. Dixon-Coles multiplies the four cells where
both sides score at most once:

```
tau(0,0) = 1 - lambda*mu*rho     tau(0,1) = 1 + lambda*rho
tau(1,0) = 1 + mu*rho            tau(1,1) = 1 - rho
```

Fitted ρ = **−0.0566** by maximum likelihood **on training data only** — it is a
model parameter like any other and fitting it on test data leaks. Negative ρ
moves mass into 0-0 and 1-1 and out of 1-0 and 0-1.

### Q. One ρ for five leagues with different scoring profiles. Isn't that wrong?

It looked wrong, I tested it, and it survived — which is a better answer than
either the assumption or the objection.

In sample the heterogeneity is real and large. Profiling the Dixon-Coles
likelihood per league puts the Bundesliga at **−0.1297 [−0.1784, −0.0787]**, an
interval that **excludes the shipped −0.0566**, against Spain's −0.0326. More
than a factor of four apart. ρ also trends with scoring level, −0.038 in the
lowest total-λ quartile to −0.070 in the highest.

Out of sample it buys nothing. Walk-forward over 18,212 fixtures and 11 folds,
every ρ fitted on strictly earlier folds:

| arm | logLik/match Δ | 95% CI | leagues better |
|---|---|---|---|
| per-league | +0.000137 | [−0.000210, +0.000489] | **1 of 5** |
| shrunk | +0.000084 | [−0.000057, +0.000225] | 3 of 5 |
| per-lambda | −0.000163 | [−0.000484, +0.000154] | 1 of 5 |

**The shape of the failure is the finding.** The gain is +0.001263 in the
Bundesliga and a small loss in every other league — the one league whose
heterogeneity motivated the experiment is the only one that gains, and it does
not clear its own interval. That is what an in-sample artefact looks like when
you finally take it out of sample.

Two design points I'd defend. The margin on exact-score hit rate was set at
**0.25 percentage points read off the CI upper bound**, not "must not fall" —
and it earned that on the first arm it touched, where `per-league` declines
0.0439pp with an interval of [−0.0933, +0.1757]. A zero-tolerance rule would
have rejected an arm over noise. And **RPS was not assumed insensitive to ρ**:
ρ moves mass between cells sitting on both sides of the draw/win boundary, so it
was measured, and it does move — by +0.000012, two orders below the floor.

No-ship, all three arms. Production keeps one global ρ, and now it keeps it for
a measured reason rather than an unexamined one.

### Q. Your site says 1-1 is the most likely score in most matches, but also that the home team will probably win. Is that a bug?

No, and it is a good test of whether someone understands the output.

An outcome sums a *triangle* of the matrix; an exact score is a single *cell*.
Inside a 0-5 grid the draw has 6 cells against 15 for each win. So 1-1 is the
single likeliest cell in 67% of matches while the home side is usually the
likeliest *winner*.

Even a perfectly even match at 2.75 expected total goals gives the draw 0.280
against 0.360 each way. The draw only leads as an outcome below about 2.0 total
goals, and these leagues run 2.7–2.8. In ~20,000 backtested matches the highest
draw probability was 37.5% and the draw was the top pick twice; the market made
it favourite 7 times.

---

## 5. Ratings

### Q. How is a player rating calculated?

Seven steps; every debatable number lives in `config/player_rating.yaml`.

1. **Per-90** — each counting stat becomes `value / (minutes / 90)`.
2. **Percentile within (role, season)** — a centre-back is only ever ranked
   against centre-backs in the same season. This buys cross-position
   comparability and immunity to season inflation at once. Gates: 450 minutes,
   and 40 peers before a pool is rated at all.
3. **Sub-ratings** — weighted means of percentiles, per role. Seven roles: GK,
   CB, FB, DM, MID, AMW, FWD.
4. **Peak blend** (`peak_weight = 0.15`) — pulls the composite toward the
   player's best sub-rating, because a weighted mean punishes specialists.
5. **Empirical-Bayes shrinkage** — small samples pulled toward the role mean.
6. **Scale** to 0–99 against fixed anchors.
7. **Blend with EA** at `fifa_weight = 0.5`, scaled by how much of the role the
   season's data actually supported.

### Q. And team strength?

```
1. role -> line      GK -> goalkeeper, CB/FB -> defence,
                     DM/MID -> midfield, AMW/FWD -> attack
2. pick the squad    top 15 by RATING (best_n, DEFAULT_BEST_N = 15)
3. per line          minutes-weighted mean of those players' ratings
4. overall           lines weighted 1 / 4 / 3 / 3, normalised by weights present
5. gate              MIN_RATED_PLAYERS = 11
```

**Selection is by rating; minutes weight the aggregation.** It is explicitly not
a predicted lineup, and no external lineup source is used.

### Q. Why 15? Why not 11, or the whole squad?

Because it was measured. `best_n` was swept 13–17 against the *next* season's
outcomes:

| best_n | RPS | atk > GF | def > GA | ovr > pts |
|--------|-----|----------|----------|-----------|
| 13 | 0.19913 | 0.7325 | −0.5196 | 0.7546 |
| 14 | 0.19922 | 0.7353 | −0.5231 | 0.7563 |
| **15** | 0.19912 | **0.7408** | −0.5232 | 0.7562 |
| 16 | 0.19904 | 0.7384 | −0.5280 | 0.7543 |
| 17 | 0.19905 | 0.7331 | **−0.5316** | 0.7555 |

Attack peaks exactly at 15, defence improves monotonically with squad size, and
overall is flat. No RPS difference is significant — the whole column spans
0.00018. So 15 stays, and the honest statement is "nothing beat it," not "15 is
optimal."

I should admit this was swept *after* 15 was shipped. It had been an unexamined
default, and the sweep found the codebase still carried a dead `best_n = 14` that
nothing used.

### Q. Defence improved all the way to 17 and attack peaked at 15. Did you chase that?

Yes — and then a replication run pulled the rug out from under the premise,
which is the more interesting outcome.

**First, the honest caveat on the question itself.** Re-running the sweep end to
end on identical inputs reproduced the *decision* and not the *ordering*: attack
peaked at 16 rather than 15, and defence was best at 16 with 17 worse. The
largest single-cell move between runs was 0.0030, larger than most gaps between
adjacent sizes. So the per-size pattern is inside the run-to-run noise and the
table supports "nothing beats 15", not "15 is a local optimum".

I only know this because I re-ran something I had no reason to doubt — and the
diagnostic that followed is the next question.

**The chase itself**, run while the trends still looked real:

The natural reading was **dilution**: a larger N adds marginal attackers to a
line that has few real ones, while the back line keeps gaining useful players. If
that were right, a per-line floor should buy the defensive gain without the
attacking cost.

Three quotas were declared in the source before the run — 1-4-3-3, 1-3-3-2 and
2-5-5-3 — as floors, never caps, so a club whose six best players are defenders
still fields them. Every one moved defence in the predicted direction, so the
mechanism is real. Every one paid for it at one-to-one or worse:

| arm | defence gain | attack cost | ratio |
|-----|--------------|-------------|-------|
| balanced | +0.0020 | −0.0045 | 1 : 2.3 |
| formation | +0.0060 | −0.0065 | 1 : 1.1 |
| full | +0.0094 | **−0.0142** | 1 : 1.5 |

So the opposing trends are not a fixable artefact — they are a genuine trade-off
in how many players stand for each line, and 15 with no quota sits at a
reasonable compromise. **No-ship, all three.**

The harness asserts at runtime that its baseline arm reproduces production's
`best_n=15` to within 1e-9. If it had not, no arm would have been readable.

### Q. You said "I could not attribute the drift". Did you find it?

Yes, and the first explanation I offered was wrong, which is the useful part.

I had written that the drift was "between-process variation", with float
reduction order under the booster's threading as the likely candidate. That was
a hypothesis stated in the register of a cause, and it was wrong.

**The bisection.** I hashed every stage of the pipeline — ingest, ratings, team
strength, match history, feature build, a single isolated model fit, and the
full 20,074-fixture walk-forward — and ran it in two separate processes.

**Every stage was bit-identical**, including the two gradient-boosted fits per
fold, the fitted ρ to fifteen decimals, and RPS to nine (`0.199159864`). So
seeds, thread counts and numerical reductions were all exonerated, and the
floating-point story had to be withdrawn.

**The divergence is at stage 1.** `fetch_player_seasons` caches *finished*
seasons only — there is a cache file for every season from 2014/15 to 2025/26
and none for 2026/27. The season in progress is fetched live from Understat on
every run. Twenty-five minutes apart, the same script saw 34,769 player-seasons
and then 34,773; the current season had grown by four. The harness was
reproducible and its inputs were not.

### Q. Four player-seasons in a season you don't even score. How did that move anything?

That is exactly the question I had, and the answer is the part worth keeping.

The in-progress season *should* be inert. It supplies 3 of 1,173 team-seasons,
and the outcome builder excludes it outright — its clubs have a median of six
matches played against a twenty-match floor, so none of them gets an outcome
row. I verified the obvious channel directly: dropping its team-strength rows
from the correlation changes the result by **0.00000**.

It is not inert. Rating the frame with the current season against rating only
the finished ones moves the finished-season correlations:

| | atk>GF | def>GA | ovr>pts |
|---|---|---|---|
| with the in-progress season | 0.73947 | −0.52705 | 0.75556 |
| without it | 0.73844 | −0.52394 | 0.75501 |

**0.0031 is the same order as the gaps between adjacent squad sizes**, which is
precisely why the ordering moved while the decision didn't.

**And I could not establish how.** Row by row, `rating`, `performance_rating`,
`fifa_overall` and `measured_share` are *identical* for all 32,574
finished-season players, and EA match coverage agrees to four decimals. Yet the
selected squads differ. Something other than those four columns carries it, and
I stopped at "measured and reproducible, mechanism unknown" rather than guessing
a second time.

### Q. What did you change?

`pipelines/vintage.py` — four functions, research-harness only, with a test
asserting nothing in `src/footballml` imports it. Production is untouched.

`pin_finished_seasons` drops the season in progress, so a run depends only on
files that are not moving. It is idempotent and a no-op out of season, so it is
not something to remember to switch off. `fingerprint` and `describe_inputs`
print each input's row count and content hash beside the results.

Verified: three further processes, pinned, produce an identical input hash and
identical correlations to five decimals, where unpinned runs twenty-five minutes
apart differed by 0.0013.

The fingerprint is the part I would argue for hardest. The original discrepancy
took a stage-by-stage bisection to locate. With the input hash in the run log it
would have been a one-line diff — and a published table would carry the vintage
it was computed from, which is the thing that made two documents disagree in the
first place.

### Q. What does this change about how you read your own results?

Three things, and they are now written into the experiment docs.

**Within a vintage the variance is exactly zero** — same process, separate
processes, repeated runs, all bit-identical. So any disagreement between two
runs is a data question, never a numerical one. That is a useful thing to know
about a pipeline, and I only know it because I measured it.

**The correlations had no measurement floor.** RPS has one, 0.0005, established
by sweeps that moved nothing. The next-season correlations never got the same
treatment, and this diagnostic is the first measurement of it: **0.003**.
Several of the differences I had been reading off those tables sit under that.

**An unreplicated ordering is noise.** The squad-size *decision* reproduced on
every vintage tested; only the ordering moved. I had let the ordering carry a
narrative — "attack peaks exactly at 15" — and build the motivation for a
follow-up experiment. The decision was always the result.

### Q. Van Dijk rates in the 3rd–11th percentile for tackles. Is your rating broken?

It looks like a bug and it is the data. He does not need to tackle — positioning
prevents the situation that requires one. Defending is measured by *volume*, not
by quality or by chances prevented.

Two fixes were measured and both made the rating **worse** against next-season
goals conceded. Dropping tackles raises him fifteen points and costs the only
thing the number is for. Aerial duels would settle it; FBref no longer publishes
them — verified by checking the HTML, where `data-stat="aerials_won"` appears
zero times, not blank but absent.

What *did* ship from that investigation: possession adjustment. Raw per-90 counts
reward a defender for playing on a side that has to defend a lot — Brentford sit
deep, so Nathan Collins scored 36 for defending while Damsgaard, a winger,
scored 95. Possession-adjusted only correlates −0.489 with next season's goals
conceded, against −0.483 for raw plus adjusted and −0.478 for EA's rating alone.

### Q. Why blend with EA's ratings at all? Isn't that borrowing someone else's answer?

Partly, and the weight is a deliberate 50/50 rather than a hidden one.

The defence is that the two measure different things. Ours is a **rank against
positional peers in one season** — a performance measurement. EA's is a scouting
assessment that includes things no season-aggregate can see: ceiling, physical
attributes, reputation, and the judgement of people who watch the player.
Blending is how the rating stays sane for a player with 500 good minutes.

`measured_share` records how much of each role's inputs the season's data
actually supported, and the blend leans on EA in proportion when ours is thin.
That is the part I would defend hardest: the fallback is explicit and published
per player, rather than the rating quietly becoming EA's with extra steps.

### Q. Leicester 2015/16 are 15th in your team strength. Doesn't that disprove the rating?

The opposite. EA's own FC16 ratings put that squad 18th of 20, below Watford and
Aston Villa, and the bookmakers priced the title at 5000-1. A squad-quality
rating that liked Leicester in 2015/16 would be the broken one — it would be
fitting the outcome, not the squad.

This is the general point about rating validity: the test is whether it predicts
*next* season's goals and points, not whether it retro-explains famous seasons.

---

## 6. Evaluation and honesty

### Q. Which features actually matter?

Two measurements, because one is not enough.

The **ablation ladder** adds families in order:

| step | ΔRPS | 95% CI | share |
|---|---|---|---|
| `base` → `+elo` | **−0.02867** | [−0.03041, −0.02691] | 93.4% |
| `+elo` → `+form` | **−0.00175** | [−0.00239, −0.00109] | 5.7% |
| `+form` → `+xg` | −0.00014 | [−0.00049, +0.00023] | 0.4% |
| `+xg` → `+squad` | −0.00015 | [−0.00032, +0.00001] | 0.5% |

The tempting headline is "this is essentially a well-tuned Elo." **That reading
is wrong**, and a Shapley attribution over family subsets is what showed it:
every family recovers ~90% of the gain on its own, so whichever went first was
always going to post 93%. Averaged over all orderings the four families are
roughly equal and `form` is marginally the largest.

What survives is weaker and still useful: **by the time the model has Elo, almost
nothing else moves it** — a statement about redundancy, not about Elo's primacy.

### Q. Your xG features are 40% of the model and contribute nothing. Why are they still there?

Because the pooled null is **two significant effects of opposite sign
cancelling**, and that is the single most useful thing the ablation found.

Understat's xG begins in 2014/15. Predicting 2015/16 means training on
2010/11–2014/15, where four of five seasons carry no xG at all, so the features
are overwhelmingly missing in training and the model fits noise. Splitting on how
many xG seasons were available *in training*:

| | n | adding xG | removing xG |
|---|---|---|---|
| 1516–1920 (≤5 xG seasons) | 9,029 | **+0.00087** worse | −0.00086 better |
| 2021–2627 (≥6) | 10,984 | **−0.00097** better | +0.00052 worse |
| pooled | 20,013 | −0.00014 — | −0.00010 — |

Both halves clear zero individually, in opposite directions, and the two
independent analyses agree. The live model trains on twelve seasons of xG and
sits firmly in the regime where xG helps.

**The caveat I would volunteer:** that split is **post-hoc**. It was chosen after
seeing the per-season table, which makes it a hypothesis rather than a confirmed
result. Three things argue it is not an artefact — the split rule comes from a
fact about the data independent of any outcome, both halves clear zero, and the
two comparisons agree in sign and magnitude — but it should be confirmed on data
the experiment has not touched.

### Q. Tell me about an experiment that failed.

Several, and they are all written up — negative results are kept in
`EXPERIMENTS.md` at the same length as positive ones.

The most instructive is the **extended Elo A/B**, because of how it failed.

Elo is the only feature crossing a league boundary. Production feeds it 829
European ties where both clubs are in the big five; another 3,178 sit unused
because one side has no squad rating, no form and no xG. Elo needs none of that —
it needs results. Measured separately, feeding them in improved an Elo-only
predictor on the ties where the full model can also be scored: 0.21760 → 0.21621,
−0.00139 [−0.00274, −0.00008], better in 8 of 10 seasons.

But Elo is **one zero-sum pool**, so changing what it learns from changes the
`elo` feature on all ~20,000 domestic fixtures, not just the European ones. So it
was framed as a **non-inferiority test** with the bar fixed in source before the
run: ship if the upper CI bound on the domestic delta is < +0.0005 *and* the arm
is neutral-or-better in ≥ 6 of 11 movable seasons.

Both arms came back nominally worse domestically. `extended` failed outright at
5 of 11 seasons. `extended-no-uecl` **met the bar and still should not ship** —
because it is not the arm the European benefit was measured on. Each arm fails a
different necessary condition, and neither has both halves.

That is a gap in how I wrote the criteria, not a licence to pick the survivor
after the fact. The honest reading: the experiment answered the question it
asked, and the question was one condition short.

### Q. You just described a mistake in your own experiment design. Any others?

One worth telling, because I caught it by reading rather than by testing.

The verdict counter in that same experiment credited the earliest test fold as
"neutral-or-better." That fold trains with the strength features absent, so **no
arm can move it** — every arm scores it identically. Counting it as a pass handed
every arm a free point against a denominator that had already excluded it,
turning 5-of-11 into 6-of-11 and a failing arm into a passing one.

I caught it because the verdict line read kinder than the table above it. It is
now fixed and pinned by a test named
`test_a_group_no_arm_can_move_is_not_counted_as_a_pass`.

The general lesson: **a summary that flatters the table it summarises is a bug
until proven otherwise.** That is the same failure mode as the RPS bug — a number
that disagreed with its own context and got explained away.

### Q. Any others?

One more, and it is the clearest instance of the same tell.

The conditional-ρ experiment has a `shrunk` arm — per-league ρ pulled toward the
pooled value. Its first run reported a delta against production of **exactly
0.000000**, on all 18,212 fixtures and all five leagues. Not approximately.
Identically.

My shrinkage factor was `1 − (k − 2) / (k · spread)` with **no sampling variance
in it**, which implicitly sets σ² = 1. ρ deviations are ~0.07, so `spread` is
~0.001, the term is ~556, and the factor hit its floor of zero on every fold.
**The arm was production in disguise, benchmarked against itself.**

Caught because a delta of exactly zero across three metrics and five leagues is
not a result, it is an identity — the same signature as the RPS bug, where a
score that did not change under shuffled outcomes was not reading the outcome.
**Three of this project's four measurement bugs announced themselves the same
way: a number too clean for what it claimed to be measuring.** A score unchanged
by shuffled outcomes, a verdict kinder than the table above it, and an arm
reporting exactly 0.000000 against itself. The fourth had no such tell and is
the next question.

Replaced with empirical-Bayes shrinkage taking each league's variance from the
curvature of its own likelihood. The variances are now a **required argument**:
omitting them applies no shrinkage rather than shrinking by an assumed one.
The whole experiment was re-executed end to end, and the two arms that do not
route through the broken function are byte-identical across both runs — which is
how I know the fault was confined to one arm rather than the harness.

`EXPERIMENTS.md` keeps the invalid run and labels it, rather than quietly
publishing the corrected numbers.

### Q. And the third?

Found by reading the documents against each other, which is the only reason it
was found at all.

The rating-travel experiment excludes 2025/26 as a **predictor** season, because
its own following EA edition is still being populated. The lookup that asks "did
the matcher already place this player in edition `t+1`?" was built from that same
filtered frame — so 2025/26 was missing from it, and the **2024/25 → 2025/26
pairs lost their entire first-tier recovery**. That fold recovered 36.2% where
every other season recovered 94–97%.

A season can be ineligible to be predicted *from* while remaining perfectly good
to be predicted *to*. The damage was not only volume: what survived was whatever
a raw name-and-age match happened to find, which favours distinctively-spelled
names — a selected third of a season, entering a model whose season fixed effects
absorb a level shift but not a composition shift.

**The pre-registration caught it.** `docs/PROTOCOLS.md` recorded 92.7% recovery
from the feasibility audit written before the experiment; the write-up said
88.9%. Two documents that should have agreed did not, and the protocol was the
one that was right. That is an argument for pre-registering the numbers you
expect, not just the criteria.

No conclusion changed — both deltas moved slightly *toward* zero and the
missingness grid got more robust — which is luck, not vindication.

### Q. How do you decide whether a result is real?

Paired bootstrap, 10,000 resamples, fixed seed, on identical fixtures — and the
arms are asserted in code to have scored the same fixture set, because pairing
two different sets makes every interval nonsense.

More importantly, there is a **measurement floor**. The defending sweep spanned
0.00012 across nine arms and the squad-size sweep 0.00018, neither significant on
~20,000 fixtures. So **0.0005 is treated as the resolution of this harness**, and
a claimed improvement smaller than that is not a result regardless of its point
estimate. That floor is asserted as a constant in the experiment sources so it
cannot be quietly moved after seeing an answer.

A second discipline: **acceptance criteria are declared in the source before the
run**, and tests assert they are the declared constants rather than numbers typed
in after seeing the table.

One more subtlety worth raising unprompted: the bootstrap resamples **fixtures**,
but the model is retrained **per season**, so every match within a fold shares
one fitted model and the errors are correlated within a season. A fixture-level
bootstrap therefore understates the true uncertainty, and intervals over ~20,000
fixtures are narrower than the experiment really earns. This is why the
per-season win/loss count is reported alongside every interval, and part of why
the floor sits at 0.0005 rather than wherever the intervals alone would put it.

---

## 7. The hard questions

### Q. You are 0.0042 behind the bookmakers. Is that a modelling failure?

I have tried hard to show that it is, and the evidence says it is information.

**Calibration is not the problem.** Marginally the model is near-perfect: home
0.443 predicted against 0.441 actual (market 0.441), draw 0.246 against 0.251.
Sharpening the output was swept from 0.6 to 1.2, the optimum is ~0.95, and RPS
does not improve.

**The distributional assumptions hold.** Pearson dispersion of actual goals
against predicted λ is 0.977 — if anything slightly *under*-dispersed, so the
Poisson choice is sound and a negative binomial would be the wrong direction.

**The gap is uniform.** Decomposed against de-vigged B365 prices on 20,007
fixtures, it is +0.0034 to +0.0048 across all five leagues, +0.0028 to +0.0047
across matchweek buckets, and +0.0022 to +0.0051 across market-confidence
buckets. There is no pocket to attack — which also rules out the obvious guesses
like a promoted-team cold start.

**Feature engineering on results has provably hit its ceiling.** A goals-based
attack/defence Elo was built, properly tuned, and enthusiastically taken up by
the booster — ranks 2, 3, 4 and 8 of 241 features, 9.4% of attributed movement,
with `elo_diff` falling from 17.9% to 16.0% to make room — and predicted exactly
as well as before: +0.00003 [−0.00036, +0.00041]. **It reorganised a tenth of
the model's reasoning and gained nothing**, because it is built from the same
match results as everything else.

What remains is the thing the model cannot see: **lineups, injuries, suspensions
and team news**, all of which the market prices. The signature fits — where the
market says 80–90%, the truth is 85.7%, the market says 83.9% and this model says
81.4%. Well calibrated, under-discriminating. That shortfall was 5.0 points
before the learning-budget retune and is now 2.4; the shape is unchanged.

### Q. So go and get lineup data.

That is the top-ranked open direction, and the design starts with a **ceiling
test rather than an acquisition project**.

`soccerdata`'s Understat reader — already a dependency, already the xG source —
exposes `read_player_match_stats`, which returns minutes, position and cards
**per player per match**. That makes two things derivable with no new source:

- a **retrospective availability feature** (what share of a club's season rating
  actually took the field), which is deliberately leaky and measures the
  *ceiling* of the whole thesis;
- **suspensions and absence streaks**, which are knowable strictly from past
  matches and are therefore leak-free and usable in production.

If a perfect-information availability feature does not move RPS by more than the
0.0005 floor, the entire missing-information thesis is falsified and no amount of
injury-news scraping would have helped. That is the kill criterion, and it is
cheaper than the project it gates.

### Q. Why don't you predict the Champions League forward, when you already project domestic tables?

Because I measured whether I could and the answer was "badly."

The obstacle is **not** the knockout draw — a draw is an ordinary Monte Carlo.
It is that only **829 of 4,007** European results on disk have two rateable
clubs. **180 clubs** appear that cannot be rated at all: no squad rating, no
form, no xG, not even a canonical name. **1,409** ties have no rateable side.

Elo *can* rate all 180 from results alone and ranks them recognisably. But those
ratings are **three times more compressed** than the big five's — standard
deviation 31.5 against 96.7 — so "covered" is not "known." Where neither club is
rateable, an Elo-only predictor barely beats a base rate: 0.22024 against
0.22789.

A league-phase projection would rest on that for two-thirds of its matches, with
form frozen from the projection date to May. **The measurement is the
deliverable; the table is not.** Shipping the tab would have been easier than
establishing why it should not exist.

### Q. What is the single weakest part of the project?

Defender rating validity — and I tested the obvious explanation for it and was
wrong, which is the more useful half of the answer.

Our ratings correlate with EA's *next-edition* move at 0.15 for forwards and
0.23 for goalkeepers, but **0.04–0.06 for centre-backs, full-backs and
midfielders** — in every season. A centre-back's rating also correlates **+0.67**
with his own team's defensive record, and `config/player_rating.yaml` gives
**CB `team_defence: 0.30`** against **FB `0.0`** — so by construction 30% of a
centre-back's composite *is* his club's expected goals against, goals against
and clean-sheet rate.

The obvious worry: the number is a team number wearing a player's name.

### Q. So how did you test that?

A transfer separates the player from the club. If the rating is about him it
should predict his next EA assessment equally well whether or not he moves; if
it is about the side he just left, it should attenuate.

**The sample design is the part worth asking about.** The obvious version —
players with a *rated* season at `t+1` — conditions on the player clearing the
450-minute gate at his new club. That is post-transfer information, and it
selects for successful transfers, which is the thing under test. EA's raw
exports are the whole worldwide database rather than the big five, so the
outcome is recoverable without that condition: 92.7% recovered, 1,347 CB movers
against 2,379 stayers, and the minutes gate out of the design entirely.

Two further corrections. The predictor is **`performance_rating`, not
`rating`** — `rating` is half `fifa_overall`, so scoring it against EA's next
edition would be substantially EA predicting itself. And the outcome is a
**within-edition z-score**, because EA's scale drifts 2.5 points across the
window.

### Q. And the answer?

**No contamination, and the verdict is inconclusive.** Both of those, and they
are not the same statement.

| role | stayer | mover | δ | ratio |
|---|---|---|---|---|
| CB | **+0.1404** [+0.1277, +0.1531] | +0.1385 | −0.0019 [−0.0197, +0.0160] | 0.986 |
| FB | **+0.0940** [+0.0810, +0.1067] | +0.0848 | −0.0092 [−0.0301, +0.0116] | 0.903 |

The stayer slopes clearly exclude zero, so the power gate passes and the ratio
is interpretable. Neither δ interval excludes zero, so nothing meets the
contamination bar.

It is reported **inconclusive** rather than as a clean null because two triggers
I wrote down before running both fired. The **rank specification disagrees in
sign** with the z-score one for both roles — two scales of the same outcome
pointing opposite ways on an effect they both call indistinguishable from zero.
And CB's equivalence verdict is **threshold-dependent**: equivalent at [0.85,
1.18] and [0.80, 1.25], not at [0.90, 1.11].

What I can say positively is narrower and still useful: no contamination was
detected on any specification, and CB's attenuation is bounded within about 12%
at 90% confidence.

### Q. You keep saying "inconclusive". Isn't that just a failed experiment?

No, and the distinction is the one I'd most want to be asked about.

A failed experiment produces a number nobody can trust. This produced a bounded
answer plus a specific, named reason it cannot be sharpened: two defensible
scalings of the same outcome disagree on a sign. That is a fact about the
instrument, and knowing it is worth more than a verdict that depends on which
scaling I happened to pick first.

The alternative was available and is what I'd have done a year ago — report the
z-score result, call it practical equivalence at the bound that happens to pass,
and not mention the rank version. The triggers exist precisely because that
temptation is predictable, so the rule has to be written before the number.

### Q. Did anything come out of it?

The thing I did not go looking for.

| | projection R² | stayer slope, raw | residualised |
|---|---|---|---|
| CB | **0.526** | +0.1404 | **+0.0347** |
| FB | 0.095 | +0.0940 | +0.0568 |

Projecting the three club aggregates out of the rating shows they explain
**52.6% of a centre-back's performance rating** against 9.5% of a full-back's —
exactly the asymmetry the 0.30 and 0.00 weights imply, and larger than I
expected. Removing them costs CB **75%** of its power to predict EA's next move.

**And it travels anyway.** The residualised δ was supposed to shrink toward zero
for CB if contamination were the mechanism; instead CB's δ was already ~0 and
the residual version is if anything positive.

So a centre-back's rating is half his club's defensive record and predicts his
next assessment just as well after he leaves. The economical reading is that
club defensive quality is a **legitimate predictor** of how a defender will next
be judged, not a contaminant that evaporates on transfer. That is
mechanism-consistent evidence against my own hypothesis — the residual is
orthogonal to three club numbers in one season, not to "the player", so it is
not proof.

The EA-free check agrees and is the strongest single number in the experiment: a
centre-back one standard deviation better predicts **0.08 fewer goals conceded
per match** at his new club, [−0.1218, −0.0464] over 259 single-arrival
club-seasons — about three goals a season, with no EA data in it at all. It
conditions on post-transfer minutes and attributes a team outcome to one
arrival, so it is corroboration, not an instrument.

### Q. How did you handle the players who vanish from the data?

This is the sharpest question about the design and the honest answer is that it
is a limitation, not a solved problem.

7.3% of cases are unrecoverable, and they are not missing at random: older
(standardised difference +0.57), played less (−0.58), rated lower (−0.37).
Critically, an unrecovered player has **no `t+1` record at all**, so **both the
outcome and the treatment are missing** — I do not know whether he moved.

That matters for what inverse-probability weighting can do. IPW identifies the
estimand only under recovery independent of treatment *and* outcome given the
observed covariates, and mover status is unobservable exactly when recovery
fails. **So IPW corrects selection on age, minutes and rating, and cannot touch
selection on mover status.** I say that rather than presenting IPW as a fix,
because the primary effect is **not point-identified without an assumption the
data cannot check**.

What replaces the fix is a bound. A pattern-mixture grid over the assumed mover
share among the vanished, crossed with three imputations of their outcome, holds
CB's ratio above the 0.85 bound in **every cell** — including the extreme anchor
where every vanished player is assumed to have moved, against an observed share
of 0.34 among the recovered. There is no tipping point inside the grid.

I also had to withdraw a claim. An earlier draft of the protocol asserted the
bias ran toward zero, reasoning that a player who drops out of football leaves
EA's database while a declining stayer does not. That is a conjecture about an
unobservable quantity and the reverse is equally arguable, so it came out and
the grid went in.

### Q. Why is there no "Barcelona without Yamal" feature? It's the obvious thing to build.

It is the most-requested feature and the one to keep refusing, because the model
cannot support the claim it would make.

The squad family is **redundant** with the other three: its leave-one-out
interval straddles zero, and the Shapley attribution puts it at 19.8% precisely
*because* the information is also carried by Elo, form and xG. So removing one
player from the strength input and re-scoring would move the win probability by
some amount — and that amount would be an artefact of how the booster happens to
weight a redundant feature, not a measurement of the player's worth.

Publishing "61% → 56%" would be inventing a number with a confidence interval
nobody could compute. The honest version of that feature needs per-match lineup
data and a model that has seen squads change mid-season, which is the direction
in the question above.

### Q. What would you do differently?

**Measure the defaults earlier.** `best_n = 15`, the squad size the entire team
rating rests on, went unexamined until a sweep found a dead `best_n = 14` sitting
beside it. The number turned out to be fine, which is luck rather than process.

**Make the metrics fail loudly from the start.** The RPS bug cost two published
tables and a week of conclusions, and a two-line guard at the boundary would have
caught it the first time.

**Write the acceptance criteria before the experiment, every time.** This is now
the rule and it is enforced by tests, but it became the rule after an experiment
whose surviving arm was the one the criteria happened not to cover.

**Check what the sample conditions on before building the model on it.** The
first design of the rating-travel experiment required a rated season after the
transfer, which silently conditioned on the player succeeding at his new club —
the exact thing under test. It came out in review, not from the code.

**Write down the numbers you expect, not just the criteria.** The pre-registered
feasibility audit said 92.7% recovery; the executed run said 88.9%. That
disagreement was the only thing that surfaced a bug which had already destroyed a
fold and would otherwise have shipped in a published table.

### Q. Several of your experiments end in "no-ship" or "inconclusive". Does anything ever work?

The model works — 0.1992 against a 0.2299 baseline, 88% of the way to the
bookmakers. What mostly does not work is the *next* idea, and that is the
expected shape once a system is near the ceiling of its data.

The honest tally: of the six experiments run most recently, one shipped (the
learning-budget retune, −0.00124 with every secondary metric moving with it) and
five did not — squad size, positional quotas, extended Elo, conditional ρ, and
rating travel. Each produced a number that now constrains what gets proposed
next, which is why they are written up at the same length as the one that
shipped.

The pattern is informative in itself. **The recurring failure mode is a signal
that is real on the measure that suggested it and absent on a held-out one:**

- the Bundesliga's ρ excludes the shipped value in sample and gains nothing out
  of sample;
- `peak_weight = 0.30` improved all three rating correlations and **failed** the
  walk-forward RPS gate, 0.2003 → 0.2008, which is why 0.15 ships;
- the transfer gap that motivated the rating-travel experiment — persistence
  0.804 for stayers against 0.609 for movers — **disappeared** once the
  predictor was the unblended rating, the outcome was EA's own move, and the
  sample stopped conditioning on post-transfer minutes.

That last one is mine twice over: I produced the suggestive number and then
designed the test that killed it. Having a framework that does that reliably is
worth more to me than any individual result in the file.

### Q. What are you proudest of?

That the negative results are written up at the same length as the positive ones,
and that several of them killed ideas I was attached to.

Concretely: the project can say *why* it does not project the Champions League,
*why* team strength has no positional quota, *why* the squad is 15 players,
*why* ρ is a single global scalar, *why* defender ratings are left as they are,
and *why* there is no player-level what-if — and each is a measurement with an
interval, not an opinion. For a portfolio piece, being able to defend the things
that are absent matters more than the length of the feature list.

The two most recent experiments are the ones I would actually show. One tested a
hypothesis I was confident in and **found the opposite**: a centre-back's rating
is half his club's defensive record and travels anyway. The other reported
**inconclusive** on triggers I had written down in advance, when the convenient
reading was available and would not have been questioned.

---

## 8. Engineering

### Q. How does it run in production?

FastAPI serving a React/Vite front end. A weekly job refetches results, rebuilds
ratings and features, retrains, and writes a versioned model artifact; the API
hot-reloads via `POST /admin/reload`.

The reload path loads into locals first and only then swaps, because a model
artifact trained without Elo once could not produce `home_elo` and every
`/matches` call died. Now an incompatible artifact costs a **refused reload**
instead of a broken site.

Predictions made before kickoff are stored and scored afterwards, so the accuracy
page is a real track record rather than a backtest replay. That currently stands
at **83 settled matches, RPS 0.1765** — better than the backtest's 0.1992, and
far too small a sample to claim as an improvement.

### Q. How is it tested?

748 tests, ruff clean. The ones that matter are not unit tests of functions:

- `test_truncation_invariance` rebuilds the whole feature table from truncated
  data and demands identical rows — the leakage invariant.
- Experiment harnesses are asserted to be **unable to write anything** — no
  `to_csv`, no `shutil`, no file opened for writing. That exists because a
  `finally` block does not survive a killed process, and one experiment once left
  `team_strength.csv` holding its own numbers with the API serving them.
- Acceptance thresholds are asserted to be the declared constants, so a bar
  cannot be moved after seeing a result.
- Team names in the ratings table must all resolve to names the match data
  knows — data-driven, so a newly promoted club with no alias fails the suite.

### Q. Tell me about a production bug.

The ratings table and the match history spell clubs differently — Understat says
"Manchester City," football-data.co.uk says "Man City." A helper called
`_canonical_team` reconciles them, and it was applied to `/predict`.

It was never applied to the Elo chart endpoint. So the team-strength panel looked
up `/teams/{name}/elo` with the ratings-table spelling, 44 of 168 clubs returned
404, and the React component renders `null` on error — so the chart was **simply
absent, with no message**, for a quarter of the site. Barcelona and Real Madrid
worked because those names happen to agree across sources, which is exactly why
it survived casual checking.

Two lessons. **A component that renders nothing on error hides its own bugs** —
an empty space looks like a design choice, not a failure. And **a fix belongs at
the boundary, applied everywhere, not at the one call site where the bug was
first noticed**; the correct helper had existed for months, one function away.

The guard is now data-driven rather than a list of names, so the next promoted
club with no alias breaks the test suite instead of quietly losing its chart.
