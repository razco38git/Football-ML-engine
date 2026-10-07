# How this project works

A guide to the shape of the system, what each tab does, and the few ideas that
explain most of the code. Written to be read start to finish, once.

## The shape

Five stages, each writing a file the next one reads. Nothing is clever about the
plumbing; the care is all in what goes into each stage and what is deliberately
kept out.

```
  football-data.co.uk  (results, odds)  ─┐
  Understat            (xG)             ├─> data/processed/*.csv ─> features ─> model ─> API ─> site
  FBref                (player stats)   │                                               :8000   :8443
  EA FC exports        (player ratings) ─┘
```

| Stage | Where | Produces |
|---|---|---|
| Ingest | `src/footballml/ingest/` | match results, xG, player stats |
| Player ratings | `src/footballml/players/` | `player_ratings.csv`, `team_strength.csv` |
| Features | `src/footballml/features/` | one row per match, pre-kickoff only |
| Model | `src/footballml/models/` | a versioned artifact in `models/` |
| Serve | `src/footballml/api/` + `web/` | the site |

`pipelines/` holds the command-line entry points that run those stages.
`pipelines/weekly.py` runs them in dependency order every Monday.

## The one discipline that matters: no leakage

Every feature describes what was knowable **before kick-off**. This is the
constraint that makes the numbers mean anything, and it is enforced rather than
trusted.

Rolling form uses `.shift(1)` so a match never sees itself. Elo records the
rating each side carried *into* a match and only then applies the update. Squad
strength joins the **previous** season, because ratings are whole-season numbers
and joining a match to its own season would let May predict October.

The definitive check rebuilds the features from truncated data and demands the
overlap match exactly (`tests/test_leakage.py`):

```python
def test_truncation_invariance(tmh, features):
    """Features must not change when future matches are removed from the input."""
    cutoff = pd.Timestamp("2025-03-01")
    truncated = build_match_features(tmh[tmh["Date"] < cutoff].copy())
    for col in _feature_cols(full_subset):
        pd.testing.assert_series_equal(full_subset[col], trunc_subset[col], ...)
```

If a feature can see the future, this fails. It covers every column, so new
features are protected without anyone remembering to write a test.

## The match model

**It does not classify home/draw/away.** It predicts *how many goals each side
is expected to score* and derives everything else from the resulting scoreline
distribution.

Two gradient boosters with a Poisson objective — one for home goals, one for
away — then a Dixon-Coles matrix turns the two rates into a grid of exact
scores. 1X2, over/under, both-teams-to-score and the likeliest scoreline all
come out of that one object, so they can never contradict each other on the page.

```python
DEFAULT_GBM_PARAMS = {
    "loss": "poisson",
    "learning_rate": 0.02,    # swept 2026-10-05; the note in match_model.py has the curve
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,  # its validation split is random, so on time-ordered
    "random_state": 7,        # data it would train against future matches
}
```

Dixon-Coles exists because independent Poissons get low scores wrong: real
football has more 0-0 and 1-1 draws than independence predicts. A single fitted
`rho` nudges the four cells at 0-0, 0-1, 1-0 and 1-1 and leaves the rest alone.

### Features, and what actually drives them

235 columns: 107 rolling windows over five matches, 107 over nineteen, three
Elo, twelve squad strength, six other. Measured by mean absolute SHAP over the
last 2,000 complete rows of the feature build, both goal models averaged:

```
Elo rating                   3 features   26.9%   <- elo_diff alone is 24.3%
Expected goals (xG/npxG)    76 features   21.3%
Shots / accuracy            26 features   13.4%
Shots on target             18 features   12.2%
Goals scored/conceded       32 features    8.1%
Deep completions             8 features    6.4%
Squad ratings               12 features    5.9%
Points / W-D-L              36 features    2.2%
Pressing (PPDA)             10 features    1.8%
Rest & congestion            5 features    0.9%
League, window metadata      9 features    0.8%
```

Quote the sample with the figure: these shares move with it, and an earlier
measurement over a different period put `elo_diff` at 17.9% rather than 24.3%.

`elo_diff` is ten times the next single feature. It is not magic: it is the
elected representative of a cluster. It correlates 0.90 with 19-match goal
difference and 0.85 with squad strength, and gradient boosting picks one feature
from a correlated group and leaves the others unused.

**This table is not an attribution of predictive power**, and the difference
matters enough that it has its own section in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md). SHAP measures how much *this fitted
model's output* moves with a feature. It does not measure how much performance
that feature is responsible for, and under redundancy the two come apart badly:
squad ratings are 5.9% here and 19.8% of the attributed gain when the families
are compared properly, because alone they recover 76% of what the full model
achieves. Four of these families are substitutable views of the same match
results, and `EXPERIMENTS.md` measures what each is actually worth.

**Bookmaker odds are deliberately absent.** They are the strongest single
predictor available and are kept strictly as an evaluation benchmark, because a
model that predicts the market by reading the market has learned nothing.

## The player rating

A 0-99 number per player per season, and the most intricate part of the project.
It is **half EA's rating, half ours**.

Our half ranks a player on per-90 output against *his own position in his own
season* — a centre back is never compared with a striker. Those percentiles roll
up into sub-ratings (defending, creation, finishing, involvement...), and the
sub-ratings into a composite weighted by position in `config/player_rating.yaml`.

EA's half is not averaged in raw. The two scales are different — EA's clusters in
the sixties and seventies, ours spreads across the whole range by construction —
so EA's number is first converted to its percentile within position and season,
then mapped onto our scale. That is why a player can read EA 63, performance 78
and overall 64 and the arithmetic is still right: the blend is of 78 and 49.

Three things worth knowing:

- **It admits what it cannot measure.** If a sub-rating's metrics are missing for
  more than half the pool it is refused rather than invented, and the blend leans
  on EA in proportion. `measured_share` carries that fraction to the page.
- **It is much better at attackers than defenders, because the data is.** FBref
  serves its tackling, possession and passing tables with the player rows
  stripped out. Scored against how EA moved a player the following year, the
  rating tracks goalkeepers at 0.23 and forwards at 0.15, but centre backs at
  0.05.
- **2014/15 and 2015/16 have no defensive data at all.** The page says so rather
  than showing bare dashes.

## Each tab

| Tab | What it answers | Where it comes from |
|---|---|---|
| **Match Predictor** | Who wins the next round, and the likeliest score | `/fixtures/upcoming` and `/matches` — the model, live |
| **What If** | What if these two played tomorrow? | `/predict`, the one live-inference path |
| **Player Ratings** | How good was this player that season | `player_ratings.csv` |
| **Player Similarity** | Who plays like him | cosine distance over two vectors: EA's six attributes, and our sub-ratings |
| **Team Strength** | How good is this squad | `team_strength.csv` |
| **Projected Tables** | Where will everyone finish | Monte Carlo over the model's scoreline grids |
| **Prediction Accuracy** | Were we right | the live store, plus the walk-forward backtest |

The forecast card leads with the **outcome call**, because that is what the model
is confident about. The likeliest exact score sits underneath it, smaller,
because a single scoreline is typically only a ~10% event — showing it first
invited readers to treat it as the forecast.

Team strength builds a notional 4-3-3: `select_eleven` fills each line by
minutes, each line is a minutes-weighted mean rating, and the lines are then
weighted `{goalkeeper: 1, defence: 4, midfield: 3, attack: 3}`.

The projection is mostly bookkeeping. The model already produces a full scoreline
distribution per fixture, so playing the remaining ones out a few thousand times
and counting points is the honest way to turn that into a table with
probabilities, rather than one tidy number per club.

## How a change is judged

The part that is easy to skip and should not be.

**The headline metric is RPS** — ranked probability score, lower is better. It
rewards being confidently right, punishes being confidently wrong, and unlike
accuracy it cares about the whole distribution rather than just the top pick.

Where it currently stands, on 20,013 walk-forward matches:

```
  base rate (always predict the average)   0.2301
  this model                               0.1991
  the bookmakers                           0.1950
```

**A change is measured, not argued.** The walk-forward backtest trains on
everything before season S and predicts S, so every prediction is genuinely
out-of-sample. Two runs are then compared by a *paired* bootstrap over per-match
differences — both runs score the same fixtures, so pairing cancels the dominant
source of variance — and a difference is only claimed when the 95% interval
excludes zero.

That discipline has rejected more than it has accepted. Measured and turned down:
a team-outcome defensive rating, dropping transferred players from squad
strength, EA's raw rating scale, FBref's `playing_time` table, reweighting
defensive midfielders (twice), tuning Elo's constants, a better-posed
attack/defence Elo, and pruning redundant features. Each rejection is recorded in
a docstring beside the thing it concerns, so nobody re-runs it.

The one accepted recently is instructive: three attempts to add *information*
all failed, and the one that worked changed how the model **consumes** the
information it already had.

## Keeping it current

`pipelines/weekly.py` runs every Monday: fetch results, rebuild ratings, retrain,
settle last week's predictions and forecast the next round, reproject the tables,
refresh the European correction, reload the API.

Two notes that have each cost a day:

- **After changing feature code, restart the API — do not reload it.**
  `POST /admin/reload` re-reads the data and the model but **not the code**, so a
  model trained with a new feature meets a server that cannot produce it. The
  reload is now refused rather than breaking every prediction, but the fix is a
  restart.
- **`python -m pipelines.<name>` imports `pipelines/__init__.py` first**, which
  makes stdout tolerant of characters the console cannot encode. Without it a
  Turkish club name killed the European fetch on a Hebrew codepage — the fetch
  had succeeded; the step died printing the report about it.
