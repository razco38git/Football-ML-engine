# Football-ML-engine

A match predictor for the top five European leagues and the Champions League,
with the player and team rating systems that feed it.

It forecasts **how many goals each side will score** rather than classifying
home/draw/away, and derives everything else — win probabilities, scorelines,
over/under, both-teams-to-score — from the resulting distribution.

```
  football-data.co.uk  (results, odds)  ─┐
  Understat            (xG)             ├─> ratings ─> features ─> model ─> site
  FBref                (player stats)   │
  EA FC exports        (player ratings) ─┘
```

## How good is it

Measured by walk-forward backtest over **20,013 matches** — each season
predicted by a model trained only on earlier ones, so nothing has seen its own
result:

| | RPS | Accuracy |
|---|---|---|
| base rate | 0.2299 | 44.1% |
| **this model** | **0.1992** | **52.9%** |
| the bookmakers | 0.1950 | 53.9% |

The baseline is the **base-rate forecast** — the long-run 44/25/31 split on
every fixture. Its accuracy is the same as always picking the home side, because
home is always its most likely outcome, but its RPS is that of the whole
distribution and is the number worth beating.

**8.9 points of accuracy above the baseline**, covering **88%** of the distance
from guessing to the bookmaker benchmark — which stays 0.0042 RPS ahead, and is
a strong benchmark rather than a theoretical ceiling: it prices team news,
lineups and injuries, none of which this model sees. Exact scoreline right 12.8%
of the time.

Every figure in that table comes from the committed backtest, and is one line to
reproduce rather than something to take on trust:

```python
import pandas as pd
from footballml.models.evaluate import ranked_probability_score, accuracy
df = pd.read_csv("data/processed/backtest_predictions.csv")
ranked_probability_score(df.FTR, df[["prob_H", "prob_D", "prob_A"]].to_numpy())
```

RPS — ranked probability score, lower is better — is the headline metric rather
than accuracy, because accuracy only asks whether the top pick came in. RPS
scores the whole distribution, so being confidently wrong costs more than being
unsure and wrong. The site explains it in full on the Prediction Accuracy tab.

Calibration sits close to the diagonal across all ten probability bins: of the
outcomes given 30%, about 30% happen. The model is **well calibrated and
under-discriminating** — the remaining gap to the market is information it does
not have (lineups, injuries), not a missing way of rearranging goals and shots.

## Running it

Two processes, and it needs both — the page is served by Vite on **8443**, and
every tab reads the API on **8000**:

```powershell
.\scripts\start_site.ps1
```

Then open **http://localhost:8443**. Safe to run twice: it starts only what is
not already listening. It is registered to run at login; remove that with

```powershell
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\FootballML.lnk"
```

Both bind to localhost. Opening the site from another device needs uvicorn on
`--host 0.0.0.0` and `VITE_API_URL` pointing at this machine — deliberately not
the default.

### After changing feature code, restart the API — do not reload it

`POST /admin/reload` re-reads the data files and the model artifact but **not
the code**. Retrain after adding a feature and the running server gets a model
expecting columns its own feature builder cannot produce. It now refuses such a
reload and keeps serving the previous model rather than breaking every
prediction, so the failure is loud — but the fix is a restart, not another
retrain.

## The tabs

| Tab | What it answers |
|---|---|
| **Match Predictor** | Who wins the next round, with the model's own reasoning per fixture |
| **Discover** | Where we disagree with the bookmakers, where we are most certain, which outsiders we back |
| **Player Ratings** | How good was this player that season, and what did he actually do |
| **Player Similarity** | Who plays like him, on EA's attributes and on our own percentiles |
| **Team Strength** | Squad quality by line, with each club's Elo history |
| **Projected Tables** | Where everyone finishes, from simulating the rest of the season |
| **Prediction Accuracy** | Were we right — against the bookmakers and against guessing |
| **What If?** | Any two clubs, scored live |

## Keeping it current

`pipelines/weekly.py` runs every Monday: fetch results, rebuild ratings,
retrain, settle last week's predictions and forecast the next round, reproject
the tables, refresh the European correction, reload the API.

```powershell
python -m pipelines.weekly              # the whole job
python -m pipelines.weekly --only train # one step
```

Individual stages are ordinary modules — `build_dataset`, `build_players`,
`train`, `backtest`, `score_upcoming`, `project_season`, `validate_european`.

## Reading the code

**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** is the guide: the five
stages, the leak-safety rule and the test that enforces it, why the model
predicts goal rates, what actually drives the predictions, and how a change is
measured before it ships.

**[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)** is the companion: where every
published number comes from. Each constant, the formula it sits in, and the
measurement that chose it — including the ones that argued against the change
they justify.

**[`EXPERIMENTS.md`](EXPERIMENTS.md)** is what the model's predictive power
actually rests on, measured rather than asserted: an ablation ladder over every
feature family, a Shapley attribution over all sixteen coalitions, and two
hypotheses that were tested and rejected. It opens with the difference between
sufficiency, attribution and necessity, because confusing those is how the first
version of its own headline came out wrong.

Two conventions worth knowing before opening anything:

**Every feature describes what was knowable before kick-off**, and it is
enforced rather than trusted. `tests/test_leakage.py` rebuilds the whole feature
table from truncated data and demands every column match, so a feature that can
see the future fails the suite without anyone writing a test for it.

**Bookmaker odds are never model inputs.** They are the evaluation benchmark —
a model that predicts the market by reading the market has learned nothing — and
`feature_columns` excludes them explicitly, with a test.

Decisions are recorded where they were made. A docstring that looks unusually
long is usually carrying a measurement, often one that argued against the change
it sits beside.

## Testing

```powershell
python -m pytest -q          # 460 tests
python -m ruff check .
cd web; npx tsc --noEmit
```

## Stack

Python 3.13, pandas, numpy, scipy, scikit-learn (one estimator:
`HistGradientBoostingRegressor`), FastAPI, React 19 + Vite + Tailwind v4.
`soccerdata` is an optional `ingest` extra — it pulls Selenium, and the API and
training paths stay installable without a browser stack.
