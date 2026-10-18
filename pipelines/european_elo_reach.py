"""Could we predict the Champions League forward? Measure the prerequisite.

The site projects domestic tables and not the UCL, and the reason usually given
is the knockout draw. That is not the obstacle -- a draw is an ordinary Monte
Carlo. The obstacle is coverage. Of the European matches on disk, **829 are kept
and 3,178 dropped**, because a tie is only predictable when both clubs are in the
big five: `big_five_only` in `src/footballml/ingest/european.py` drops the rest,
since a club outside England, Spain, Germany, Italy or France has no squad
rating, no form and no xG here. **180 distinct clubs** appear that we cannot
rate, and **1,409 dropped ties have no rateable side at all**.

Simulating the competition would therefore mean inventing two thirds of its own
inputs. Before building any of that, this measures the one thing that decides
whether it is worth starting:

    Is Elo -- which needs only results -- good enough to stand in where form,
    xG and squad strength are missing?

The decisive comparison is on the ties where **both** can be measured: score an
Elo-only predictor on the big-five ties the full model also predicts, and see
what is lost by dropping to Elo alone. `data/processed/league_adjustment.json`
holds the full model's figures for exactly that population.

**The result most likely to be mistaken for success**, and the reason the rating
spread is reported as loudly as the RPS: `SEASON_REGRESSION` pulls every rating
a quarter of the way back to 1500 each summer, and a club outside the big five
plays six to eight European matches a year. Their ratings may never leave the
neighbourhood of 1500 -- in which case Elo "covers" all 180 clubs while knowing
nothing about any of them, and a good headline number would be measuring the
big-five side of each tie.

Writes nothing outside ``experiments/``. Run with::

    python -m pipelines.european_elo_reach
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footballml.data import PROCESSED_DIR, load_team_match_history  # noqa: E402
from footballml.features.build import european_elo_rows  # noqa: E402
from footballml.features.elo import START, add_elo, final_ratings  # noqa: E402
from footballml.models.evaluate import paired_bootstrap, rps_per_match  # noqa: E402

logger = logging.getLogger("european_elo_reach")

OUT_DIR = ROOT / "experiments" / "european_elo_reach"
TMH = PROCESSED_DIR / "team_match_history_all.csv"
KEPT = PROCESSED_DIR / "european_matches.csv"
DROPPED = PROCESSED_DIR / "european_dropped.csv"
SCORED = PROCESSED_DIR / "european_predictions.csv"
ADJUSTMENT = PROCESSED_DIR / "league_adjustment.json"

OUTCOMES = ("H", "D", "A")


def european_long(kept: pd.DataFrame, dropped: pd.DataFrame) -> pd.DataFrame:
    """Every European result, kept and dropped alike, in the Elo walk's shape.

    `european_elo_rows` already does the conversion and takes whatever frame it
    is handed; production only ever hands it the kept file. The dropped file
    carries the same columns plus a `reason`, including `FTHG`/`FTAG`, so the
    results needed to rate the other 180 clubs have been on disk all along.
    """
    frames = [f for f in (kept, dropped) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    both = pd.concat(frames, ignore_index=True)
    both = both.dropna(subset=["FTHG", "FTAG"])
    return european_elo_rows(both)


def _wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per match, with both sides' pre-match Elo.

    The long frame holds two rows per match; the home row carries the home
    side's rating and the away row the away side's, so the two are joined back
    together on the fixture.
    """
    home = long[long["Venue"] == "Home"].rename(
        columns={"Team": "HomeTeam", "Opponent": "AwayTeam", "elo": "elo_home"}
    )
    away = long[long["Venue"] == "Away"][["Date", "Team", "elo"]].rename(
        columns={"Team": "AwayTeam", "elo": "elo_away"}
    )
    merged = home.merge(away, on=["Date", "AwayTeam"], how="inner")
    merged["elo_diff"] = merged["elo_home"] - merged["elo_away"]
    merged["result"] = np.where(
        merged["GoalsFor"] > merged["GoalsAgainst"],
        "H",
        np.where(merged["GoalsFor"] == merged["GoalsAgainst"], "D", "A"),
    )
    return merged


def fit_elo_only(domestic: pd.DataFrame) -> LogisticRegression:
    """Map an Elo gap to home/draw/away probabilities.

    Fitted on **domestic** matches only and applied to European ties, so the
    evaluation population is never in the fit. `_expected` in `elo.py` returns
    an expected *score*, a single number between 0 and 1, which cannot express
    a draw rate -- hence a three-way model rather than reusing it directly.

    Deliberately the simplest thing that can work: one feature, no interactions.
    The question is whether Elo carries enough signal at all, and a richer
    fitted head would blur the answer with its own capacity.
    """
    X = domestic[["elo_diff"]].to_numpy()
    y = domestic["result"].to_numpy()
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def walk_forward_probs(
    domestic: pd.DataFrame, scored: pd.DataFrame, min_train: int = 2000
) -> tuple[np.ndarray, pd.DataFrame]:
    """Score each European season with a head fitted only on earlier matches.

    The first version of this fitted one head on all 29,190 domestic matches and
    applied it to ties from every season, so a 2017 tie was scored by a mapping
    that had seen 2024. The ratings were always pre-match; the *calibration* was
    not, and on a result this close to its benchmark that is not a detail to
    wave through.

    One fit per European season, on domestic matches strictly before that
    season's first tie -- the convention `run_backtest` uses. The head is a
    one-feature logistic regression, so eleven fits cost nothing.

    Returns:
        ``(probs, scored_rows)`` covering only the seasons with enough history
        to train on, so the caller scores exactly the rows that were predicted.
    """
    chunks, kept_rows = [], []
    for season, block in scored.groupby(scored["Season"].astype(str), sort=True):
        cutoff = block["Date"].min()
        train = domestic[domestic["Date"] < cutoff]
        if len(train) < min_train or train["result"].nunique() < len(OUTCOMES):
            # Not enough history to calibrate against. Dropping the season is
            # honest; fitting it on the future is what this function exists to
            # stop.
            logger.info("  %s: only %d prior domestic matches, skipped", season, len(train))
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(train[["elo_diff"]].to_numpy(), train["result"].to_numpy())
        chunks.append(_probs(model, block))
        kept_rows.append(block)
    if not chunks:
        return np.empty((0, len(OUTCOMES))), scored.head(0)
    return np.vstack(chunks), pd.concat(kept_rows)


def _probs(model: LogisticRegression, frame: pd.DataFrame) -> np.ndarray:
    """Columns ordered H, D, A, whatever order the fit put the classes in."""
    raw = model.predict_proba(frame[["elo_diff"]].to_numpy())
    order = [list(model.classes_).index(o) for o in OUTCOMES]
    return raw[:, order]


def _rps(frame: pd.DataFrame, probs: np.ndarray) -> float:
    # `result` already holds H/D/A, which is exactly what `rps_per_match` wants.
    # Mapping it to 0/1/2 first is the mistake that made every score in an
    # earlier sweep a function of the forecast alone.
    return float(rps_per_match(frame["result"], probs).mean())


def _base_rate(fit_on: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """The same three numbers for every match: how often each outcome happens.

    The floor any model has to clear. Estimated on the same population it is
    scored against, which flatters it slightly -- deliberately, so that beating
    it means something.
    """
    rates = fit_on["result"].value_counts(normalize=True)
    row = np.array([rates.get(o, 0.0) for o in OUTCOMES])
    return np.tile(row, (len(frame), 1))


def walk(domestic: pd.DataFrame, *european: pd.DataFrame) -> pd.DataFrame:
    """One Elo walk over domestic plus whichever European results it may learn
    from, returned one row per match with both sides' pre-match rating.

    Production passes European ties to `add_elo` as ``extra=`` so they never
    reach a rolling form window; the ratings that come out are the same either
    way, and here the European rows are the subject, so they belong in the
    frame.
    """
    frames = [domestic, *(f for f in european if f is not None and not f.empty)]
    return _wide(add_elo(pd.concat(frames, ignore_index=True)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Save tables under experiments/.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kept = pd.read_csv(KEPT, low_memory=False)
    dropped = pd.read_csv(DROPPED, low_memory=False)
    print(f"European results: {len(kept)} kept, {len(dropped)} dropped")

    big_five = set(kept["HomeTeam"]) | set(kept["AwayTeam"])
    everyone = big_five | set(dropped["HomeTeam"]) | set(dropped["AwayTeam"])
    others = everyone - big_five
    print(f"clubs: {len(big_five)} rateable by the model, {len(others)} not")

    domestic_long = load_team_match_history(TMH)
    kept_long = european_long(kept, None)
    dropped_long = european_long(None, dropped)
    print(f"long rows: {len(domestic_long)} domestic, {len(kept_long)} kept, "
          f"{len(dropped_long)} dropped")

    # Two walks. `production` learns from exactly what `add_elo(extra=...)` is
    # given today; `extended` also learns from the ties the model cannot
    # predict. The difference between them is the whole question.
    production = walk(domestic_long, kept_long)
    extended = walk(domestic_long, kept_long, dropped_long)

    comps = set(kept["League"].unique())
    dom_p = production[~production["League"].isin(comps)]
    euro_p = production[production["League"].isin(comps)]
    euro_e = extended[extended["League"].isin(comps)]
    dom_e = extended[~extended["League"].isin(comps)]

    print("\n=== 1. What the extension does to the ratings ===")
    ratings = final_ratings(pd.concat([domestic_long, kept_long, dropped_long], ignore_index=True))
    other_r, big_r = ratings[ratings.index.isin(others)], ratings[ratings.index.isin(big_five)]
    print(f"  rated non-big-five clubs: {len(other_r)} of {len(others)}")
    for label, series in (("non-big-five", other_r), ("big five", big_r)):
        print(
            f"  {label:13s} n={len(series):4d}  mean={series.mean():7.1f}  "
            f"sd={series.std():6.1f}  spread={series.max() - series.min():6.1f}  "
            f"|r-{START:.0f}|>50: {(series.sub(START).abs() > 50).mean():.1%}"
        )
    print("  strongest and weakest outside the big five:")
    for team, value in pd.concat([other_r.head(4), other_r.tail(4)]).items():
        print(f"    {str(team):26s} {value:7.1f}")

    print("\n=== 2. Does learning from the dropped ties help where we can check? ===")
    scored = pd.read_csv(SCORED, low_memory=False)
    scored["Date"] = pd.to_datetime(scored["Date"])
    scored["result"] = scored["FTR"]

    arms = {}
    for name, euro, dom in (("production", euro_p, dom_p), ("extended", euro_e, dom_e)):
        both = scored.merge(
            euro[["Date", "HomeTeam", "AwayTeam", "elo_diff"]],
            on=["Date", "HomeTeam", "AwayTeam"], how="inner",
        ).dropna(subset=["prob_home_win", "prob_draw", "prob_away_win", "elo_diff"])
        probs, rows = walk_forward_probs(dom, both)
        arms[name] = (rows, probs)

    # The arms must score the same fixtures or the comparison is meaningless --
    # asserted rather than trusted, as `align` in ablation_ladder puts it.
    keys = [set(zip(r["Date"], r["HomeTeam"], r["AwayTeam"], strict=True))
            for r, _ in arms.values()]
    shared = set.intersection(*keys)
    print(f"  ties scored by both arms: {len(shared)}")
    assert len(shared) == len(arms["production"][0]) == len(arms["extended"][0]), (
        "the two arms scored different fixtures; the comparison would be invalid"
    )

    rows, _ = arms["production"][0], None
    full = rows[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy()
    print(f"  full model          RPS {_rps(rows, full):.5f}")
    for name, (r, probs) in arms.items():
        print(f"  Elo only, {name:11s} RPS {_rps(r, probs):.5f}")
    print(f"  base rate           RPS {_rps(rows, _base_rate(rows, rows)):.5f}")

    # Paired on identical fixtures, so the interval is the right one to read.
    # 827 ties is a fortieth of the walk-forward backtest, so the bar for
    # believing a difference here is correspondingly higher.
    pr, pp = arms["production"]
    er, ep = arms["extended"]
    boot = paired_bootstrap(
        rps_per_match(pr["result"], pp), rps_per_match(er["result"], ep)
    )
    print(
        f"  extended vs production: {boot['delta']:+.5f} "
        f"[{boot['lo']:+.5f}, {boot['hi']:+.5f}] "
        f"{'significant' if boot['excludes_zero'] else 'not significant'}"
    )
    vs_full = paired_bootstrap(
        rps_per_match(rows["result"], full), rps_per_match(er["result"], ep)
    )
    print(
        f"  extended vs full model: {vs_full['delta']:+.5f} "
        f"[{vs_full['lo']:+.5f}, {vs_full['hi']:+.5f}] "
        f"{'significant' if vs_full['excludes_zero'] else 'not significant'}"
    )
    # A bootstrap interval that nearly touches zero, on a fortieth of the
    # backtest's sample, is exactly the shape this project has twice been
    # misled by: the resampling treats fixtures as independent when matches
    # within a season share a calibration. So require the season record too.
    print()
    print("  by season (negative = the extended walk is better):")
    per = pr.assign(
        d=rps_per_match(er["result"], ep) - rps_per_match(pr["result"], pp)
    )
    wins = 0
    seasons = 0
    for season, block in per.groupby(per["Season"].astype(str), sort=True):
        if len(block) < 20:
            continue
        seasons += 1
        wins += block["d"].mean() < 0
        print(f"    {season}  n={len(block):4d}  {block['d'].mean():+.5f}")
    print(f"  improved in {wins} of {seasons} seasons")

    if ADJUSTMENT.exists():
        h = json.loads(ADJUSTMENT.read_text(encoding="utf-8")).get("holdout", {})
        print(f"  (shipped cross-league holdout: corrected {h.get('rps_corrected')}, "
              f"n={h.get('n')})")

    print("\n=== 3. The ties production cannot score at all ===")
    unrateable = euro_e[
        ~(euro_e["HomeTeam"].isin(big_five) & euro_e["AwayTeam"].isin(big_five))
    ]
    probs, rows = walk_forward_probs(dom_e, unrateable)
    print(f"  ties with an unrateable club: {len(unrateable)}, scored {len(rows)}")
    print("  production cannot score these: the clubs never enter its walk.")
    if len(rows):
        print(f"  Elo only, extended  RPS {_rps(rows, probs):.5f}")
        print(f"  base rate           RPS {_rps(rows, _base_rate(rows, rows)):.5f}")
        neither = rows[
            ~(rows["HomeTeam"].isin(big_five) | rows["AwayTeam"].isin(big_five))
        ]
        if len(neither):
            idx = rows.index.get_indexer(neither.index)
            print(f"  of which neither side is rateable: {len(neither)}")
            print(f"  Elo only, neither   RPS {_rps(neither, probs[idx]):.5f}")
            print(f"  base rate, neither  RPS {_rps(neither, _base_rate(neither, neither)):.5f}")

    print(
        "\nThe head is now fitted per season on earlier domestic matches only, so"
        "\nno tie is scored by a calibration that saw its future. Shipping the"
        "\nextended walk would change `elo` for domestic matches too and needs a"
        "\nfull walk-forward backtest first -- not measured here."
    )

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ratings.rename("elo").to_frame().to_csv(OUT_DIR / "final_ratings.csv")
        print(f"\nWrote {OUT_DIR / 'final_ratings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
