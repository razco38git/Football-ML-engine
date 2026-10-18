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
from footballml.models.evaluate import rps_per_match  # noqa: E402

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
    print(f"clubs: {len(big_five)} rateable, {len(others)} not")

    domestic_long = load_team_match_history(TMH)
    euro_long = european_long(kept, dropped)
    print(f"long rows: {len(domestic_long)} domestic, {len(euro_long)} European")

    # Everything in one walk, in date order, so a European tie updates the
    # ratings the next domestic match carries in and vice versa. Production
    # passes the European rows as `extra` instead, because there they must not
    # reach a rolling form window; here they are the subject, so they belong in
    # the frame.
    walked = add_elo(pd.concat([domestic_long, euro_long], ignore_index=True))
    ratings = final_ratings(pd.concat([domestic_long, euro_long], ignore_index=True))

    print("\n=== 1. Does Elo actually learn anything about the other clubs? ===")
    appearances = (
        pd.concat([euro_long["Team"], euro_long["Opponent"]]).value_counts()
    )
    other_ratings = ratings[ratings.index.isin(others)]
    big_ratings = ratings[ratings.index.isin(big_five)]
    print(f"  rated non-big-five clubs: {len(other_ratings)} of {len(others)}")
    for label, series in (("non-big-five", other_ratings), ("big five", big_ratings)):
        if series.empty:
            continue
        print(
            f"  {label:13s} n={len(series):4d}  mean={series.mean():7.1f}  "
            f"sd={series.std():6.1f}  min={series.min():7.1f}  max={series.max():7.1f}  "
            f"|rating-{START}|>50: {(series.sub(START).abs() > 50).mean():.1%}"
        )
    euro_only = appearances[appearances.index.isin(others)]
    print(
        f"  European matches per non-big-five club: median {euro_only.median():.0f}, "
        f"max {euro_only.max():.0f}"
    )
    print("  strongest and weakest of them:")
    for team, value in pd.concat([other_ratings.head(5), other_ratings.tail(5)]).items():
        print(f"    {str(team):28s} {value:7.1f}  ({int(appearances.get(team, 0))} matches)")

    wide = _wide(walked)
    domestic_wide = wide[~wide["League"].isin(kept["League"].unique())]
    euro_wide = wide[wide["League"].isin(kept["League"].unique())]
    print(f"\n  matches with both Elos: {len(domestic_wide)} domestic, {len(euro_wide)} European")

    model = fit_elo_only(domestic_wide)
    print(f"  Elo-only head fitted on {len(domestic_wide)} domestic matches")

    print("\n=== 2. What does dropping to Elo alone cost, where both can be measured? ===")
    scored = pd.read_csv(SCORED, low_memory=False)
    scored["Date"] = pd.to_datetime(scored["Date"])
    scored["result"] = scored["FTR"]
    both = scored.merge(
        euro_wide[["Date", "HomeTeam", "AwayTeam", "elo_diff"]],
        on=["Date", "HomeTeam", "AwayTeam"],
        how="inner",
    ).dropna(subset=["prob_home_win", "prob_draw", "prob_away_win", "elo_diff"])
    print(f"  ties the full model and Elo both cover: {len(both)}")
    if len(both):
        full = both[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy()
        elo_only = _probs(model, both)
        base = _base_rate(both, both)
        print(f"  full model  RPS {_rps(both, full):.5f}")
        print(f"  Elo only    RPS {_rps(both, elo_only):.5f}")
        print(f"  base rate   RPS {_rps(both, base):.5f}")
        if ADJUSTMENT.exists():
            holdout = json.loads(ADJUSTMENT.read_text(encoding="utf-8")).get("holdout", {})
            print(
                f"  (for reference, the shipped cross-league holdout: "
                f"uncorrected {holdout.get('rps_uncorrected')}, "
                f"corrected {holdout.get('rps_corrected')}, n={holdout.get('n')})"
            )

    print("\n=== 3. The ties nobody can predict today ===")
    unrateable = euro_wide[
        ~(euro_wide["HomeTeam"].isin(big_five) & euro_wide["AwayTeam"].isin(big_five))
    ]
    print(f"  ties with at least one unrateable club: {len(unrateable)}")
    if len(unrateable):
        elo_only = _probs(model, unrateable)
        base = _base_rate(unrateable, unrateable)
        print(f"  Elo only    RPS {_rps(unrateable, elo_only):.5f}")
        print(f"  base rate   RPS {_rps(unrateable, base):.5f}")
        neither = unrateable[
            ~(unrateable["HomeTeam"].isin(big_five) | unrateable["AwayTeam"].isin(big_five))
        ]
        print(f"  of which neither side is rateable: {len(neither)}")
        if len(neither):
            elo_neither = _rps(neither, _probs(model, neither))
            base_neither = _rps(neither, _base_rate(neither, neither))
            print(f"  Elo only, neither side  RPS {elo_neither:.5f}")
            print(f"  base rate, neither side RPS {base_neither:.5f}")

    print(
        "\nRead it in this order: if Elo-only is far worse than the full model in"
        "\nsection 2, a forward simulation would be built on that gap for two"
        "\nthirds of its matches. And section 1 decides whether section 3's number"
        "\nmeans anything -- ratings clustered at 1500 would make it an artefact of"
        "\nthe rateable side of each tie."
    )

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ratings.rename("elo").to_frame().to_csv(OUT_DIR / "final_ratings.csv")
        print(f"\nWrote {OUT_DIR / 'final_ratings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
