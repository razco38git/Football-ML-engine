"""Predict genuinely upcoming, unplayed fixtures.

Trains on all completed matches, pulls the live fixture list from
football-data.co.uk, and forecasts matches that have not happened yet. Unlike
:mod:`pipelines.predict`, there is no result to check against -- this is the
real thing.

Where the source publishes opening odds, the implied market probability is
shown alongside ours. It is never fed to the model; it is there so you can see
where we disagree with the bookmakers, which is the interesting part.

Run with::

    python -m pipelines.predict_upcoming
    python -m pipelines.predict_upcoming --league E0 --explain
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import (  # noqa: E402
    ODDS_COLUMNS,
    PROCESSED_DIR,
    load_team_match_history,
    load_team_strength,
)
from footballml.features.build import (  # noqa: E402
    build_match_features,
    build_upcoming_features,
)
from footballml.ingest.matchhistory import LEAGUES, fetch_fixtures  # noqa: E402
from footballml.labels import humanise  # noqa: E402
from footballml.models.evaluate import odds_implied_probs  # noqa: E402
from footballml.models.match_model import MatchPredictor, feature_columns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", nargs="+", choices=sorted(LEAGUES))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    path = PROCESSED_DIR / "team_match_history_all.csv"
    if not path.exists():
        raise SystemExit("Run `python -m pipelines.build_dataset` first")
    tmh = load_team_match_history(path)

    fixtures = fetch_fixtures(args.league)
    if fixtures.empty:
        raise SystemExit("No upcoming fixtures published right now")

    # Train on every completed match, including the current season so far.
    strength = load_team_strength()
    history = build_match_features(tmh, strength=strength)
    cols = feature_columns(history)
    played = history[history["FTR"].notna()]
    print(
        f"Training on {len(played):,} matches "
        f"(through {played['Date'].max().date()}) ..."
    )
    model = MatchPredictor().fit(played[cols], played["FTHG"], played["FTAG"])

    upcoming = build_upcoming_features(tmh, fixtures, strength=strength)
    if upcoming.empty:
        raise SystemExit("Fixtures found, but none could be matched to known teams")

    # Carry the published odds across: the feature builder only returns feature
    # columns, and we want the market line beside our own for comparison.
    odds_cols = [c for c in ODDS_COLUMNS if c in fixtures.columns]
    if odds_cols:
        key = ["League", "Date", "HomeTeam", "AwayTeam"]
        upcoming = upcoming.merge(
            fixtures[[*key, *odds_cols]].assign(Date=pd.to_datetime(fixtures["Date"])),
            on=key,
            how="left",
        )
    upcoming = upcoming.head(args.limit)

    preds = model.predict_frame(upcoming[cols])
    table = pd.concat([upcoming.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    has_odds = (
        all(c in table.columns for c in ODDS_COLUMNS)
        and table[list(ODDS_COLUMNS)].notna().all(axis=1).any()
    )

    print(f"\n=== {len(table)} upcoming fixtures ===\n")
    header = (
        f"{'date':<11} {'lg':<4} {'match':<34} {'pick':>4} {'xG':>11} "
        f"{'H/D/A %':>18} {'sc':>4}"
    )
    if has_odds:
        header += f" {'market H/D/A':>18}"
    print(header)
    print("-" * len(header))

    for _, r in table.iterrows():
        match = f"{r.HomeTeam} v {r.AwayTeam}"[:33]
        xg = f"{r.expected_goals_home:.2f}-{r.expected_goals_away:.2f}"
        hda = f"{r.prob_home_win * 100:4.1f}/{r.prob_draw * 100:4.1f}/{r.prob_away_win * 100:4.1f}"
        modal = f"{int(r.modal_score_home)}-{int(r.modal_score_away)}"
        line = (
            f"{r.Date.date()!s:<11} {r.League:<4} {match:<34} "
            f"{r.predicted_outcome:>4} {xg:>11} {hda:>18} {modal:>4}"
        )
        if has_odds:
            if pd.notna(r.get("B365H")):
                mkt = odds_implied_probs(pd.DataFrame([r]), ODDS_COLUMNS)[0] * 100
                line += f" {mkt[0]:5.1f}/{mkt[1]:4.1f}/{mkt[2]:4.1f}"
            else:
                line += f" {'-':>18}"
        print(line)

    if args.explain:
        print("\n=== Why? ===")
        drivers = model.explain(upcoming[cols], top_n=3)
        for entry, (_, r) in zip(drivers, upcoming.iterrows(), strict=True):
            print(f"\n{r.HomeTeam} v {r.AwayTeam}")
            for side in ("home", "away"):
                team = r.HomeTeam if side == "home" else r.AwayTeam
                print(f"  {team}:")
                for name, value in entry[side]:
                    print(f"    {'up  ' if value > 0 else 'down'} {value:+.3f}  {humanise(name)}")


if __name__ == "__main__":
    main()
