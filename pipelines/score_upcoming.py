"""Score upcoming fixtures and settle finished ones.

The job that builds the track record. Intended to run on a schedule:

1. Predict every published upcoming fixture and store it, stamped with the model
   version and the moment it was made.
2. Attach results to any stored prediction whose match has since been played.

Step 1 must happen *before* kickoff for the record to mean anything, which is
why this runs on a timer rather than on demand.

Run with::

    python -m pipelines.score_upcoming
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml import registry, store  # noqa: E402
from footballml.data import (  # noqa: E402
    ODDS_COLUMNS,
    PROCESSED_DIR,
    load_team_match_history,
    load_team_strength,
)
from footballml.features.build import build_match_features, build_upcoming_features  # noqa: E402
from footballml.ingest.matchhistory import LEAGUES, fetch_fixtures  # noqa: E402

OUTPUT_COLUMNS = [
    "League", "Date", "HomeTeam", "AwayTeam",
    "expected_goals_home", "expected_goals_away",
    "prob_home_win", "prob_draw", "prob_away_win", "predicted_outcome",
    "modal_score_home", "modal_score_away", "prob_over_2_5", "prob_btts",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", choices=sorted(LEAGUES))
    parser.add_argument("--settle-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("score_upcoming")

    tmh = load_team_match_history(PROCESSED_DIR / "team_match_history_all.csv")
    strength = load_team_strength()

    # Settle first: a fixture predicted last week may have been played since.
    played = build_match_features(tmh, strength=strength)
    settled = store.settle(played[played["FTR"].notna()])
    log.info("Settled %d previously stored predictions", settled)

    if args.settle_only:
        return

    model, metadata = registry.load()
    log.info("Using model %s", metadata.version)

    fixtures = fetch_fixtures(args.leagues)
    if fixtures.empty:
        log.info("No upcoming fixtures published")
        return

    # Without strength here the artifact's feature list cannot be satisfied, and
    # this is the path that writes the live pre-kickoff track record.
    scored = build_upcoming_features(tmh, fixtures, strength=strength)
    if scored.empty:
        log.warning("No fixtures could be matched to known teams")
        return

    preds = model.predict_frame(scored[metadata.feature_names])
    frame = pd.concat([scored.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    odds_cols = [c for c in ODDS_COLUMNS if c in fixtures.columns]
    columns = [*OUTPUT_COLUMNS]
    if odds_cols:
        key = ["League", "Date", "HomeTeam", "AwayTeam"]
        frame = frame.merge(
            fixtures[[*key, *odds_cols]].assign(Date=pd.to_datetime(fixtures["Date"])),
            on=key,
            how="left",
        )
        columns += odds_cols

    added = store.append(frame[columns], metadata.version)
    log.info("Stored %d new predictions", added)

    total = len(store.load())
    done = len(store.settled())
    print(f"\nPrediction store: {total} total, {done} settled, {total - done} awaiting results")


if __name__ == "__main__":
    main()
