"""Fetch Understat xG and merge it into the team-match history.

Writes ``data/processed/team_match_history_xg.csv``. Run before the backtest to
enable the xG feature groups::

    python -m pipelines.ingest_xg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import PROCESSED_DIR, load_team_match_history  # noqa: E402
from footballml.ingest.understat import (  # noqa: E402
    UNDERSTAT_LEAGUES,
    fetch_understat_team_match,
    merge_xg,
    season_labels,
)

OUTPUT = PROCESSED_DIR / "team_match_history_xg.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="E0", choices=sorted(UNDERSTAT_LEAGUES))
    parser.add_argument("--start-season", default="1415")
    parser.add_argument("--end-season", default="2425")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("ingest_xg")

    tmh = load_team_match_history()
    known = set(tmh["Team"].unique())

    seasons = season_labels(args.start_season, args.end_season)
    log.info("Fetching %d seasons of %s", len(seasons), args.league)

    xg = fetch_understat_team_match(
        league=UNDERSTAT_LEAGUES[args.league], seasons=seasons, known_teams=known
    )
    log.info("Fetched %d team-match rows", len(xg))

    merged = merge_xg(tmh, xg)
    covered = merged["xGFor"].notna().sum()
    log.info(
        "xG coverage %d/%d rows (%.1f%%)", covered, len(merged), covered / len(merged) * 100
    )

    merged.to_csv(OUTPUT, index=False)
    log.info("Wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
