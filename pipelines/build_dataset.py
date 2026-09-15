"""Build the full multi-league dataset from scratch.

Fetches football-data.co.uk results and Understat xG for the top five European
leagues, reconciles team names, and writes the processed tables the feature
builder and backtest consume.

Run with::

    python -m pipelines.build_dataset
    python -m pipelines.build_dataset --leagues E0 SP1 --start-season 1819
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import PROCESSED_DIR  # noqa: E402
from footballml.ingest.matchhistory import (  # noqa: E402
    LEAGUES,
    fetch_match_history,
    to_matches,
    to_team_match_history,
)
from footballml.ingest.understat import (  # noqa: E402
    fetch_understat_team_match,
    merge_xg,
    season_labels,
)

MATCHES_OUT = PROCESSED_DIR / "matches_all.csv"
TMH_OUT = PROCESSED_DIR / "team_match_history_all.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", default=sorted(LEAGUES), choices=sorted(LEAGUES))
    parser.add_argument("--start-season", default="1011")
    parser.add_argument(
        "--end-season", default=None, help="Defaults to the season currently underway."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("build_dataset")

    seasons = season_labels(args.start_season, args.end_season)
    log.info("Building %s over %d seasons", ", ".join(args.leagues), len(seasons))

    raw = fetch_match_history(args.leagues, seasons)
    matches = to_matches(raw)
    tmh = to_team_match_history(matches)
    log.info("Built %d matches -> %d team-match rows", len(matches), len(tmh))

    # Understat starts in 2014/15, so restrict the xG request to seasons it can
    # actually serve rather than logging a warning for every earlier one.
    xg_seasons = [s for s in seasons if s >= "1415"]
    known = set(tmh["Team"].unique())
    xg = fetch_understat_team_match(args.leagues, xg_seasons, known_teams=known)
    log.info("Fetched %d Understat team-match rows", len(xg))

    merged = merge_xg(tmh, xg)
    covered = merged["xGFor"].notna().sum()
    log.info("xG coverage %d/%d (%.1f%%)", covered, len(merged), covered / len(merged) * 100)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_OUT, index=False)
    merged.to_csv(TMH_OUT, index=False)
    log.info("Wrote %s and %s", MATCHES_OUT.name, TMH_OUT.name)

    print("\n=== Coverage by league ===")
    summary = merged.groupby("League").agg(
        team_matches=("Team", "size"),
        teams=("Team", "nunique"),
        seasons=("Season", "nunique"),
        with_xg=("xGFor", lambda s: int(s.notna().sum())),
    )
    summary["xg_pct"] = (summary["with_xg"] / summary["team_matches"] * 100).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    main()
