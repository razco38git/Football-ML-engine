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
from footballml.ingest.fbref_results import (  # noqa: E402
    fetch_fbref_results,
    top_up,
)
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

#: football-data's own results, one row per match, and nothing else's.
#:
#: Deliberately *not* the feature table. The FBref and Understat top-ups below
#: are applied to `team_match_history_all.csv` after this is written, so the two
#: diverge whenever football-data is behind -- on 2026-10-19 this file ended on
#: the 12th while the history reached the 18th. That is the contract, not a bug:
#: this is the single-source archive in match shape, the history is the merged
#: multi-source table the model reads.
#:
#: It has one consumer besides this module. `tests/test_team_names.py` takes it
#: as the authority for which club names the match data knows, which is what
#: stops a newly promoted side silently losing its Elo chart. Nothing in the
#: daily or weekly path reads it any more -- the refresh gate used to, and that
#: was the bug fixed in `f3c11e1`.
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

    # football-data falls behind -- three weeks, in October 2026 -- and that
    # froze the projected tables, because `project_season` simulates from this
    # table and will not simulate from rows missing half their columns.
    #
    # Of what football-data brings, only goals, shots and shots on target ever
    # reach the model: fouls, corners and cards are stored and read by nothing.
    # FBref publishes all three, Understat supplies the xG family below, so a
    # topped-up row is complete in every column that becomes a feature. It is
    # not an approximation standing in for one, which is why it is allowed here
    # rather than only on the display path.
    #
    # Adds only. football-data's own rows always win, so a week when it is
    # current produces exactly the frame it produced before.
    # The current season only. Everything earlier is finished and football-data
    # published it long ago, so there is nothing there to top up -- and asking
    # for more is expensive in a way that is easy to miss: FBref is rate limited
    # to one page every seven seconds, so a first run over twelve seasons of
    # five leagues spent 51 minutes scraping 2023/24 Serie A before it was
    # stopped. One season is a couple of minutes.
    tmh = top_up(tmh, fetch_fbref_results(args.leagues, seasons[-1:], known=tmh))

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
