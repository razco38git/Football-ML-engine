"""Build the full multi-league dataset from scratch.

Fetches football-data.co.uk results and Understat xG for the top five European
leagues, reconciles team names, and writes the processed tables the feature
builder and backtest consume.

If Understat cannot be reached the results are still published and the xG
already on disk is carried forward -- see the comment beside the fetch for why
that is allowed here and what it costs. ``--require-xg`` restores the old
behaviour of failing instead.

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

from footballml.atomic import write_csv
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
    xg_from_published,
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
    parser.add_argument(
        "--require-xg",
        action="store_true",
        help=(
            "Fail instead of carrying forward the xG already published when "
            "Understat cannot be reached. The old behaviour, kept for a run "
            "whose whole purpose is to refresh xG."
        ),
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

    # Results are published even when the xG top-up cannot be fetched.
    #
    # This is a deliberate exception to the rule above it, and the distinction
    # is between a partial *result set* and a complete one missing an optional
    # column family. football-data and FBref give a row everything the model
    # reads: `EXPERIMENTS.md` measures the whole `xg` family -- 94 of the 235
    # features -- at -0.00010 RPS [-0.00045, +0.00023] when dropped, an
    # interval straddling zero. Stopping the entire weekly for it is the more
    # expensive mistake, and it is the one that was being made: on 2026-11-02
    # understat.com's authoritative nameservers went unreachable and four
    # consecutive runs died here, so results went stale and the model went
    # eight days without a retrain over a feature worth nothing.
    #
    # The xG already on disk is carried forward rather than discarded, so the
    # published file keeps every match Understat has ever served and only
    # matches played during the outage come back without it. A later run fills
    # them in; nothing has to be rebuilt from scratch.
    try:
        xg = fetch_understat_team_match(args.leagues, xg_seasons, known_teams=known)
        log.info("Fetched %d Understat team-match rows", len(xg))
    except Exception as exc:  # noqa: BLE001 -- see below
        # Broad on purpose. The failure arrives from three layers -- DNS, TLS
        # and soccerdata's own parsing -- as a different exception type each
        # time, and enumerating them would mean a new outage mode slips
        # through as a crash. The type and message are logged, and the
        # coverage line below is what makes a silent problem visible.
        if args.require_xg:
            raise
        log.error("Understat unreachable (%s: %s)", type(exc).__name__, exc)
        xg = xg_from_published(TMH_OUT)
        log.warning(
            "Carrying forward %d previously published xG rows. Results are "
            "current; matches played since the last successful fetch have no "
            "xG until a later run adds it.",
            len(xg),
        )

    merged = merge_xg(tmh, xg)
    covered = merged["xGFor"].notna().sum()
    log.info("xG coverage %d/%d (%.1f%%)", covered, len(merged), covered / len(merged) * 100)

    # Named rather than left to be inferred from a percentage: a reader of the
    # log should not have to work out which matches are affected.
    stale = merged[merged["Season"] >= "1415"]
    gaps = stale[stale["xGFor"].isna()]
    if not gaps.empty:
        log.warning(
            "%d team-match rows since 2014/15 have no xG, newest %s. Re-run "
            "once Understat answers to fill them.",
            len(gaps),
            gaps["Date"].max().date() if gaps["Date"].notna().any() else "unknown",
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(matches, MATCHES_OUT)
    write_csv(merged, TMH_OUT)
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
