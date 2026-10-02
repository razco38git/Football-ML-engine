"""Fetch UEFA club matches between big-five clubs.

Writes ``data/processed/european_matches.csv`` (the ones we can score) and
``data/processed/european_dropped.csv`` (the ones we cannot, with a reason).

**Read the dropped report.** Club names are canonicalised through the ``fbref``
alias map, which ``pipelines/derive_aliases.py`` derives from domestic fixtures
only. A big-five club spelled a new way on a European page is dropped silently,
and that is indistinguishable from correctly dropping Benfica -- so the run
prints the most common unmatched names for a human to look at.

Run with::

    python -m pipelines.fetch_european
    python -m pipelines.fetch_european --start-season 2223 --competitions UCL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import PROCESSED_DIR, load_team_match_history  # noqa: E402
from footballml.ingest.european import (  # noqa: E402
    EUROPEAN_LEAGUES,
    FIRST_SEASON,
    big_five_only,
    fetch_european_matches,
    unmatched_clubs,
)
from footballml.ingest.matchhistory import current_season_label  # noqa: E402

MATCHES_OUT = PROCESSED_DIR / "european_matches.csv"
DROPPED_OUT = PROCESSED_DIR / "european_dropped.csv"
TMH = PROCESSED_DIR / "team_match_history_all.csv"

#: How many unmatched club names to print. Long enough that a big-five side
#: cannot hide below the fold.
REVIEW_ROWS = 25

logger = logging.getLogger("fetch_european")


def season_labels(start: str, end: str) -> list[str]:
    """``"1617"`` through ``"2627"`` inclusive."""
    labels, label = [], start
    while label <= end:
        labels.append(label)
        first = int(label[:2]) + 1
        label = f"{first % 100:02d}{(first + 1) % 100:02d}"
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", default=FIRST_SEASON)
    parser.add_argument("--end-season", default=None)
    parser.add_argument(
        "--competitions", nargs="+", choices=sorted(EUROPEAN_LEAGUES), default=None
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    end = args.end_season or current_season_label(pd.Timestamp.today())
    seasons = season_labels(args.start_season, end)
    logger.info("Seasons %s .. %s (%d)", seasons[0], seasons[-1], len(seasons))

    matches = fetch_european_matches(seasons, args.competitions)
    if matches.empty:
        logger.error(
            "Nothing fetched. soccerdata is in the optional 'ingest' extra -- "
            "install it with `pip install -e .[ingest]`."
        )
        return 1

    tmh = load_team_match_history(TMH)
    kept, dropped = big_five_only(matches, tmh)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    kept.to_csv(MATCHES_OUT, index=False)
    dropped.to_csv(DROPPED_OUT, index=False)

    print(f"\n=== Kept: {len(kept)} big-five vs big-five matches ===")
    if not kept.empty:
        table = kept.pivot_table(
            index="Season", columns="League", values="Date", aggfunc="count", fill_value=0
        )
        print(table.to_string())
        print("\nBy league pairing (home vs away domestic division):")
        pairing = kept.pivot_table(
            index="home_league", columns="away_league", values="Date",
            aggfunc="count", fill_value=0,
        )
        print(pairing.to_string())

    print(f"\n=== Dropped: {len(dropped)} ===")
    review = unmatched_clubs(dropped)
    print(f"{len(review)} distinct clubs could not be resolved. Top {REVIEW_ROWS}:")
    print(review.head(REVIEW_ROWS).to_string())
    print(
        "\nCheck this list by eye: anything here that plays in England, Spain,\n"
        "Germany, Italy or France is an alias gap, not a foreign club."
    )
    print(f"\nWrote {MATCHES_OUT}\n      {DROPPED_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
