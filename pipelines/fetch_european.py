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

A narrower ``--start-season`` refreshes only those seasons and leaves the rest
of the file alone; see `merge_seasons`.
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


def merge_seasons(
    existing: pd.DataFrame, fetched: pd.DataFrame, seasons: list[str]
) -> pd.DataFrame:
    """Replace only the seasons this run fetched, keeping the rest.

    This used to be ``fetched.to_csv(...)``, which made ``--start-season`` a
    destructive flag rather than a narrowing one: asking for 2026/27 alone
    rewrote the file with 2026/27 alone, discarding ten seasons of European
    results. Nothing warned, and the loss only shows up later and indirectly,
    in `fit_league_adjustment` quietly refitting the cross-league correction on
    a fraction of the ties it was derived from.

    A season this run asked for is replaced wholesale rather than merged
    row-by-row, because that is what a refetch means: a tie dropped from the
    source, or recognised as a duplicate, must disappear from our copy too.
    Seasons the run did not ask for are untouched.
    """
    if existing.empty:
        return fetched
    if fetched.empty:
        # Nothing fetched is not a reason to delete what is already there.
        return existing

    keep = existing[~existing["Season"].astype(str).isin([str(s) for s in seasons])]
    if keep.empty:
        return fetched

    merged = pd.concat([keep, fetched], ignore_index=True)
    logger.info(
        "Kept %d row(s) from %d season(s) this run did not fetch",
        len(keep), keep["Season"].nunique(),
    )
    return merged.sort_values(["Season", "Date", "HomeTeam"]).reset_index(drop=True)


#: The column the merge cannot work without.
REQUIRED_COLUMN = "Season"


def _read(path: Path) -> pd.DataFrame:
    """What is already on disk, or an empty frame when there is nothing.

    Refuses rather than overwrites when the file is there but unusable. This
    fails *closed*, unlike the player cache, and for the opposite reason: a
    cache that cannot be read costs a refetch, whereas this file may hold ten
    seasons the current run is not fetching, and overwriting it destroys
    exactly what the merge exists to protect.

    The schema check is the part that matters. ``pd.read_csv`` raises on very
    little -- handed binary rubbish it returns a cheerful one-column frame --
    so a parse that merely succeeds proves nothing.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not read {path.name} ({exc}). Not overwriting it.") from exc

    if not frame.empty and REQUIRED_COLUMN not in frame.columns:
        raise SystemExit(
            f"{path.name} has no {REQUIRED_COLUMN!r} column, so the seasons it "
            f"holds cannot be identified. Not overwriting it; move it aside to "
            f"rebuild from scratch."
        )
    return frame


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
    # Merge, never replace. See `merge_seasons`.
    kept = merge_seasons(_read(MATCHES_OUT), kept, seasons)
    dropped = merge_seasons(_read(DROPPED_OUT), dropped, seasons)
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
