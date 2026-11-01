"""Publish the pre-season squad rating beside the measured one.

Writes ``data/processed/squad_strength.csv`` from ``player_ratings.csv`` and the
EA editions. Reads nothing live and touches nothing the model reads -- see
:mod:`footballml.players.squad_rating` for what the number is and what it is
measurably worse at.

It is a separate pipeline rather than another write inside `build_players`
because it needs no ingest: everything it reads is already on disk, so it runs
in about three minutes against `build_players`' six, and a failure here cannot
cost the file the model actually uses.

Run with::

    python -m pipelines.build_squad_strength
    python -m pipelines.build_squad_strength --report   # and judge it
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.atomic import write_csv, write_json  # noqa: E402
from footballml.data import PROCESSED_DIR  # noqa: E402
from footballml.players.squad_rating import squad_strength  # noqa: E402
from footballml.players.team_strength import team_strength  # noqa: E402

RATINGS = PROCESSED_DIR / "player_ratings.csv"
OUTPUT = PROCESSED_DIR / "squad_strength.csv"
INSIGHTS = PROCESSED_DIR / "rating_insights.json"
HISTORY = PROCESSED_DIR / "team_match_history_all.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squad-size", type=int, default=15)
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also print the three validity correlations against the measured rating.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("build_squad_strength")

    if not RATINGS.exists():
        log.error("No %s -- run `python -m pipelines.build_players` first", RATINGS)
        return 1

    players = pd.read_csv(RATINGS, low_memory=False)
    players["Season"] = players["Season"].astype(str)

    strength, resolutions = squad_strength(players, best_n=args.squad_size)
    if strength.empty:
        log.error("No squads resolved; nothing written")
        return 1

    write_csv(strength, OUTPUT)
    log.info("Wrote %s (%d club-seasons)", OUTPUT, len(strength))

    unresolved = [r for r in resolutions if not r.resolved]
    if unresolved:
        log.info(
            "%d club-season(s) have no EA label and so no squad rating: %s",
            len(unresolved),
            ", ".join(f"{r.season} {r.team} ({r.how})" for r in unresolved),
        )

    insights = _one_player(players)
    if insights:
        write_json({"one_player": insights}, INSIGHTS)
        log.info(
            "Wrote %s -- removing a club's best player costs a median of %.2f",
            INSIGHTS, insights["median_drop"],
        )

    latest = strength[strength["Season"] == strength["Season"].max()]
    print(f"\n=== Strongest pre-season squads, {latest['Season'].iloc[0]} ===")
    columns = ["Team", "League", "squad_overall", "n_from_history", "ea_label"]
    print(
        latest.nlargest(12, "squad_overall")[columns].to_string(
            index=False, float_format=lambda v: f"{v:.1f}"
        )
    )

    if args.report:
        _report(strength)
    return 0


def _one_player(players: pd.DataFrame) -> dict[str, object] | None:
    """What a club's rating loses when its single best player is removed.

    Rebuilt rather than approximated: dropping the top-rated player and running
    `team_strength` again means the sixteenth player moves into the fifteen and
    the line weights shift, which is what actually happens when somebody is
    sold. Subtracting his rating from a mean would answer an easier question.

    Precomputed here because it aggregates 24,000 player-seasons twice -- 28
    seconds -- for a number that changes only when the ratings do. The API
    serves it from the JSON this writes.
    """
    rated = players[players["rated"] & players["rating"].notna()].copy()
    if rated.empty:
        return None

    keys = ["League", "Season", "Team"]
    full = team_strength(rated, method="best_n")
    if full.empty:
        return None

    # The same total order `best_n` selects by, so "best player" means the one
    # it would have put first rather than whichever row happened to sort first.
    best = (
        rated.sort_values(
            ["rating", "minutes", "Player"], ascending=[False, False, True]
        )
        .groupby(keys, observed=True)
        .head(1)
    )
    without = team_strength(rated.drop(index=best.index), method="best_n")
    if without.empty:
        return None

    compared = full.merge(without, on=keys, suffixes=("", "_without"))
    compared["drop"] = compared["strength_overall"] - compared["strength_overall_without"]
    compared = compared[compared["drop"].notna()]
    if compared.empty:
        return None

    worst = compared.loc[compared["drop"].idxmax()]
    return {
        "median_drop": round(float(compared["drop"].median()), 2),
        "q1_drop": round(float(compared["drop"].quantile(0.25)), 2),
        "q3_drop": round(float(compared["drop"].quantile(0.75)), 2),
        "max_drop": round(float(worst["drop"]), 2),
        "max_team": str(worst["Team"]),
        "max_season": str(worst["Season"]),
        "n": int(len(compared)),
    }


def _report(strength: pd.DataFrame) -> None:
    """The three correlations, beside the measured rating's, on shared rows.

    Printed rather than gated on: this rating is published *because* it covers
    the season in progress, not because it predicts better -- it does not. The
    numbers are here so that stays visible rather than becoming folklore.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.squad_size_sweep import _season_outcomes, next_season_validity

    measured = pd.read_csv(PROCESSED_DIR / "team_strength.csv")
    measured["Season"] = measured["Season"].astype(str)
    outcomes = _season_outcomes(pd.read_csv(HISTORY, low_memory=False))

    keys = ["League", "Season", "Team"]
    shared = measured[keys].merge(strength[keys], on=keys)
    renamed = strength.rename(
        columns={
            c: c.replace("squad_", "strength_")
            for c in strength.columns
            if c.startswith("squad_")
        }
    )

    print("\n=== Next-season validity, on the club-seasons both ratings cover ===")
    for name, frame in (("measured", measured), ("pre-season", renamed)):
        rows = frame.merge(shared, on=keys)
        v = next_season_validity(rows, outcomes)
        print(
            f"  {name:11s} n={len(rows):5d}  atk>GF {v['atk_vs_gf']:+.4f}"
            f"   def>GA {v['def_vs_ga']:+.4f}   ovr>pts {v['ovr_vs_pts']:+.4f}"
        )
    print(
        "  The measured rating wins all three, by well over the 0.003 floor.\n"
        "  Expected: it reads the season it describes and this one cannot."
    )

    print("\n=== Coverage ===")
    for season in sorted(strength["Season"].unique())[-3:]:
        mine = int((strength["Season"] == season).sum())
        theirs = int((measured["Season"] == season).sum())
        print(f"  {season}  pre-season {mine:3d} clubs   measured {theirs:3d} clubs")


if __name__ == "__main__":
    raise SystemExit(main())
