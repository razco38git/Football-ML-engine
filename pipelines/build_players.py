"""Fetch player statistics and compute the 0-99 rating.

Writes ``data/processed/player_ratings.csv``.

Run with::

    python -m pipelines.build_players
    python -m pipelines.build_players --start-season 2223 --top 25
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import PROCESSED_DIR  # noqa: E402
from footballml.ingest.understat import UNDERSTAT_LEAGUES, season_labels  # noqa: E402
from footballml.players.ingest import fetch_player_seasons  # noqa: E402
from footballml.players.rating import latest_ratings, rate_players  # noqa: E402
from footballml.players.team_strength import team_strength  # noqa: E402

OUTPUT = PROCESSED_DIR / "player_ratings.csv"
TEAM_OUTPUT = PROCESSED_DIR / "team_strength.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", choices=sorted(UNDERSTAT_LEAGUES))
    parser.add_argument("--start-season", default="1415")
    parser.add_argument("--end-season", default=None)
    parser.add_argument("--top", type=int, default=20, help="How many to print.")
    parser.add_argument(
        "--squad-size",
        type=int,
        default=15,
        help="Players per team when --team-method=best_n.",
    )
    parser.add_argument(
        "--team-method",
        default="best_n",
        choices=["eleven", "best_n", "squad"],
        help="How a team rating is built. Default is the top-rated core squad.",
    )
    args = parser.parse_args()

    # Player names carry accents that a default Windows console codepage cannot
    # encode, which otherwise kills the script at the final print after all the
    # work is done.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("build_players")

    seasons = season_labels(args.start_season, args.end_season)
    log.info("Fetching players across %d seasons", len(seasons))

    players = fetch_player_seasons(args.leagues, seasons)
    log.info("Fetched %d player-seasons", len(players))

    rated = rate_players(players)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rated.to_csv(OUTPUT, index=False)
    log.info("Wrote %s", OUTPUT)

    print("\n=== Coverage ===")
    print(
        rated.assign(status=rated["unrated_reason"].fillna("rated"))
        .groupby("status")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    current = latest_ratings(rated[rated["rated"]])
    print(f"\n=== Top {args.top} players (most recent rated season) ===")
    cols = ["rating", "Player", "Team", "position_group", "Season", "minutes"]
    subs = [c for c in current.columns if c.startswith("sub_")]
    print(
        current.nlargest(args.top, "rating")[cols + subs].to_string(
            index=False, float_format=lambda v: f"{v:.0f}"
        )
    )

    print("\n=== Rating distribution (rated players, latest season) ===")
    print(current["rating"].describe().to_string(float_format=lambda v: f"{v:.1f}"))

    strength = team_strength(rated, method=args.team_method, best_n=args.squad_size)
    if not strength.empty:
        strength.to_csv(TEAM_OUTPUT, index=False)
        log.info("Wrote %s", TEAM_OUTPUT)

        strength["Season"] = strength["Season"].astype(str)
        latest = strength[strength["Season"] == strength["Season"].max()]
        lines = [c for c in latest.columns if c.startswith("strength_")]
        print(f"\n=== Strongest squads, {latest['Season'].iloc[0]} ===")
        print(
            latest.nlargest(12, "strength_overall")[["Team", "League", *lines]].to_string(
                index=False, float_format=lambda v: f"{v:.1f}"
            )
        )


if __name__ == "__main__":
    main()
