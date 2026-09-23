"""Project the final table for each league by simulating the rest of the season.

Run with::

    python -m pipelines.project_season
    python -m pipelines.project_season --leagues E0 --sims 2000

.. warning::
    **Every remaining fixture is scored with form as it stands today**, held
    constant through to May. Form is the model's strongest input -- recent xG
    and shot volume outrank squad strength -- so a projection made five
    matchweeks into a season is closer to a squad-strength prior than a genuine
    forecast, and it cannot know that a team about to go on a ten-game run is
    about to do so.

    Simulating evolving form would mean re-deriving features inside the loop and
    is a much larger job. The page says how many matches are still to play so a
    reader can weigh the number accordingly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml import registry  # noqa: E402
from footballml.data import (  # noqa: E402
    PROCESSED_DIR,
    load_team_match_history,
    load_team_strength,
)
from footballml.features.build import build_upcoming_features  # noqa: E402
from footballml.models.dixon_coles import score_matrix  # noqa: E402
from footballml.projection import (  # noqa: E402
    current_table,
    remaining_fixtures,
    simulate,
)

OUTPUT = PROCESSED_DIR / "season_projection.csv"

#: Relegation places per league, and how many qualify for the Champions League.
#:
#: Approximate on purpose. Coefficient-based extra places move between leagues
#: from season to season, and the Bundesliga and Ligue 1 send their third-bottom
#: to a play-off rather than straight down -- counted here as relegated, since
#: most play-offs are lost. Edit this one dict rather than the UI.
LEAGUE_RULES: dict[str, dict[str, int]] = {
    "E0": {"relegation": 3, "top": 4},
    "SP1": {"relegation": 3, "top": 4},
    "I1": {"relegation": 3, "top": 4},
    "D1": {"relegation": 3, "top": 4},
    "F1": {"relegation": 3, "top": 4},
}

logger = logging.getLogger("project_season")


def project(
    tmh: pd.DataFrame,
    league: str,
    season: str,
    model: object,
    feature_names: list[str],
    strength: pd.DataFrame,
    n_sims: int,
) -> pd.DataFrame:
    """One league's projected table."""
    fixtures = remaining_fixtures(tmh, league, season)
    table = current_table(tmh, league, season)
    logger.info(
        "%s: %d teams, %d played, %d remaining",
        league, len(table), int(table["played"].sum() // 2), len(fixtures),
    )

    if fixtures.empty:
        matrices = None
    else:
        # Dated beyond the last result so the feature builder treats them as
        # unplayed and each side picks up its current form.
        dated = fixtures.assign(
            Date=pd.Timestamp(tmh["Date"].max()) + pd.Timedelta(days=7)
        )
        scored = build_upcoming_features(tmh, dated, strength=strength)
        if len(scored) != len(fixtures):
            logger.warning(
                "%s: features built for %d of %d fixtures; the rest are teams "
                "the model has no history for",
                league, len(scored), len(fixtures),
            )
            fixtures = scored[["League", "Season", "HomeTeam", "AwayTeam"]]
        mu_home, mu_away = model.predict_goal_rates(scored[feature_names])
        matrices = score_matrix(mu_home, mu_away, rho=model.rho_)

    rules = LEAGUE_RULES.get(league, {"relegation": 3, "top": 4})
    projected = simulate(
        fixtures, matrices, table, n_sims=n_sims,
        relegation_places=rules["relegation"],
    )

    out = projected.merge(table, on="Team", how="left")
    out.insert(0, "League", league)
    out.insert(1, "Season", season)
    out["remaining"] = len(fixtures)
    return out.sort_values("projected_points", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", default=sorted(LEAGUE_RULES))
    parser.add_argument("--season", default=None, help="Defaults to the latest.")
    parser.add_argument("--sims", type=int, default=10_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tmh = load_team_match_history(PROCESSED_DIR / "team_match_history_all.csv")
    season = args.season or str(tmh["Season"].astype(str).max())
    model, metadata = registry.load()
    strength = load_team_strength()
    logger.info("Projecting %s with model %s", season, metadata.version)

    tables = [
        project(tmh, league, season, model, metadata.feature_names, strength, args.sims)
        for league in args.leagues
    ]
    result = pd.concat(tables, ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    logger.info("Wrote %d rows to %s", len(result), OUTPUT.name)

    for league, group in result.groupby("League"):
        top = group.nlargest(3, "projected_points")
        line = ", ".join(
            f"{r.Team} {r.projected_points:.0f} ({r.title_pct:.0%})"
            for r in top.itertuples()
        )
        print(f"  {league}: {line}")


if __name__ == "__main__":
    main()
