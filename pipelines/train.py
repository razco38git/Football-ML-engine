"""Train the production model and persist it with its metadata.

Trains on every completed match, then records how the same configuration scored
on a held-out final season. Those metrics travel with the artifact so the API
can report honest performance figures rather than asking anyone to trust a
number from a README.

Run with::

    python -m pipelines.train
    python -m pipelines.train --leagues E0 SP1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml import registry  # noqa: E402
from footballml.data import (  # noqa: E402
    PROCESSED_DIR,
    load_team_match_history,
    load_team_strength,
)
from footballml.features.build import build_match_features  # noqa: E402
from footballml.ingest.matchhistory import LEAGUES  # noqa: E402
from footballml.models.evaluate import base_rate_probs, evaluate  # noqa: E402
from footballml.models.match_model import MatchPredictor, feature_columns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", choices=sorted(LEAGUES))
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip the held-out evaluation (faster, but the artifact carries no metrics).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("train")

    path = PROCESSED_DIR / "team_match_history_all.csv"
    if not path.exists():
        raise SystemExit("Run `python -m pipelines.build_dataset` first")

    tmh = load_team_match_history(path)
    if args.leagues:
        tmh = tmh[tmh["League"].isin(args.leagues)]

    # Squad strength from the previous season -- the same quantity every serving
    # path supplies. Train and serve must agree on it, or the artifact expects
    # columns the API cannot build.
    features = build_match_features(tmh, strength=load_team_strength())
    played = features[features["FTR"].notna()].copy()
    cols = feature_columns(features)
    leagues = sorted(played["League"].unique())

    if "home_strength_overall" in played.columns:
        covered = played["home_strength_overall"].notna()
        log.info(
            "Squad strength on %d of %d matches (%.0f%%)",
            int(covered.sum()), len(played), 100 * covered.mean(),
        )
    else:
        log.warning("No team_strength.csv -- training without squad strength")

    metrics: dict[str, float] = {}
    if not args.skip_holdout:
        # Hold out the most recent *complete* season. The season in progress has
        # too few matches to measure anything useful.
        seasons = sorted(played["Season"].astype(str).unique())
        holdout = seasons[-2] if len(seasons) > 2 else seasons[-1]
        train = played[played["Season"].astype(str) < holdout]
        test = played[played["Season"].astype(str) == holdout]

        log.info("Held-out evaluation on season %s (%d matches)", holdout, len(test))
        probe = MatchPredictor().fit(train[cols], train["FTHG"], train["FTAG"])
        probs = probe.predict_proba(test[cols])
        mu_h, mu_a = probe.predict_goal_rates(test[cols])
        metrics = {
            k: v
            for k, v in evaluate(
                test["FTR"], probs, test["FTHG"], test["FTAG"], mu_h, mu_a
            ).items()
            if isinstance(v, float)
        }
        metrics["rps_base_rate"] = evaluate(test["FTR"], base_rate_probs(train["FTR"], len(test)))[
            "rps"
        ]
        metrics["holdout_season"] = float(holdout)
        for name, value in metrics.items():
            log.info("  %-16s %.4f", name, value)

    log.info("Training production model on %d matches", len(played))
    model = MatchPredictor().fit(played[cols], played["FTHG"], played["FTAG"])

    trained_through = str(played["Date"].max().date())
    metadata = registry.ModelMetadata(
        version=registry.make_version(trained_through),
        trained_at=__import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        trained_through=trained_through,
        n_train=len(played),
        leagues=leagues,
        feature_names=cols,
        rho=model.rho_,
        metrics=metrics,
    )

    target = registry.save(model, metadata)
    log.info("Saved %s", target)
    print(f"\nModel version: {metadata.version}")
    print(f"  trained through {metadata.trained_through} on {metadata.n_train:,} matches")
    print(f"  {metadata.n_features} features, leagues {', '.join(metadata.leagues)}")
    if metrics:
        print(f"  held-out RPS {metrics['rps']:.4f} (base rate {metrics['rps_base_rate']:.4f})")


if __name__ == "__main__":
    main()
