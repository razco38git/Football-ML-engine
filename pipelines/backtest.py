"""Walk-forward backtest for the match predictor.

Season by season: train on everything that came before, predict the season, move
on. This is the only honest way to evaluate a time-series model. Random k-fold
cross-validation would train on 2024 matches to predict 2018 ones and report a
score that cannot be reproduced in production.

Run with::

    python -m pipelines.backtest
    python -m pipelines.backtest --start-season 1819 --no-dixon-coles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import (  # noqa: E402
    ODDS_COLUMNS,
    PROCESSED_DIR,
    load_raw_odds,
    load_team_match_history,
    load_team_strength,
)
from footballml.features.build import AUTO, build_match_features  # noqa: E402
from footballml.models.calibration import (  # noqa: E402
    TemperatureCalibrator,
    expected_calibration_error,
)
from footballml.models.evaluate import (  # noqa: E402
    base_rate_probs,
    calibration_table,
    evaluate,
    odds_implied_probs,
)
from footballml.models.match_model import MatchPredictor, feature_columns  # noqa: E402

#: Minimum pooled out-of-sample predictions before temperature scaling is
#: trusted. Fitting on a single season (380 matches) produced temperatures
#: swinging between 0.89 and 1.82 that did not transfer to the next season,
#: leaving calibration marginally worse than doing nothing.
MIN_CALIBRATION_MATCHES = 700


def run_backtest(
    features: pd.DataFrame,
    start_season: int,
    use_dixon_coles: bool = True,
    odds: pd.DataFrame | None = None,
    calibrate: bool = True,
    train_from: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk forward one season at a time.

    Args:
        features: The built feature table.
        start_season: First season to predict; everything earlier is
            training-only.
        use_dixon_coles: Apply the low-score correction.
        odds: Closing prices, merged into the output and used for the market
            benchmark. Never a model input.
        calibrate: Temperature-scale using accumulated out-of-sample predictions.
        train_from: Oldest season the model may **train** on, inclusive. ``None``
            -- the default and what production uses -- means every earlier
            season. It never affects which seasons are *tested*, so two runs
            differing only in this predict identical fixtures and can be paired.
            Added for the training-window A/B in ``pipelines.training_window_ab``;
            see ``EXPERIMENTS.md``.

    Returns:
        ``(per_season_metrics, predictions)`` where predictions carries the
        model's probabilities alongside the actual result for every test match.
    """
    cols = feature_columns(features)
    seasons = sorted(features["Season"].unique())
    test_seasons = [s for s in seasons if s >= start_season]

    rows: list[dict] = []
    all_preds: list[pd.DataFrame] = []

    # The walk-forward itself generates genuine out-of-sample predictions each
    # iteration. Accumulating them gives the calibrator a growing, honest
    # training set for free -- no extra model fits, and far more stable than the
    # single most recent season.
    oos_probs: list[np.ndarray] = []
    oos_actual: list[np.ndarray] = []

    for season in test_seasons:
        train = features[features["Season"] < season]
        if train_from is not None:
            train = train[train["Season"] >= train_from]
        test = features[features["Season"] == season]
        if train.empty or test.empty:
            continue

        model = MatchPredictor(use_dixon_coles=use_dixon_coles).fit(
            train[cols], train["FTHG"], train["FTAG"]
        )

        raw_probs = model.predict_proba(test[cols])
        probs = raw_probs
        temperature = 1.0

        pooled_n = sum(len(p) for p in oos_probs)
        if calibrate and pooled_n >= MIN_CALIBRATION_MATCHES:
            calibrator = TemperatureCalibrator().fit(
                np.vstack(oos_probs), np.concatenate(oos_actual)
            )
            probs = calibrator.transform(raw_probs)
            temperature = calibrator.temperature_

        # Record *uncalibrated* predictions: the calibrator must always be fitted
        # on the model's raw output, never on already-scaled probabilities.
        oos_probs.append(raw_probs)
        oos_actual.append(np.asarray(test["FTR"], dtype=object))

        mu_h, mu_a = model.predict_goal_rates(test[cols])

        metrics = evaluate(
            test["FTR"], probs, test["FTHG"], test["FTAG"], mu_h, mu_a
        )
        metrics["season"] = season
        metrics["n_train"] = len(train)
        metrics["rho"] = model.rho_
        metrics["temperature"] = temperature
        metrics["ece"] = expected_calibration_error(test["FTR"], probs)
        metrics["ece_raw"] = expected_calibration_error(test["FTR"], raw_probs)
        metrics["rps_raw"] = evaluate(test["FTR"], raw_probs)["rps"]

        # Baseline: predict the training-set outcome frequencies every time.
        base = base_rate_probs(train["FTR"], len(test))
        metrics["rps_base_rate"] = evaluate(test["FTR"], base)["rps"]

        id_cols = [c for c in ("League", "Season", "Date", "HomeTeam", "AwayTeam") if c in test]
        pred_df = test[[*id_cols, "FTHG", "FTAG", "FTR"]].copy()
        pred_df[["prob_H", "prob_D", "prob_A"]] = probs
        pred_df["mu_home"] = mu_h
        pred_df["mu_away"] = mu_a

        if odds is not None:
            merged = pred_df.merge(odds, on=["Date", "HomeTeam", "AwayTeam"], how="left")
            has_odds = merged[list(ODDS_COLUMNS)].notna().all(axis=1)
            if has_odds.any():
                sub = merged[has_odds]
                market = odds_implied_probs(sub, ODDS_COLUMNS)
                metrics["rps_market"] = evaluate(sub["FTR"], market)["rps"]
                metrics["n_with_odds"] = int(has_odds.sum())
                # Our RPS on the same subset, so the comparison is like-for-like.
                own = sub[["prob_H", "prob_D", "prob_A"]].to_numpy()
                metrics["rps_on_odds_subset"] = evaluate(sub["FTR"], own)["rps"]
            pred_df = merged

        rows.append(metrics)
        all_preds.append(pred_df)

    return pd.DataFrame(rows), pd.concat(all_preds, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-season",
        type=int,
        default=1516,
        help="First season to test on; everything earlier is training-only.",
    )
    parser.add_argument("--no-dixon-coles", action="store_true", help="Plain independent Poisson.")
    parser.add_argument(
        "--no-calibration", action="store_true", help="Skip temperature scaling."
    )
    parser.add_argument(
        "--xg",
        action="store_true",
        help="Use the Understat-enriched history (run pipelines.ingest_xg first).",
    )
    parser.add_argument(
        "--all-leagues",
        action="store_true",
        help="Use the five-league dataset (run pipelines.build_dataset first).",
    )
    parser.add_argument(
        "--leagues", nargs="+", help="Restrict to these division codes, e.g. E0 SP1."
    )
    parser.add_argument(
        "--no-strength",
        action="store_true",
        help="Exclude squad-strength features, for an A/B against the baseline.",
    )
    parser.add_argument(
        "--no-european",
        action="store_true",
        help=(
            "Build Elo from domestic matches only, as it was before UEFA ties "
            "were fed in. The A/B for that change."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write every walk-forward prediction to data/processed/.",
    )
    args = parser.parse_args()

    # Was `PROCESSED_DIR / (... if args.all_leagues else None or "")`, which in
    # the else branch resolved to the directory itself. Harmless only because
    # every branch below reassigns `path`.
    path = None
    if args.all_leagues:
        path = PROCESSED_DIR / "team_match_history_all.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found -- run `python -m pipelines.build_dataset` first")
    elif args.xg:
        path = PROCESSED_DIR / "team_match_history_xg.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found -- run `python -m pipelines.ingest_xg` first")

    tmh = load_team_match_history(path)
    if args.leagues:
        tmh = tmh[tmh["League"].isin(args.leagues)]

    # Strength is joined from the *previous* season. It is computed over a whole
    # season, so a match seeing its own would be predicting October from May.
    strength = None if args.no_strength else load_team_strength()
    european = None if args.no_european else AUTO
    features = build_match_features(tmh, strength=strength, european=european)

    cols = feature_columns(features)
    note = ""
    if strength is not None and "home_strength_overall" in features.columns:
        covered = features["home_strength_overall"].notna()
        seasons = sorted(features.loc[covered, "Season"].astype(str).unique())
        note = f" | strength on {int(covered.sum())} matches, seasons {seasons}"
    print(f"features: {len(cols)} columns, {len(features)} matches{note}")
    odds = load_raw_odds()

    per_season, preds = run_backtest(
        features,
        start_season=args.start_season,
        use_dixon_coles=not args.no_dixon_coles,
        odds=odds if not odds.empty else None,
        calibrate=not args.no_calibration,
    )

    show = [
        c
        for c in [
            "season", "n", "rps", "rps_raw", "rps_market",
            "ece", "ece_raw", "temperature", "accuracy", "goals_mae",
        ]
        if c in per_season.columns
    ]
    print("\n=== Walk-forward by season ===")
    print(per_season[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Pooled across all test seasons ===")
    probs = preds[["prob_H", "prob_D", "prob_A"]].to_numpy()
    pooled = evaluate(
        preds["FTR"], probs, preds["FTHG"], preds["FTAG"], preds["mu_home"], preds["mu_away"]
    )
    for k, v in pooled.items():
        print(f"  {k:12s} {v:.4f}" if isinstance(v, float) else f"  {k:12s} {v}")

    if "rps_market" in per_season.columns:
        w = per_season["n"].to_numpy(dtype="float64")
        print(f"\n  {'model RPS':12s} {np.average(per_season['rps'], weights=w):.4f}")
        print(f"  {'market RPS':12s} {np.average(per_season['rps_market'], weights=w):.4f}")
        print(f"  {'base rate':12s} {np.average(per_season['rps_base_rate'], weights=w):.4f}")

    if "League" in preds.columns and len(preds["League"].unique()) > 1:
        print("\n=== By league ===")
        rows = []
        for league, grp in preds.groupby("League"):
            p = grp[["prob_H", "prob_D", "prob_A"]].to_numpy()
            entry = {"league": league, **evaluate(grp["FTR"], p)}
            has_odds = grp[list(ODDS_COLUMNS)].notna().all(axis=1) if "B365H" in grp else None
            if has_odds is not None and has_odds.any():
                sub = grp[has_odds]
                entry["rps_market"] = evaluate(
                    sub["FTR"], odds_implied_probs(sub, ODDS_COLUMNS)
                )["rps"]
            rows.append(entry)
        by_league = pd.DataFrame(rows)[
            [c for c in ["league", "n", "rps", "rps_market", "accuracy"] if c in rows[0]]
        ]
        print(by_league.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.save_predictions:
        # Every walk-forward prediction, each made by a model trained only on
        # earlier seasons. This is what the site's accuracy page reads as its
        # backtest history.
        out = PROCESSED_DIR / "backtest_predictions.csv"
        preds.to_csv(out, index=False)
        print(f"\nWrote {len(preds):,} walk-forward predictions to {out.name}")

    print("\n=== Calibration ===")
    table = calibration_table(preds["FTR"], probs)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
