"""Predict fixtures and show why, with actual results when they exist.

Trains on everything before a cutoff date, then predicts matches on or after it.
Because the dataset runs to May 2025, predicting a past date shows the
prediction *and* what actually happened -- which is the accuracy tracker in
miniature, and the quickest way to see whether the model is sane.

Run with::

    python -m pipelines.predict --from 2025-05-01
    python -m pipelines.predict --from 2025-04-01 --league E0 --limit 8 --explain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import PROCESSED_DIR, load_team_match_history  # noqa: E402
from footballml.features.build import build_match_features  # noqa: E402
from footballml.labels import humanise  # noqa: E402
from footballml.models.evaluate import evaluate  # noqa: E402
from footballml.models.match_model import MatchPredictor, feature_columns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="cutoff", default="2025-05-01")
    parser.add_argument("--league", nargs="+", help="Division codes, e.g. E0 SP1.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--explain", action="store_true", help="Show SHAP drivers.")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff)

    path = PROCESSED_DIR / "team_match_history_all.csv"
    tmh = load_team_match_history(path if path.exists() else None)
    features = build_match_features(tmh)

    train = features[features["Date"] < cutoff]
    upcoming = features[features["Date"] >= cutoff]
    if args.league:
        upcoming = upcoming[upcoming["League"].isin(args.league)]
    if upcoming.empty:
        raise SystemExit(f"No fixtures on or after {cutoff.date()}")

    cols = feature_columns(features)
    print(f"Training on {len(train):,} matches before {cutoff.date()} ...")
    model = MatchPredictor().fit(train[cols], train["FTHG"], train["FTAG"])
    print(f"Fitted. Dixon-Coles rho = {model.rho_:.4f}\n")

    shown = upcoming.head(args.limit)
    preds = model.predict_frame(shown[cols])
    table = pd.concat([shown.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    print(f"=== {len(shown)} fixtures from {cutoff.date()} ===\n")
    header = f"{'match':<38} {'pred':>5} {'xG':>11} {'H/D/A %':>18} {'score':>6}  actual"
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        match = f"{r.HomeTeam} v {r.AwayTeam}"[:37]
        xg = f"{r.expected_goals_home:.2f}-{r.expected_goals_away:.2f}"
        hda = (
            f"{r.prob_home_win * 100:4.1f}/{r.prob_draw * 100:4.1f}/{r.prob_away_win * 100:4.1f}"
        )
        modal = f"{int(r.modal_score_home)}-{int(r.modal_score_away)}"
        actual = f"{int(r.FTHG)}-{int(r.FTAG)} ({r.FTR})" if pd.notna(r.FTHG) else "-"
        hit = "OK " if pd.notna(r.FTR) and r.predicted_outcome == r.FTR else "   "
        print(f"{match:<38} {r.predicted_outcome:>5} {xg:>11} {hda:>18} {modal:>6}  {hit}{actual}")

    settled = table[table["FTR"].notna()]
    if not settled.empty:
        probs = settled[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy()
        metrics = evaluate(settled["FTR"], probs)
        print(
            f"\nOn these {metrics['n']} settled fixtures: "
            f"accuracy {metrics['accuracy']:.1%}, RPS {metrics['rps']:.4f}"
        )

    if args.explain:
        print("\n=== Why? (top drivers of each side's expected goals) ===")
        drivers = model.explain(shown[cols], top_n=4)
        for entry, (_, r) in zip(drivers, shown.iterrows(), strict=True):
            print(f"\n{r.HomeTeam} v {r.AwayTeam}")
            for side in ("home", "away"):
                team = r.HomeTeam if side == "home" else r.AwayTeam
                print(f"  {team} expected goals:")
                for name, value in entry[side]:
                    arrow = "up  " if value > 0 else "down"
                    print(f"    {arrow} {value:+.3f}  {humanise(name)}")


if __name__ == "__main__":
    main()
