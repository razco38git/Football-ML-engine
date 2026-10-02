"""Fit the cross-league correction from scored European ties.

Reads ``european_predictions.csv`` -- the model's own output on 822 UEFA
matches -- and fits one offset per league to what it got wrong. Writes
``data/processed/league_adjustment.json``, which the API applies to any pairing
whose two clubs come from different leagues.

Run after ``pipelines.validate_european``, and again whenever the model is
retrained: the offsets describe *this* model's residual error, so a new model
needs new ones.

    python -m pipelines.fit_league_adjustment
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml import registry  # noqa: E402
from footballml.data import PROCESSED_DIR  # noqa: E402
from footballml.league_adjust import REFERENCE, fit, save  # noqa: E402

PREDICTIONS = PROCESSED_DIR / "european_predictions.csv"

logger = logging.getLogger("fit_league_adjustment")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument(
        "--dry-run", action="store_true", help="Fit and report without writing."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not PREDICTIONS.exists():
        print(
            f"No {PREDICTIONS}.\n"
            "Run `python -m pipelines.fetch_european` then "
            "`python -m pipelines.validate_european` first."
        )
        return 1

    fitted = fit(pd.read_csv(PREDICTIONS), reference=args.reference)
    # Stamped with the model it was fitted against: these offsets describe that
    # model's residual error, so a retrain makes them stale and the API says so.
    try:
        fitted["model_version"] = registry.load()[1].version
    except Exception as exc:  # noqa: BLE001 - no artifact yet is survivable
        logger.warning("Could not record the model version (%s)", exc)
    offsets = fitted["offsets"]
    holdout = fitted["holdout"]

    print(f"\n=== League offsets, fitted on {fitted['n_matches']} cross-league ties ===")
    print("Log-scale shift applied to the goal rates; the reference is pinned at zero.")
    print(f"{'league':8s} {'offset':>8s}  {'vs ' + str(args.reference):>14s}")
    for league, value in sorted(offsets.items(), key=lambda kv: -kv[1]):
        print(f"{league:8s} {value:+8.3f}  {'reference' if value == 0 else f'{value:+.3f}':>14s}")

    if holdout:
        print("\n=== Leave-one-season-out: fitted without a season, scored on it ===")
        print(f"  uncorrected RPS  {holdout['rps_uncorrected']:.5f}")
        print(f"  corrected RPS    {holdout['rps_corrected']:.5f}")
        print(f"  improvement      {holdout['improvement']:+.5f}  on {holdout['n']} matches")
        if holdout["improvement"] <= 0:
            print("\n  The correction does not help out of sample. Do not apply it.")
            if not args.dry_run:
                print("  Writing it anyway would be dishonest; nothing written.")
            return 1
    else:
        print("\nToo few matches for a held-out check -- treat the offsets as unverified.")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    path = save(fitted)
    print(f"\nWrote {path}")
    print("The API applies this to any cross-league pairing; restart it to pick it up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
