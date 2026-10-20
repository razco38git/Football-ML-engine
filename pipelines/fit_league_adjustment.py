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
from pipelines.validate_european import MODEL_VERSION_COLUMN  # noqa: E402

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

    predictions = pd.read_csv(PREDICTIONS)
    # Stamped with the model it was fitted against: these offsets describe that
    # model's residual error, so a retrain makes them stale and the API says so.
    try:
        serving = registry.load()[1].version
    except Exception as exc:  # noqa: BLE001 - reported, not guessed at
        # Not survivable any more, and `save` enforces it. An unstamped file
        # reads as "no version claimed" to the API, which used to skip the
        # compatibility check entirely -- so it would be applied to every later
        # model in silence. Better to leave the previous correction in place.
        logger.error(
            "Could not record the model version (%s). Refusing to save an "
            "adjustment that cannot be checked against the model serving it; "
            "the previous correction is unchanged.", exc,
        )
        return 1

    # `validate_european` is optional in the weekly, so when it fails the job
    # continues to here with the *previous* model's predictions still on disk.
    # Refitting those and stamping them `serving` is the one outcome the stamp
    # exists to prevent: the API would compare equal, report the correction
    # current, and apply offsets measured against a model that no longer exists.
    #
    # A missing column is refused too, not waved through. A file written before
    # the column existed cannot be told apart from one written by another model,
    # and "cannot be told apart" has to mean no rather than yes here -- the cost
    # of refusing is one week without a refreshed correction, which `load`
    # already degrades to safely.
    # Every row, not every row that happens to carry a stamp. Dropping the
    # blanks first would accept a file half of whose rows came from an unknown
    # model, which is the case being guarded against rather than an exception
    # to it. Nothing writes such a file today -- `validate_european` stamps the
    # whole frame -- but a later change to score incrementally would make it
    # the normal shape, and this check would have waved it through.
    stamps = predictions.get(MODEL_VERSION_COLUMN)
    unstamped = len(predictions) if stamps is None else int(stamps.isna().sum())
    recorded = sorted(set(stamps.dropna())) if stamps is not None else []
    if unstamped or recorded != [serving]:
        logger.error(
            "%s was scored by %s, but the model being served is %s. Refusing to "
            "fit a correction on another model's residuals and stamp it as this "
            "one's; the previous correction is unchanged. Re-run "
            "`python -m pipelines.validate_european` first.",
            PREDICTIONS.name,
            ", ".join(recorded) + (f" (+{unstamped} row(s) with no stamp)" if unstamped else "")
            if recorded or unstamped else "no recorded model",
            serving,
        )
        return 1

    fitted = fit(predictions, reference=args.reference)
    fitted["model_version"] = serving
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
