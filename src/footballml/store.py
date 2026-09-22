"""Append-only prediction storage and settlement.

This is what turns a model into a track record. Predictions are written once,
stamped with the model version and the moment they were made, and never
rewritten. Results are attached later by :func:`settle`.

The append-only rule is the entire point. If predictions could be regenerated
after the fact, "how accurate are we?" would quietly become "how accurate is
today's model on matches it has already seen" -- which is not a forecast, and
would make the published accuracy meaningless.

CSV rather than Parquet because ``pyarrow`` is not installed, and rather than
Postgres because the database layer is not built yet. The interface is
deliberately narrow so swapping the backend touches only this module.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from footballml.data import PROCESSED_DIR

logger = logging.getLogger(__name__)

PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"

#: Identifies one forecast. A model version may predict a fixture only once.
KEY = ("model_version", "League", "Date", "HomeTeam", "AwayTeam")

#: Result columns, null until the match is played and settled.
ACTUAL_COLUMNS = ("actual_home_goals", "actual_away_goals", "actual_result")


def append(
    predictions: pd.DataFrame, model_version: str, path: Path | None = None
) -> int:
    """Store new predictions, ignoring any already recorded for this version.

    Args:
        predictions: Rows with the identifying columns plus model outputs.
        model_version: Version that produced them.
        path: Override the store location (used by tests).

    Returns:
        Number of genuinely new rows written.
    """
    path = path or PREDICTIONS_PATH
    incoming = predictions.copy()
    incoming["model_version"] = model_version
    incoming["predicted_at"] = datetime.now(UTC).isoformat()
    incoming["Date"] = pd.to_datetime(incoming["Date"]).dt.strftime("%Y-%m-%d")
    for col in ACTUAL_COLUMNS:
        # Object dtype, not the float64 an all-NA column would default to:
        # settle() writes result letters into these later.
        incoming[col] = pd.Series([None] * len(incoming), dtype="object")

    existing = load(path)
    if not existing.empty:
        # load() parses Date back to datetime, while incoming holds strings.
        # Align both before the dedupe merge or it raises on dtype mismatch.
        seen = existing[list(KEY)].copy()
        seen["Date"] = pd.to_datetime(seen["Date"]).dt.strftime("%Y-%m-%d")
        merged = incoming.merge(seen.assign(_seen=True), on=list(KEY), how="left")
        incoming = incoming[merged["_seen"].isna().to_numpy()]
        existing["Date"] = pd.to_datetime(existing["Date"]).dt.strftime("%Y-%m-%d")

    if incoming.empty:
        logger.info("No new predictions to store")
        return 0

    combined = (
        pd.concat([existing, incoming], ignore_index=True)
        if not existing.empty
        else incoming
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    logger.info("Stored %d new predictions (%d total)", len(incoming), len(combined))
    return len(incoming)


def load(path: Path | None = None) -> pd.DataFrame:
    """Read every stored prediction. Empty frame when nothing exists yet."""
    path = path or PREDICTIONS_PATH
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def settle(matches: pd.DataFrame, path: Path | None = None) -> int:
    """Attach actual results to stored predictions for matches now played.

    Only fills blanks -- an already-settled prediction is never touched, so a
    rerun cannot alter recorded history.

    Args:
        matches: Played matches with ``FTHG``, ``FTAG``, ``FTR``.
        path: Override the store location.

    Returns:
        Number of predictions newly settled.
    """
    path = path or PREDICTIONS_PATH
    stored = load(path)
    if stored.empty:
        return 0

    key = ["League", "Date", "HomeTeam", "AwayTeam"]
    results = matches[[*key, "FTHG", "FTAG", "FTR"]].copy()
    results["Date"] = pd.to_datetime(results["Date"])
    results = results.dropna(subset=["FTR"]).drop_duplicates(subset=key)

    merged = stored.merge(results, on=key, how="left", suffixes=("", "_new"))
    # Read back from CSV, an all-blank column arrives as float64, which rejects
    # the result letters written below. Widen to object first.
    merged["actual_result"] = merged["actual_result"].astype("object")

    unsettled = merged["actual_result"].isna() & merged["FTR"].notna()
    count = int(unsettled.sum())
    if count:
        merged.loc[unsettled, "actual_home_goals"] = merged.loc[unsettled, "FTHG"]
        merged.loc[unsettled, "actual_away_goals"] = merged.loc[unsettled, "FTAG"]
        merged.loc[unsettled, "actual_result"] = merged.loc[unsettled, "FTR"]
        merged.drop(columns=["FTHG", "FTAG", "FTR"]).to_csv(path, index=False)
        logger.info("Settled %d predictions", count)

    return count


def settled(path: Path | None = None) -> pd.DataFrame:
    """Stored predictions that have a known result, one row per fixture.

    `append` de-duplicates per model version, so a fixture predicted before
    kickoff, then re-predicted after a retrain while it was still unplayed, is
    stored twice. That is right for the store -- each version's own forecast is
    worth keeping -- but wrong for a published record: nine La Liga fixtures
    appeared as eighteen, doubling `n` and counting every hit and miss twice.

    The earliest prediction per fixture wins. It is the most conservative claim
    available: the forecast committed furthest ahead of kickoff, made with the
    least information.
    """
    stored = load(path)
    if stored.empty:
        return stored

    rows = stored[stored["actual_result"].notna()]
    if "predicted_at" in rows.columns:
        rows = rows.sort_values("predicted_at")
    key = ["League", "Date", "HomeTeam", "AwayTeam"]
    present = [c for c in key if c in rows.columns]
    if present:
        rows = rows.drop_duplicates(present, keep="first")
    return rows.reset_index(drop=True)
