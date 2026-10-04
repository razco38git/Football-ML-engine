"""Score historical European ties with the deployed model, and report honestly.

The What-If tab answers "Real Madrid against Manchester City" and has never
been checked. European competitions are the only place the five leagues
actually meet, so they are the only ground truth a cross-league prediction has.

**Nothing here changes the model.** European matches are fixtures to be
predicted, never history: they are not appended to the team-match history, so
they never enter a rolling window or the Elo walk. Otherwise this would measure
a different model from the one serving the site.

**One fixture batch per date.** ``build_upcoming_features`` returns NaN when a
team appears twice in a batch -- its placeholder rows roll their form over each
other -- and on any European matchday a club plays at most once. Grouping by
date is therefore both correct and the reason this runs a few hundred feature
builds rather than one.

**Two comparisons, and only one of them is fair.** The deployed model was
trained on every domestic match, so scoring a 2018 European tie with it is
partly in-sample. The like-for-like reference is therefore *the same model on
domestic matches over the same seasons* -- equally flattered, so the difference
between the two is the finding. The walk-forward number from
``backtest_predictions.csv`` is printed too, clearly labelled, because it is the
honest absolute figure even though it is not the right thing to subtract.

Run with::

    python -m pipelines.validate_european
    python -m pipelines.validate_european --competitions UCL --limit-dates 20
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml import registry  # noqa: E402
from footballml.data import PROCESSED_DIR, load_team_match_history, load_team_strength  # noqa: E402
from footballml.european import fit_league_strength, head_to_head  # noqa: E402
from footballml.features.build import (  # noqa: E402
    LEAGUE_CODES,
    build_match_features,
    build_upcoming_features,
)
from footballml.models.evaluate import (  # noqa: E402
    base_rate_probs,
    calibration_table,
    evaluate,
)

MATCHES_IN = PROCESSED_DIR / "european_matches.csv"
TMH = PROCESSED_DIR / "team_match_history_all.csv"
PREDICTIONS_OUT = PROCESSED_DIR / "european_predictions.csv"
BACKTEST = PROCESSED_DIR / "backtest_predictions.csv"

PROB_COLUMNS = ["prob_home_win", "prob_draw", "prob_away_win"]

#: Built features, cached between runs.
#:
#: A past European tie's features cannot change. They are built from the
#: domestic matches *before* it, and history does not move -- so rebuilding all
#: 438 matchdays every week to re-score them with a new model is 92 minutes
#: spent recomputing immutable rows. Only genuinely new ties need work, which
#: in European weeks is a handful of matchdays and otherwise none.
#:
#: **Pickle, not CSV, and that is not a style choice.** CSV loses the last bits
#: of a float64 -- about 2 ULP -- and `float_format="%.17g"` does not recover
#: them. The booster *bins* its inputs, so a value sitting on a split boundary
#: lands in a different bin and the prediction moves for real: round-tripping
#: this cache through CSV changed 151 of 822 predictions, two of them by enough
#: to flip the predicted outcome, and one expected-goals figure by 0.12. A cache
#: that silently rewrites 18% of its answers is a second implementation, not a
#: cache. No parquet engine is installed, so pickle it is; it round-trips
#: float64 exactly and this file is derived data that can always be rebuilt.
FEATURES_CACHE = PROCESSED_DIR / "european_features.pkl"

#: What the cache depends on besides the fixture itself. Squad strength is
#: rebuilt weekly and joined from the *previous* season, so a change there does
#: reach back into old rows -- the one input that can invalidate the cache.
#: Recorded as a hash of the file it came from.
FINGERPRINT_COLUMN = "_strength_fingerprint"

KEY = ["Date", "HomeTeam", "AwayTeam"]

logger = logging.getLogger("validate_european")


def strength_fingerprint(path: Path | None = None) -> str:
    """Hash of the squad strength that cached features actually depend on.

    **Completed seasons only.** ``build_players`` runs every week and rewrites
    the whole file, so hashing it whole would invalidate the cache every Monday
    and put the 92-minute rebuild back. But strength is joined from the
    *previous* season, so a past European tie depends only on seasons that are
    over -- the current season's ratings move weekly and reach none of them.

    A change in a completed season means the ratings themselves were rebuilt
    differently (a config change, a new source), and then every cached row is
    genuinely stale and the rebuild is the right answer.
    """
    path = path or (PROCESSED_DIR / "team_strength.csv")
    if not path.exists():
        return "absent"
    frame = pd.read_csv(path)
    if "Season" in frame.columns and not frame.empty:
        seasons = frame["Season"].astype(str)
        frame = frame[seasons < seasons.max()]
    payload = frame.sort_values(list(frame.columns)).to_csv(index=False).encode()
    return hashlib.md5(payload).hexdigest()  # noqa: S324 - a cache key, not security


def load_cache(fingerprint: str, path: Path | None = None) -> pd.DataFrame:
    """Cached features still valid for this squad-strength build, or empty."""
    path = path or FEATURES_CACHE
    if not path.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_pickle(path)
    except Exception as exc:  # noqa: BLE001 - a pickle from another pandas, say
        logger.warning("Could not read the feature cache (%s); rebuilding", exc)
        return pd.DataFrame()
    if cached.empty or FINGERPRINT_COLUMN not in cached.columns:
        return pd.DataFrame()
    stale = set(cached[FINGERPRINT_COLUMN].unique()) != {fingerprint}
    if stale:
        logger.info("Squad strength has changed since the cache was built; rebuilding all")
        return pd.DataFrame()
    cached["Date"] = pd.to_datetime(cached["Date"])
    return cached


def _probs(frame: pd.DataFrame) -> np.ndarray:
    return frame[PROB_COLUMNS].to_numpy(dtype=float)


#: How many matchdays to build before writing the cache again.
#:
#: The cache used to be written once, after the last of 438 matchdays. Anything
#: that stopped the run first -- a timeout, a laptop sleeping, Ctrl-C -- threw
#: away every minute of it, and at ~9 seconds a matchday that is an hour and a
#: half with nothing to show. One run was killed at matchday 200 and the next
#: started again from zero.
#:
#: 25 matchdays is ~4 minutes of work at risk, against a write of a file that
#: is 1.6 MB. The write is atomic (see `_checkpoint`), so a kill *during* one
#: cannot leave a half-written cache behind either.
CHECKPOINT_EVERY = 25


def _checkpoint(frames: list[pd.DataFrame], fingerprint: str | None) -> None:
    """Write what has been built so far, atomically.

    Via a temporary file and a replace, because the alternative is a cache
    truncated mid-write that still carries a valid fingerprint -- which the
    next run would load and trust. `load_cache` survives an unreadable file,
    but not a readable and wrong one.
    """
    if fingerprint is None or not frames:
        return
    partial = pd.concat(frames, ignore_index=True)
    partial["Date"] = pd.to_datetime(partial["Date"])
    tmp = FEATURES_CACHE.with_suffix(".pkl.tmp")
    partial.assign(**{FINGERPRINT_COLUMN: fingerprint}).to_pickle(tmp)
    tmp.replace(FEATURES_CACHE)


def build_features(
    european: pd.DataFrame,
    tmh: pd.DataFrame,
    strength: pd.DataFrame,
    cached: pd.DataFrame | None = None,
    fingerprint: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features for every European tie, one matchday at a time.

    Returns:
        ``(scored_frame, raw_features)``. The second is what goes in the cache:
        features alone, before the results are merged on, so a later run can
        re-score them against a new model without rebuilding anything.

    The expensive half: each call rebuilds the form windows and the Elo walk
    over the whole domestic history, so this is ~13 seconds per matchday.

    The fixture is stamped with the **home** side's division, because
    ``LEAGUE_CODES`` has no entry for a European competition and labelling the
    row ``UCL`` would leave ``league_code`` NaN -- a value no training row ever
    carried. :func:`predict_with` then re-runs the prediction with the away
    side's code instead, which costs nothing and shows whether that choice
    moved anything.
    """
    pending = european
    reused = pd.DataFrame()
    if cached is not None and not cached.empty:
        have = cached[KEY].astype(str).apply(tuple, axis=1)
        wanted = european[KEY].astype(str).apply(tuple, axis=1)
        reused = cached[have.isin(set(wanted))]
        pending = european[~wanted.isin(set(have))]
        if not reused.empty:
            logger.info("Reused features for %d match(es) from the cache", len(reused))

    out = [reused.drop(columns=[FINGERPRINT_COLUMN], errors="ignore")] if not reused.empty else []
    dates = pending["Date"].nunique()
    if dates:
        print(f"Building features for {dates} new matchday(s) (~13s each)...", flush=True)

    for i, (date, batch) in enumerate(pending.groupby("Date", sort=True), start=1):
        repeated = pd.concat([batch["HomeTeam"], batch["AwayTeam"]]).duplicated().any()
        if repeated:
            # Would come back NaN: the placeholder rows roll their form over
            # each other. Not seen in practice -- a club plays one European
            # match per matchday -- but silence here would be a lie.
            logger.warning("%s: a club appears twice; skipping %d match(es)", date, len(batch))
            continue

        fixtures = pd.DataFrame(
            {
                "League": batch["home_league"].to_numpy(),
                "Season": batch["Season"].astype(str).to_numpy(),
                "Date": pd.to_datetime(batch["Date"]).to_numpy(),
                "HomeTeam": batch["HomeTeam"].to_numpy(),
                "AwayTeam": batch["AwayTeam"].to_numpy(),
            }
        )
        built = build_upcoming_features(tmh, fixtures, strength=strength)
        if built.empty:
            logger.warning("%s: no features built for %d match(es)", date, len(batch))
            continue
        out.append(built)
        if i % CHECKPOINT_EVERY == 0:
            print(f"  ... {i}/{dates} matchdays", flush=True)
            _checkpoint(out, fingerprint)

    if not out:
        return pd.DataFrame(), pd.DataFrame()

    # Raw features, cacheable: these depend only on the domestic history before
    # each tie, which never changes, so they are worth keeping between runs.
    raw = pd.concat(out, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"])

    # Truth comes from the *whole* input, not just what was rebuilt -- taking it
    # from `pending` would silently drop every cached match.
    columns = [
        "League", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "home_league", "away_league",
    ]
    if "Round" in european.columns:
        columns.append("Round")
    truth = european[columns].copy()
    truth["Date"] = pd.to_datetime(truth["Date"])
    truth = truth.rename(columns={"League": "competition"})

    # Drop the fixture's own League/Season (the competition label is clearer)
    # and its empty target columns -- a placeholder has no result, and leaving
    # them would collide with the real scores on the merge and silently become
    # FTHG_x / FTHG_y.
    features = raw.drop(columns=["League", "Season", "FTHG", "FTAG", "FTR"], errors="ignore")
    joined = truth.merge(features, on=KEY, how="inner")
    if len(joined) < len(truth):
        logger.warning(
            "%d of %d matches had no features and were dropped",
            len(truth) - len(joined), len(truth),
        )
    return joined, raw


def predict_with(features: pd.DataFrame, model, league_code_from: str = "home") -> pd.DataFrame:
    """Predict, taking ``league_code`` from one side's domestic division.

    Everything else about the feature vector is identical between the two
    choices, so swapping this one column and re-predicting isolates exactly the
    variable in question -- and costs nothing, unlike rebuilding.
    """
    missing = [c for c in model.feature_names_ if c not in features.columns]
    if missing:
        raise SystemExit(f"Model expects columns the build did not produce: {missing[:5]}")

    frame = features.copy()
    column = "home_league" if league_code_from == "home" else "away_league"
    frame["league_code"] = frame[column].map(LEAGUE_CODES).astype("float64")

    predicted = model.predict_frame(frame[model.feature_names_])
    # Squad strength travels with the prediction, because the site reads these
    # rows back. `/matches?league=UCL` serves this file directly -- European
    # results are deliberately kept out of the match history, so the API cannot
    # rebuild the features for them -- and without these columns every
    # Champions League card under "Recent results" came back with no squad
    # strength at all, while the identical domestic card showed it.
    strength = [c for c in frame.columns if "_strength_" in c]
    keep = [
        c
        for c in [
            "competition", "Season", "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR", "Round", "home_league", "away_league",
            *strength,
        ]
        if c in frame.columns
    ]
    return pd.concat(
        [frame[keep].reset_index(drop=True), predicted.reset_index(drop=True)], axis=1
    )


def domestic_reference(
    tmh: pd.DataFrame, strength: pd.DataFrame, model, seasons: set[str]
) -> dict[str, float] | None:
    """The same model on domestic matches over the same seasons.

    Equally in-sample, so the gap against the European figure is meaningful
    even though neither number is an out-of-sample estimate.
    """
    features = build_match_features(tmh, strength=strength)
    features = features[
        features["FTR"].notna() & features["Season"].astype(str).isin(seasons)
    ]
    if features.empty:
        return None
    predicted = model.predict_frame(features[model.feature_names_])
    return evaluate(features["FTR"].to_numpy(), _probs(predicted))


def _report(label: str, metrics: dict[str, float]) -> None:
    print(
        f"  {label:38s} RPS {metrics['rps']:.4f}   log loss {metrics['log_loss']:.4f}"
        f"   Brier {metrics['brier']:.4f}   acc {metrics['accuracy']:.1%}   n={int(metrics['n'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competitions", nargs="+", default=None)
    parser.add_argument(
        "--limit-dates", type=int, default=None,
        help="Score only the first N matchdays. For timing the full run.",
    )
    parser.add_argument("--no-domestic-reference", action="store_true")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild every matchday's features instead of reusing the cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if not MATCHES_IN.exists():
        print(f"No {MATCHES_IN}. Run `python -m pipelines.fetch_european` first.")
        return 1

    european = pd.read_csv(MATCHES_IN)
    european["Date"] = pd.to_datetime(european["Date"])
    european["Season"] = european["Season"].astype(str)
    if args.competitions:
        european = european[european["League"].isin(args.competitions)]
    if args.limit_dates:
        keep = sorted(european["Date"].unique())[: args.limit_dates]
        european = european[european["Date"].isin(keep)]
    if european.empty:
        print("No matches to score after filtering.")
        return 1

    tmh = load_team_match_history(TMH)
    strength = load_team_strength()
    model, metadata = registry.load()
    print(f"Model {metadata.version}, trained through {metadata.trained_through}\n")

    fingerprint = strength_fingerprint()
    cached = pd.DataFrame() if args.no_cache else load_cache(fingerprint)
    features, raw = build_features(
        european, tmh, strength, cached=cached,
        fingerprint=None if args.no_cache else fingerprint,
    )
    if features.empty:
        print("No features built.")
        return 1

    if not args.no_cache:
        raw.assign(**{FINGERPRINT_COLUMN: fingerprint}).to_pickle(FEATURES_CACHE)

    scored = predict_with(features, model, "home")
    scored.to_csv(PREDICTIONS_OUT, index=False)

    actual = scored["FTR"].to_numpy()
    probs = _probs(scored)

    print(f"=== European ties, scored by the deployed model ({len(scored)} matches) ===")
    _report("all competitions", evaluate(actual, probs))
    for competition, block in scored.groupby("competition"):
        _report(f"{competition}", evaluate(block["FTR"].to_numpy(), _probs(block)))

    print("\n=== References ===")
    base = base_rate_probs(actual, len(actual))
    _report("base rate (same matches)", evaluate(actual, base))

    if not args.no_domestic_reference:
        reference = domestic_reference(tmh, strength, model, set(scored["Season"]))
        if reference:
            _report("SAME model, domestic, same seasons", reference)
            print("    ^ the fair comparison: both are in-sample, so the gap is the finding")

    if BACKTEST.exists():
        backtest = pd.read_csv(BACKTEST)
        cols = ["prob_H", "prob_D", "prob_A"]
        if set(cols).issubset(backtest.columns) and "FTR" in backtest:
            walk = evaluate(backtest["FTR"].to_numpy(), backtest[cols].to_numpy(dtype=float))
            _report("walk-forward domestic (out-of-sample)", walk)
            print("    ^ the honest absolute number, but not like-for-like with the above")

    print("\nNo market benchmark exists here: football-data.co.uk publishes no odds")
    print("for UEFA competitions, so the usual market RPS cannot be computed.")

    # Does the made-up league_code change the answer? Free to check: only that
    # one column differs, so the features are reused rather than rebuilt.
    swapped = predict_with(features, model, "away")
    print("\n=== Sensitivity: league_code taken from the away side instead ===")
    _report("away side's division", evaluate(swapped["FTR"].to_numpy(), _probs(swapped)))
    print("    ^ if this differs materially, the metric is an artefact of a made-up label")

    if "Round" in scored.columns and scored["Round"].notna().any():
        from footballml.ingest.european import classify_round

        kind = scored["Round"].map(classify_round)
        print("\n=== By round: the confounds, measured rather than assumed ===")
        labels = {
            "round_robin": "group / league phase (single leg)",
            "two_legged": "two-legged knockout ties",
            "neutral": "finals (neutral venue)",
            "unclassified": "UNCLASSIFIED -- check classify_round",
        }
        for key, label in labels.items():
            block = scored[kind == key]
            if len(block) >= 20:
                _report(label, evaluate(block["FTR"].to_numpy(), _probs(block)))
            elif len(block):
                print(f"  {label:38s} only {len(block)} matches, too few to report")
        print("    ^ a second leg is not an independent match: a side three up rests players,")
        print("      and a final is at a neutral ground where home advantage does not apply.")

    print("\n=== Calibration ===")
    print(calibration_table(actual, probs).to_string(index=False))

    print("\n=== League strength from European results (goals per match) ===")
    table = fit_league_strength(scored)
    if table.empty:
        print("  Not enough cross-league matches to fit.")
    else:
        print(table.round(3).to_string(index=False))
        print("\n=== The evidence under it: each league's cross-league record ===")
        print(head_to_head(scored).to_string())

    print(f"\nWrote {PREDICTIONS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
