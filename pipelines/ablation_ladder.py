"""Ablation ladder: what each feature family is actually worth.

Runs the production walk-forward backtest once per rung, adding one feature
family at a time, and reports the marginal RPS with a paired bootstrap interval
around every step.

**Nothing changes between rungs except which columns the model may see.** Same
fixtures, same walk-forward, same Poisson HistGradientBoosting settings, same
Dixon-Coles, same temperature policy, same ``random_state``, same baseline and
market benchmark. That is the point: any difference is attributable to the
features, because nothing else differs.

**Hyperparameters are fixed to production, deliberately.** They were tuned for
the full 235-feature model, so a three-feature rung is being run with a learning
budget chosen for a much larger one. The alternative -- retuning each rung --
measures something different and less useful: "the best model obtainable from
this family set" rather than "what this family adds to the model we ship". The
second is the question, so the budget stays fixed, and the first rungs should be
read as a floor rather than as the best those features could do.

Run::

    python -m pipelines.ablation_ladder
    python -m pipelines.ablation_ladder --quick      # 1920 onward, for a smoke test
    python -m pipelines.ablation_ladder --no-leave-one-out

Outputs land in ``experiments/ablation_ladder/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import (  # noqa: E402
    PROCESSED_DIR,
    PROJECT_ROOT,
    load_raw_odds,
    load_team_match_history,
    load_team_strength,
)
from footballml.experiments.ablation import (  # noqa: E402
    FAMILY_PATTERNS,
    ID_COLUMNS,
    LADDER,
    LEAVE_ONE_OUT,
    classify,
    rung_columns,
)
from footballml.features.build import build_match_features  # noqa: E402
from footballml.models.evaluate import (  # noqa: E402
    accuracy,
    brier_score,
    log_loss,
    odds_implied_probs,
    paired_bootstrap,
    ranked_probability_score,
    rps_per_match,
)
from footballml.models.match_model import DEFAULT_GBM_PARAMS, feature_columns  # noqa: E402
from pipelines.backtest import run_backtest  # noqa: E402

logger = logging.getLogger("ablation")

OUT_DIR = PROJECT_ROOT / "experiments" / "ablation_ladder"

#: Fixture key for pairing predictions across rungs. A paired bootstrap is only
#: valid on identical fixtures in identical order, so rows are joined on this
#: and the alignment is asserted rather than assumed.
FIXTURE_KEY = ["League", "Season", "Date", "HomeTeam", "AwayTeam"]

PROB_COLUMNS = ["prob_H", "prob_D", "prob_A"]


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def load_features() -> pd.DataFrame:
    """The exact production feature table, built the way `pipelines.train` builds it."""
    path = PROCESSED_DIR / "team_match_history_all.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run `python -m pipelines.build_dataset` first"
        )
    tmh = load_team_match_history(path)
    return build_match_features(tmh, strength=load_team_strength())


def pooled_metrics(preds: pd.DataFrame) -> dict[str, float]:
    """Metrics over every walk-forward prediction pooled.

    Pooling the predictions rather than averaging per-season scores matters for
    the seasons with fewer matches: every metric here is a mean over fixtures, so
    a plain mean of season scores would weight a 250-match season like an
    1,826-match one.
    """
    probs = preds[PROB_COLUMNS].to_numpy()
    y = preds["FTR"].to_numpy()
    return {
        "rps": ranked_probability_score(y, probs),
        "log_loss": log_loss(y, probs),
        "brier": brier_score(y, probs),
        "accuracy": accuracy(y, probs),
        "n": int(len(preds)),
    }


def pooled_rps_uncalibrated(per_season: pd.DataFrame) -> float:
    """Pre-calibration pooled RPS, from the per-season raw scores.

    `run_backtest` reports `rps_raw` per season but returns only the calibrated
    probabilities, and RPS is a mean over fixtures -- so weighting the per-season
    raw scores by their match counts reconstructs the pooled value exactly, with
    no second pass over the model.
    """
    weights = per_season["n"].to_numpy(dtype="float64")
    return float((per_season["rps_raw"].to_numpy() * weights).sum() / weights.sum())


def market_rps(preds: pd.DataFrame) -> tuple[float, int]:
    """Benchmark RPS from de-vigged closing odds, on the subset that has them."""
    odds_cols = ["B365H", "B365D", "B365A"]
    if not all(c in preds.columns for c in odds_cols):
        return float("nan"), 0
    sub = preds.dropna(subset=odds_cols)
    if sub.empty:
        return float("nan"), 0
    return ranked_probability_score(sub["FTR"], odds_implied_probs(sub, tuple(odds_cols))), len(sub)


def run_rung(
    name: str,
    families: tuple[str, ...],
    features: pd.DataFrame,
    all_feature_cols: list[str],
    odds: pd.DataFrame,
    start_season: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """One rung: restrict the columns, run the production walk-forward."""
    cols = rung_columns(all_feature_cols, families)
    frame = features[[*ID_COLUMNS, *cols]].copy()

    # Guard against the one mistake that would invalidate everything: a rung
    # seeing a column it should not.
    visible = set(feature_columns(frame))
    if visible != set(cols):
        raise AssertionError(
            f"rung {name!r} exposes {sorted(visible - set(cols))} and hides "
            f"{sorted(set(cols) - visible)}"
        )

    started = time.time()
    per_season, preds = run_backtest(frame, start_season=start_season, odds=odds)
    elapsed = time.time() - started

    metrics = pooled_metrics(preds)
    metrics["rps_uncalibrated"] = pooled_rps_uncalibrated(per_season)
    mkt, n_mkt = market_rps(preds)
    metrics.update(
        rung=name,
        families=",".join(families) if families else "ALL",
        n_features=len(cols),
        rps_market=mkt,
        n_with_odds=n_mkt,
        rps_base_rate=float(
            (per_season["rps_base_rate"] * per_season["n"]).sum() / per_season["n"].sum()
        ),
        seconds=round(elapsed, 1),
    )
    per_season = per_season.assign(rung=name)
    logger.info(
        "%-24s %3d features  RPS %.5f (raw %.5f)  acc %.4f  %.0fs",
        name, len(cols), metrics["rps"], metrics["rps_uncalibrated"],
        metrics["accuracy"], elapsed,
    )
    return metrics, preds, per_season


def align(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort two prediction frames onto the same fixtures, or fail.

    The experiment's validity rests on every rung predicting exactly the same
    matches. Only the columns change between rungs, so this should always hold --
    which is precisely why it is worth asserting rather than trusting.
    """
    a = left.sort_values(FIXTURE_KEY).reset_index(drop=True)
    b = right.sort_values(FIXTURE_KEY).reset_index(drop=True)
    if len(a) != len(b):
        raise AssertionError(f"rungs predicted different match counts: {len(a)} vs {len(b)}")
    mismatch = (a[FIXTURE_KEY] != b[FIXTURE_KEY]).any(axis=1)
    if mismatch.any():
        raise AssertionError(f"{int(mismatch.sum())} fixtures differ between rungs")
    return a, b


def compare(
    name: str, kind: str, baseline: pd.DataFrame, variant: pd.DataFrame, n_boot: int
) -> dict:
    """Paired bootstrap of the RPS difference between two rungs."""
    a, b = align(baseline, variant)
    base_scores = rps_per_match(a["FTR"], a[PROB_COLUMNS].to_numpy())
    var_scores = rps_per_match(b["FTR"], b[PROB_COLUMNS].to_numpy())
    result = paired_bootstrap(base_scores, var_scores, n_boot=n_boot)
    result.update(comparison=name, kind=kind)
    logger.info(
        "  [%-12s] %-42s delta %+.5f [%+.5f, %+.5f] %s",
        kind, name, result["delta"], result["lo"], result["hi"],
        "SIGNIFICANT" if result["excludes_zero"] else "not significant",
    )
    return result


def plot(results: pd.DataFrame, out_dir: Path) -> list[Path]:
    """RPS across the ladder, and the marginal contribution of each family."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots (pip install -e '.[experiments]')")
        return []

    ladder = results[results["kind"] == "ladder"]
    written = []

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ladder))
    ax.plot(x, ladder["rps"], marker="o", color="#1f77b4", lw=2, zorder=3, label="model")
    ax.axhline(ladder["rps_market"].iloc[0], ls="--", color="#d62728",
               label=f"bookmakers {ladder['rps_market'].iloc[0]:.4f}")
    ax.axhline(ladder["rps_base_rate"].iloc[0], ls=":", color="#7f7f7f",
               label=f"base rate {ladder['rps_base_rate'].iloc[0]:.4f}")
    for xi, (_, row) in zip(x, ladder.iterrows(), strict=True):
        ax.annotate(f"{row['rps']:.4f}", (xi, row["rps"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("base+", "+") for r in ladder["rung"]], rotation=20, ha="right")
    ax.set_ylabel("RPS (lower is better)")
    ax.set_title(f"Ablation ladder, walk-forward over {int(ladder['n'].iloc[0]):,} matches")
    ax.invert_yaxis()  # lower is better, so better reads upward
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "rps_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    steps = results[results["kind"] == "marginal"].copy()
    if not steps.empty:
        # Negative delta = improvement. Plot the improvement so bars grow upward.
        gain = -steps["delta"].to_numpy()
        err = np.vstack([gain - (-steps["hi"].to_numpy()), -steps["lo"].to_numpy() - gain])
        colors = ["#2ca02c" if e else "#c7c7c7" for e in steps["excludes_zero"]]
        # The family each step adds: "a+b -> a+b+c" becomes "+c".
        labels = [c.split("->")[-1].strip().split("+")[-1] for c in steps["comparison"]]
        labels = [f"+{lab}" if lab != "full" else "+nothing\n(full)" for lab in labels]
        x = np.arange(len(steps))

        # Two panels, because Elo is an order of magnitude larger than anything
        # else and on one scale it flattens the other families into the axis --
        # which is the finding, but it must still be possible to read them.
        fig, (ax, zoom) = plt.subplots(1, 2, figsize=(12, 5))
        for axis, subset in ((ax, slice(None)), (zoom, slice(1, None))):
            xi, gi = x[subset], gain[subset]
            axis.bar(xi, gi, yerr=err[:, subset], capsize=4,
                     color=colors[subset.start or 0:])
            axis.axhline(0, color="black", lw=0.8)
            axis.set_xticks(xi)
            axis.set_xticklabels(labels[subset.start or 0:], fontsize=9)
            axis.grid(axis="y", alpha=0.3)
            for xx, gg in zip(xi, gi, strict=True):
                axis.annotate(f"{gg:+.5f}", (xx, gg), textcoords="offset points",
                              xytext=(0, 10 if gg >= 0 else -16), ha="center", fontsize=8)
        ax.set_ylabel("RPS improvement (higher is better)")
        ax.set_title("Marginal contribution of each family")
        zoom.set_title("Same, excluding Elo")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.text(0.5, 0.02, "95% paired-bootstrap interval; grey = interval includes zero",
                 ha="center", fontsize=10)
        p = out_dir / "marginal_contribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=1516,
                        help="First season to test on (production default: 1516).")
    parser.add_argument("--quick", action="store_true",
                        help="Start at 2324 instead: a fast smoke test, NOT a result.")
    parser.add_argument("--n-boot", type=int, default=10_000, help="Bootstrap resamples.")
    parser.add_argument("--no-leave-one-out", action="store_true",
                        help="Skip the secondary leave-one-family-out pass.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("footballml").setLevel(logging.WARNING)

    start_season = 2324 if args.quick else args.start_season
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building the production feature table ...")
    features = load_features()
    all_cols = feature_columns(features)
    grouped = classify(all_cols)
    logger.info(
        "%d features: %s",
        len(all_cols), "  ".join(f"{f}={len(c)}" for f, c in grouped.items()),
    )
    odds = load_raw_odds()

    rows, per_season_all, preds_by_rung = [], [], {}
    logger.info("\n=== ladder ===")
    for name, families in LADDER:
        metrics, preds, per_season = run_rung(
            name, families, features, all_cols, odds, start_season
        )
        metrics["kind"] = "ladder"
        rows.append(metrics)
        per_season_all.append(per_season)
        preds_by_rung[name] = preds

    if not args.no_leave_one_out:
        logger.info("\n=== leave-one-family-out from full ===")
        for family in LEAVE_ONE_OUT:
            keep = tuple(f for f in FAMILY_PATTERNS if f != family)
            name = f"full-minus-{family}"
            metrics, preds, per_season = run_rung(
                name, keep, features, all_cols, odds, start_season
            )
            metrics["kind"] = "leave_one_out"
            rows.append(metrics)
            per_season_all.append(per_season)
            preds_by_rung[name] = preds

    results = pd.DataFrame(rows)
    ladder_names = [n for n, _ in LADDER]

    # Differences against the previous rung and against base, for every metric
    # the brief asks for.
    ladder_mask = results["rung"].isin(ladder_names)
    base_row = results[results["rung"] == "base"].iloc[0]
    for metric in ("rps", "log_loss", "brier", "accuracy"):
        prev = results.loc[ladder_mask, metric].shift(1)
        results.loc[ladder_mask, f"d_{metric}_vs_prev"] = results.loc[ladder_mask, metric] - prev
        results.loc[ladder_mask, f"d_{metric}_vs_base"] = (
            results.loc[ladder_mask, metric] - base_row[metric]
        )

    logger.info("\n=== paired bootstraps ===")
    boots = []
    for prev, cur in zip(ladder_names, ladder_names[1:], strict=False):
        boots.append(
            compare(f"{prev} -> {cur}", "marginal", preds_by_rung[prev],
                    preds_by_rung[cur], args.n_boot)
        )
    for cur in ladder_names[1:]:
        boots.append(
            compare(f"base -> {cur}", "vs_base", preds_by_rung["base"],
                    preds_by_rung[cur], args.n_boot)
        )
    if not args.no_leave_one_out:
        for family in LEAVE_ONE_OUT:
            boots.append(
                compare(f"full vs full-minus-{family}", "leave_one_out",
                        preds_by_rung["full"], preds_by_rung[f"full-minus-{family}"],
                        args.n_boot)
            )
    bootstrap = pd.DataFrame(boots)

    # Share of the total base->full gain contributed by each step, filled only
    # for the marginal rows -- it is meaningless for a leave-one-out comparison,
    # whose deltas do not sum to anything.
    total_gain = base_row["rps"] - results[results["rung"] == "full"].iloc[0]["rps"]
    is_marginal = bootstrap["kind"] == "marginal"
    bootstrap["share_of_total_gain"] = np.nan
    if total_gain:
        bootstrap.loc[is_marginal, "share_of_total_gain"] = (
            -bootstrap.loc[is_marginal, "delta"] / total_gain
        )

    plots = plot(
        pd.concat(
            [
                results[ladder_mask].assign(kind="ladder"),
                bootstrap[is_marginal].assign(kind="marginal"),
            ],
            ignore_index=True,
        ),
        args.out_dir,
    )

    results.to_csv(args.out_dir / "results.csv", index=False)
    bootstrap.to_csv(args.out_dir / "bootstrap_results.csv", index=False)

    # Every rung's per-match probabilities, so any subgroup question -- does a
    # family earn its place in recent seasons? in one league? -- can be answered
    # with a paired bootstrap later instead of by re-running 45 minutes of
    # walk-forward. Gzipped: ten rungs x 20,013 fixtures is large for a repo.
    pd.concat(
        [
            preds[[*FIXTURE_KEY, "FTR", *PROB_COLUMNS]].assign(rung=name)
            for name, preds in preds_by_rung.items()
        ],
        ignore_index=True,
    ).to_csv(args.out_dir / "predictions.csv.gz", index=False, compression="gzip")
    pd.concat(per_season_all, ignore_index=True).to_csv(
        args.out_dir / "per_season.csv", index=False
    )

    config = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "start_season": start_season,
        "quick_mode": bool(args.quick),
        "n_bootstrap": args.n_boot,
        "n_matches": int(results["n"].max()),
        "n_features_total": len(all_cols),
        "gbm_params": DEFAULT_GBM_PARAMS,
        "hyperparameters_retuned_per_rung": False,
        "dixon_coles": True,
        "calibration": "temperature, fitted on accumulated out-of-sample walk-forward predictions",
        "families": {f: cols for f, cols in grouped.items()},
        "ladder": {name: (list(fams) or "ALL") for name, fams in LADDER},
        "leave_one_out": list(LEAVE_ONE_OUT),
        "total_gain_base_to_full_rps": float(total_gain),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    logger.info("\nWrote results.csv, bootstrap_results.csv, per_season.csv, config.json")
    for p in plots:
        logger.info("Wrote %s", p.name)

    show = [
        "rung", "n_features", "rps", "rps_uncalibrated", "log_loss", "brier", "accuracy",
        "d_rps_vs_prev", "d_rps_vs_base",
    ]
    logger.info("\n%s", results[show].to_string(index=False))


if __name__ == "__main__":
    main()
