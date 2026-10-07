"""Shapley attribution: one number per feature family, order-independent.

The ablation ladder credits a family for everything it adds *given the families
before it*; leave-one-out credits it only for what nothing else replaces. For
`elo` and `form`, both built from match results, the two disagree sharply and
neither is the family's value.

This runs **every** coalition of the four families -- 16 walk-forward backtests,
with `base` present throughout as the floor -- and averages each family's
marginal contribution over all orderings. The result is the unique attribution
that sums exactly to the total gain.

Identical to the ladder in every other respect: same fixtures, same production
hyperparameters, same Dixon-Coles, same calibration policy, same `random_state`.
Only the visible columns change.

Confidence intervals are exact rather than approximate. A Shapley value is a
linear combination of mean per-match differences, so it rewrites as the mean of
a per-match quantity, and the project's usual paired bootstrap applies to that
directly -- no extra fits, no resampling of models.

Run::

    python -m pipelines.shapley_families
    python -m pipelines.shapley_families --quick    # smoke test, not a result
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import PROJECT_ROOT, load_raw_odds  # noqa: E402
from footballml.experiments.ablation import ID_COLUMNS, rung_columns  # noqa: E402
from footballml.experiments.shapley import (  # noqa: E402
    check_efficiency,
    coalitions,
    per_match_contributions,
)
from footballml.models.evaluate import (  # noqa: E402
    paired_bootstrap,
    rps_per_match,
)
from footballml.models.match_model import DEFAULT_GBM_PARAMS, feature_columns  # noqa: E402
from pipelines.ablation_ladder import (  # noqa: E402
    FIXTURE_KEY,
    PROB_COLUMNS,
    _git_revision,
    load_features,
)
from pipelines.backtest import run_backtest  # noqa: E402

logger = logging.getLogger("shapley")

OUT_DIR = PROJECT_ROOT / "experiments" / "shapley_families"

#: The players. `base` is excluded deliberately -- it is league identity and
#: rest, carries no information about either side's quality, and is present in
#: every coalition as the floor. That makes v(empty) the `base` rung, a number
#: already measured, rather than a model with no features at all.
PLAYERS: tuple[str, ...] = ("elo", "form", "xg", "squad")

#: Always present, in every coalition.
FLOOR: tuple[str, ...] = ("base",)


def label(coalition: tuple[str, ...]) -> str:
    return "+".join(coalition) if coalition else "(base only)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=1516)
    parser.add_argument("--quick", action="store_true",
                        help="Start at 2425: a smoke test, NOT a result.")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("footballml").setLevel(logging.WARNING)
    start_season = 2425 if args.quick else args.start_season
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building the production feature table ...")
    features = load_features()
    all_cols = feature_columns(features)
    odds = load_raw_odds()

    every = coalitions(PLAYERS)
    logger.info("%d coalitions over %s, floor %s\n", len(every), list(PLAYERS), list(FLOOR))

    scores: dict[tuple[str, ...], np.ndarray] = {}
    reference: pd.DataFrame | None = None
    rows = []

    for i, coalition in enumerate(every, 1):
        cols = rung_columns(all_cols, (*FLOOR, *coalition))
        frame = features[[*ID_COLUMNS, *cols]].copy()
        started = time.time()
        _, preds = run_backtest(frame, start_season=start_season, odds=odds)
        elapsed = time.time() - started

        preds = preds.sort_values(FIXTURE_KEY).reset_index(drop=True)
        if reference is None:
            reference = preds[FIXTURE_KEY].copy()
        elif not preds[FIXTURE_KEY].equals(reference):
            raise AssertionError(
                f"coalition {label(coalition)} predicted different fixtures; "
                "the attribution requires identical, identically-ordered matches"
            )

        scores[coalition] = rps_per_match(preds["FTR"], preds[PROB_COLUMNS].to_numpy())
        rps = float(scores[coalition].mean())
        rows.append(
            {
                "coalition": label(coalition),
                "size": len(coalition),
                "n_features": len(cols),
                "rps": rps,
                "n": int(len(preds)),
                "seconds": round(elapsed, 1),
            }
        )
        logger.info(
            "  %2d/%d  %-26s %3d features  RPS %.5f  %.0fs",
            i, len(every), label(coalition), len(cols), rps, elapsed,
        )

    contributions = per_match_contributions(scores, PLAYERS)
    total_gain = check_efficiency(contributions, scores, PLAYERS)
    logger.info(
        "\nEfficiency holds: the four values sum to the total gain, %.5f", total_gain
    )

    logger.info("\n=== Shapley values (RPS gain over the base rung) ===")
    shapley_rows = []
    zero = np.zeros_like(contributions[PLAYERS[0]])
    for player in PLAYERS:
        boot = paired_bootstrap(zero, contributions[player], n_boot=args.n_boot)
        # paired_bootstrap reports variant - baseline; contributions are already
        # gains, so the sign needs no flipping here.
        shapley_rows.append(
            {
                "family": player,
                "shapley": boot["delta"],
                "lo": boot["lo"],
                "hi": boot["hi"],
                "excludes_zero": boot["excludes_zero"],
                "share_of_total": boot["delta"] / total_gain if total_gain else np.nan,
                "n": boot["n"],
            }
        )
        logger.info(
            "  %-6s %+.5f [%+.5f, %+.5f]  %5.1f%%  %s",
            player, boot["delta"], boot["lo"], boot["hi"],
            100 * shapley_rows[-1]["share_of_total"],
            "SIGNIFICANT" if boot["excludes_zero"] else "not significant",
        )

    results = pd.DataFrame(rows)
    shapley = pd.DataFrame(shapley_rows)
    results.to_csv(args.out_dir / "coalitions.csv", index=False)
    shapley.to_csv(args.out_dir / "shapley.csv", index=False)
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "git_revision": _git_revision(),
                "players": list(PLAYERS),
                "floor": list(FLOOR),
                "n_coalitions": len(every),
                "start_season": start_season,
                "quick_mode": bool(args.quick),
                "n_bootstrap": args.n_boot,
                "n_matches": int(results["n"].max()),
                "gbm_params": DEFAULT_GBM_PARAMS,
                "hyperparameters_retuned_per_coalition": False,
                "value_function": "gain = RPS(base) - RPS(S)",
                "total_gain": float(total_gain),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot(shapley, results, total_gain, args.out_dir)
    logger.info("\nWrote coalitions.csv, shapley.csv, config.json")
    logger.info("\n%s", results.to_string(index=False))


def plot(
    shapley: pd.DataFrame, results: pd.DataFrame, total_gain: float, out_dir: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot")
        return

    order = shapley.sort_values("shapley", ascending=False)
    fig, (ax, right) = plt.subplots(1, 2, figsize=(12, 5))

    gains = order["shapley"].to_numpy()
    err = np.vstack([gains - order["lo"].to_numpy(), order["hi"].to_numpy() - gains])
    colors = ["#2ca02c" if e else "#c7c7c7" for e in order["excludes_zero"]]
    y = np.arange(len(order))[::-1]
    ax.barh(y, gains, xerr=err, capsize=4, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(order["family"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Shapley value: RPS gain over the base rung")
    ax.set_title("Attribution averaged over all orderings")
    for yy, g, s in zip(y, gains, order["share_of_total"], strict=True):
        ax.annotate(f"{g:+.5f}  ({100 * s:.1f}%)", (g, yy),
                    textcoords="offset points", xytext=(8, 0), va="center", fontsize=9)
    ax.set_xlim(right=max(gains) * 1.45)
    ax.grid(axis="x", alpha=0.3)

    # Every coalition, so the interaction structure is visible rather than
    # summarised: how much each family adds depends on who is already there.
    #
    # Spread within each size band -- six coalitions of size 2 land within
    # 0.001 RPS of each other, so stacked on one x they and their labels
    # overlap into an unreadable smear.
    by_size = results.sort_values(["size", "rps"]).copy()
    by_size["x"] = 0.0
    for size, group in by_size.groupby("size"):
        count = len(group)
        offsets = np.linspace(-0.3, 0.3, count) if count > 1 else np.array([0.0])
        by_size.loc[group.index, "x"] = size + offsets
    right.scatter(by_size["x"], by_size["rps"], s=42, color="#1f77b4", zorder=3)
    for _, row in by_size.iterrows():
        right.annotate(row["coalition"].replace("(base only)", "base"),
                       (row["x"], row["rps"]), textcoords="offset points",
                       xytext=(0, 7), ha="center", fontsize=6.5, rotation=20)
    right.set_xticks(range(len(PLAYERS) + 1))
    right.set_xlabel("families present (beyond base)")
    right.set_ylabel("RPS (lower is better)")
    right.set_title(f"All {len(results)} coalitions")
    right.invert_yaxis()
    right.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.02,
             f"Values sum to the total gain of {total_gain:.5f}; "
             "95% paired-bootstrap intervals; grey = includes zero",
             ha="center", fontsize=10)
    fig.savefig(out_dir / "shapley_values.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote shapley_values.png")


if __name__ == "__main__":
    main()
