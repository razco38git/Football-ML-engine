"""Training-window A/B: is pre-xG history worth training on?

The ablation ladder found that xG *hurts* in test seasons 1516-1920 and *helps*
from 2021 on, both significant, and the cause is data availability rather than
football: Understat begins in 1415, so the early walk-forward trains on seasons
where 94 of the 235 features are entirely missing.

That raises an obvious question the ladder cannot answer. The model trains from
1011. Four of those seasons carry no xG at all. **Is that history helping,
because more rows is more rows -- or hurting, because the rows are missing 40% of
the feature set?**

Two arms, identical in every respect except the oldest season the model may
train on:

====  =========================================  ===============================
arm   trains on                                  rationale
====  =========================================  ===============================
A     every season before the test season        production, unchanged
B     seasons in ``[1415, test)``                the xG era only
====  =========================================  ===============================

Both predict the same 20,013 fixtures across the same 12 test seasons, so the
comparison is paired.

**Pre-specified before running**, because this experiment has an obvious way to
fool itself -- the ladder already told us the eras differ, so picking the
favourable era afterwards would be choosing the answer:

* **Primary**: pooled RPS over all 20,013 matches, paired bootstrap, the same
  gate every other change in this project has had to pass. A change is adopted
  only if this interval excludes zero in its favour.
* **Secondary, reported either way**: the same split by era used in the ladder
  (1516-1920 against 2021-2627). This is expected to move -- arm B throws away
  most of the training data for the early seasons and almost none for the late
  ones -- and is reported for understanding, not as a decision rule.

Run::

    python -m pipelines.training_window_ab
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import PROJECT_ROOT, load_raw_odds  # noqa: E402
from footballml.models.evaluate import (  # noqa: E402
    accuracy,
    brier_score,
    log_loss,
    paired_bootstrap,
    ranked_probability_score,
    rps_per_match,
)
from pipelines.ablation_ladder import (  # noqa: E402
    FIXTURE_KEY,
    PROB_COLUMNS,
    align,
    load_features,
)
from pipelines.backtest import run_backtest  # noqa: E402

logger = logging.getLogger("train_window")

OUT_DIR = PROJECT_ROOT / "experiments" / "training_window_ab"

#: First season Understat covers. Everything earlier has no xG, npxG, PPDA or
#: deep completions at all -- 94 of the 235 features missing outright.
XG_ERA_START = 1415

#: The era split from the ablation ladder, reused unchanged so the two
#: experiments are directly comparable.
EARLY = [1516, 1617, 1718, 1819, 1920]
LATE = [2021, 2122, 2223, 2324, 2425, 2526, 2627]


def metrics_for(preds: pd.DataFrame) -> dict[str, float]:
    probs = preds[PROB_COLUMNS].to_numpy()
    y = preds["FTR"].to_numpy()
    return {
        "rps": ranked_probability_score(y, probs),
        "log_loss": log_loss(y, probs),
        "brier": brier_score(y, probs),
        "accuracy": accuracy(y, probs),
        "n": int(len(preds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=1516)
    parser.add_argument("--train-from", type=int, default=XG_ERA_START,
                        help="Oldest training season for arm B.")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("footballml").setLevel(logging.WARNING)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building the production feature table ...")
    features = load_features()
    odds = load_raw_odds()

    arms = {
        "A_all_history": None,
        f"B_from_{args.train_from}": args.train_from,
    }
    preds_by_arm, rows, per_season_all = {}, [], []
    for name, train_from in arms.items():
        started = time.time()
        per_season, preds = run_backtest(
            features, start_season=args.start_season, odds=odds, train_from=train_from
        )
        elapsed = time.time() - started
        row = metrics_for(preds) | {
            "arm": name,
            "train_from": train_from if train_from is not None else "all",
            "median_train_rows": int(per_season["n_train"].median()),
            "seconds": round(elapsed, 1),
        }
        rows.append(row)
        per_season_all.append(per_season.assign(arm=name))
        preds_by_arm[name] = preds
        logger.info(
            "%-16s RPS %.5f  acc %.4f  median train rows %6d  %.0fs",
            name, row["rps"], row["accuracy"], row["median_train_rows"], elapsed,
        )

    a_name, b_name = list(arms)
    a, b = align(preds_by_arm[a_name], preds_by_arm[b_name])
    sa = rps_per_match(a["FTR"], a[PROB_COLUMNS].to_numpy())
    sb = rps_per_match(b["FTR"], b[PROB_COLUMNS].to_numpy())

    logger.info("\n=== paired bootstrap (negative favours arm B) ===")
    boots = []
    for label, seasons in (
        ("PRIMARY pooled", EARLY + LATE),
        ("secondary early 1516-1920", EARLY),
        ("secondary late 2021-2627", LATE),
    ):
        mask = a["Season"].isin(seasons).to_numpy()
        result = paired_bootstrap(sa[mask], sb[mask], n_boot=args.n_boot)
        result["subset"] = label
        boots.append(result)
        logger.info(
            "  %-28s n=%5d  delta %+.5f [%+.5f, %+.5f]  %s",
            label, result["n"], result["delta"], result["lo"], result["hi"],
            "SIGNIFICANT" if result["excludes_zero"] else "not significant",
        )

    results = pd.DataFrame(rows)
    bootstrap = pd.DataFrame(boots)
    results.to_csv(args.out_dir / "results.csv", index=False)
    bootstrap.to_csv(args.out_dir / "bootstrap_results.csv", index=False)
    pd.concat(per_season_all, ignore_index=True).to_csv(
        args.out_dir / "per_season.csv", index=False
    )
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "arms": {k: (v or "all") for k, v in arms.items()},
                "start_season": args.start_season,
                "xg_era_start": XG_ERA_START,
                "n_bootstrap": args.n_boot,
                "primary_endpoint": "pooled RPS, paired bootstrap, 95% CI excluding zero",
                "fixture_key": FIXTURE_KEY,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    primary = boots[0]
    verdict = (
        "ADOPT arm B" if primary["excludes_zero"] and primary["delta"] < 0
        else "REJECT: keep production (arm A)"
    )
    logger.info("\nVerdict on the primary endpoint: %s", verdict)
    logger.info("\n%s", results.to_string(index=False))


if __name__ == "__main__":
    main()
