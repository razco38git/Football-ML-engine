"""Should the match model fall back to the pre-season squad rating?

`add_team_strength` joins the **previous** season's measured strength, and
leaves the rest NaN. The rest is not a rounding error. Across 1516-2627,
counting team-sides whose season *and* its predecessor are inside Understat's
coverage, **5,674 of 40,334 (14.1%)** have no previous-season measured strength,
steady at 13-15% every season. They are not random: Metz, Frosinone, Troyes,
Norwich, Burnley, Fulham -- promoted and yo-yo sides, which the measured rating
*structurally* cannot reach, because a club that was in Ligue 2 last season has
no top-flight season to measure.

`squad_strength.csv` covers **5,636 of them, 99.3%**. It is a different
quantity -- EA's September roster, each player carrying his most recent rating
from strictly before this season -- and until now the model never saw it.

This measures whether filling the hole with it is worth shipping.

Why the result is likely to be small, said before the run
---------------------------------------------------------

The whole `squad` family -- all twelve columns, on every fixture -- is worth
**-0.00015 RPS [-0.00032, +0.00001]** in the ablation ladder, which does not
clear zero on 20,013 matches. This change does not add a family; it improves
one family's coverage from 86% to 99%. So the honest prior is that the pooled
RPS cannot resolve it, and that is why the targeted endpoint below is declared
*here* rather than reached for afterwards if the headline disappoints.

Acceptance criteria, fixed before the run
-----------------------------------------

Three endpoints, in order. `delta = variant - baseline`, so negative is better.

1. **Pooled RPS**, all walk-forward fixtures. The headline, and the only one
   that can justify shipping on prediction grounds by itself.
2. **Targeted RPS**, the fixtures where at least one side was actually filled.
   The pooled figure dilutes any effect across the ~84% of fixtures the change
   cannot touch; this is the subgroup where it can act at all. Declared now,
   with its mask defined as `home_` or `away_strength_is_preseason`, so it
   cannot be re-cut later until it says something.
3. **Next-season strength validity**, the project's secondary criterion when
   RPS cannot separate two configurations: `atk` against next season's goals
   for, `def` against goals against, `ovr` against points, over the strength
   value *as the model actually reads it*. This one does not need a model fit,
   so it resolves where RPS does not.

        ship         an RPS interval excludes zero on the better side; or
                     both RPS intervals straddle zero (no harm, upper bound
                     within the +0.0005 measurement floor) **and** validity
                     improves on all three lines
        no-ship      an RPS interval excludes zero on the worse side, or
                     validity degrades on any line
        inconclusive -> no-ship; the default is the status quo

+0.0005 is the measurement floor established in EXPERIMENTS.md, below which
this harness cannot resolve a difference at all.

A `control` arm re-runs the baseline unchanged and must come back bit-identical.
Without it a 0.0001 movement cannot be told from run-to-run drift -- and this
pipeline is deterministic within one data vintage and demonstrably not between
vintages.

Run with::

    python -m pipelines.preseason_fallback_ab
    python -m pipelines.preseason_fallback_ab --quick   # 2223 on, a smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import (  # noqa: E402
    PROCESSED_DIR,
    load_squad_strength,
    load_team_match_history,
    load_team_strength,
)
from footballml.features.build import (  # noqa: E402
    PRESEASON_COLUMNS,
    PRESEASON_FLAG,
    _next_season,
    build_match_features,
)
from footballml.models.evaluate import paired_bootstrap, rps_per_match  # noqa: E402
from pipelines.backtest import load_raw_odds, run_backtest  # noqa: E402
from pipelines.squad_size_sweep import _season_outcomes, next_season_validity  # noqa: E402

logger = logging.getLogger("preseason_fallback_ab")

TMH = PROCESSED_DIR / "team_match_history_all.csv"

#: The floor below which this harness cannot resolve an RPS difference.
#: Measured in EXPERIMENTS.md over several sweeps that each spanned less.
MARGIN = 0.0005

#: Match identifiers, for carrying the fill mask onto the predictions.
_KEY = ["League", "Season", "Date", "HomeTeam", "AwayTeam"]


def arms(preseason: pd.DataFrame) -> list[tuple[str, pd.DataFrame | None]]:
    """The arms, baseline first, with a determinism control second.

    `baseline` passes **no** pre-season table, which is production exactly as it
    ships -- measured against anything else, every delta below would describe a
    configuration nobody runs. `control` is `baseline` a second time and must
    come back bit-identical; without it a movement of 0.0001 cannot be told
    from run-to-run drift.
    """
    return [("baseline", None), ("control", None), ("fallback", preseason)]


def _score(preds: pd.DataFrame) -> tuple[np.ndarray, float]:
    # The raw H/D/A column, not an integer encoding of it: `_one_hot` compares
    # against the strings, so integers silently one-hot to all zeros and RPS
    # stops reading the result at all.
    truth = preds["FTR"]
    probs = preds[["prob_H", "prob_D", "prob_A"]].to_numpy()
    actual = truth.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return rps_per_match(truth, probs), float((probs.argmax(axis=1) == actual).mean())


def effective_strength(strength: pd.DataFrame, preseason: pd.DataFrame) -> pd.DataFrame:
    """The strength table *as the model reads it*, labelled for the validity test.

    `next_season_validity` takes a frame keyed at season S and correlates it
    against season S+1's outcomes, because S strength is what an S+1 match sees.
    The fallback breaks that one-to-one labelling: a filled side's value comes
    from season S+1's own pre-season table. So a filled (S+1, team) is written
    back here as a row labelled S, which keeps "the number joined onto this
    match" and "the season it is read in" lined up exactly as the join does.

    Only holes are filled, mirroring `_fill_from_preseason`: a team-season that
    already has a measured row keeps it untouched.
    """
    measured = strength.copy()
    measured["Season"] = measured["Season"].astype(str)
    have = set(zip(measured["Season"], measured["Team"], strict=True))

    extra = preseason.rename(columns=PRESEASON_COLUMNS).copy()
    extra["Season"] = extra["Season"].astype(str)
    # Label each pre-season row with the season *before* the one it describes,
    # which is the slot the measured table would have occupied.
    extra["Season"] = extra["Season"].map(_previous_season)
    keep = [(s, t) not in have for s, t in zip(extra["Season"], extra["Team"], strict=True)]
    extra = extra[keep]

    columns = [c for c in measured.columns if c in extra.columns]
    return pd.concat([measured, extra[columns]], ignore_index=True)


def _previous_season(label: str) -> str:
    """``"2526"`` -> ``"2425"``. The inverse of `_next_season`."""
    text = str(label)
    if len(text) != 4 or not text.isdigit():
        return text
    start = int(text[:2]) - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _validity(label: str, strength: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, float]:
    scores = next_season_validity(strength, outcomes)
    print(
        f"  {label:28s} atk_vs_gf {scores['atk_vs_gf']:+.4f}   "
        f"def_vs_ga {scores['def_vs_ga']:+.4f}   ovr_vs_pts {scores['ovr_vs_pts']:+.4f}"
    )
    return scores


def validity_by_population(
    strength: pd.DataFrame, preseason: pd.DataFrame, outcomes: pd.DataFrame
) -> None:
    """The pooled correlation, split by which population a row belongs to.

    This is a diagnostic for endpoint 3, not a fourth endpoint, and it exists
    because the pooled figure can only be read with it. The rows the fallback
    adds are promoted and yo-yo sides, which are a *harder* population than
    established clubs by any measure. So a pooled correlation that falls
    slightly is the expected consequence of adding harder rows, not evidence
    that the added values are wrong -- and the two cannot be told apart from
    the pooled number alone.

    The question this answers instead: on the rows it adds, does the pre-season
    rating point the right way at all?
    """
    pairs = (("strength_attack", "gf"), ("strength_defence", "ga"), ("strength_overall", "pts"))
    have = set(zip(strength["Season"].astype(str), strength["Team"], strict=True))

    pre = preseason.rename(columns=PRESEASON_COLUMNS).copy()
    pre["_read_in"] = pre["Season"].astype(str)
    pre["_slot"] = pre["_read_in"].map(_previous_season)
    fills = [(s, t) not in have for s, t in zip(pre["_slot"], pre["Team"], strict=True)]

    # The measured table as the model reads it: shifted a season forward.
    shifted = strength.assign(_read_in=strength["Season"].astype(str).map(_next_season))

    populations = (
        ("pre-season, on the rows it FILLS", pre[fills]),
        ("pre-season, on rows already covered", pre[[not f for f in fills]]),
        ("measured, on its own rows", shifted),
    )
    print("\n  By population, each value against the season it is read in:")
    for label, frame in populations:
        cells = []
        for column, outcome in pairs:
            lhs = frame[["League", "Team", "_read_in", column]].dropna(subset=[column])
            merged = lhs.merge(
                outcomes.rename(columns={"Season": "_read_in"}),
                on=["League", "Team", "_read_in"],
                how="inner",
            )
            correlation = (
                merged[column].corr(merged[outcome]) if len(merged) >= 50 else float("nan")
            )
            cells.append(f"{correlation:+.4f}")
        print(f"    {label:38s} {cells[0]}  {cells[1]}  {cells[2]}   n {len(merged)}")


def endpoint_three(tmh: pd.DataFrame, strength: pd.DataFrame, preseason: pd.DataFrame) -> int:
    """Next-season strength validity, and how many of the three lines improved.

    Separate from the rest of `main` because it needs no model fit: when RPS
    cannot resolve a question, this is the criterion that has to, and waiting
    ten minutes for three backtests to re-answer it is how a secondary
    criterion stops being consulted.
    """
    print("\n=== Endpoint 3: next-season strength validity, as the model reads it ===")
    print("  atk_vs_gf and ovr_vs_pts should be positive, def_vs_ga negative.")
    outcomes = _season_outcomes(tmh)
    before = _validity("measured only (baseline)", strength, outcomes)
    after = _validity("with pre-season fallback", effective_strength(strength, preseason), outcomes)

    signs = {"atk_vs_gf": 1.0, "def_vs_ga": -1.0, "ovr_vs_pts": 1.0}
    print("\n  improvement (positive = the fallback reads the next season better):")
    improved = 0
    for line, sign in signs.items():
        gain = sign * (after[line] - before[line])
        improved += gain > 0
        print(f"    {line:12s} {gain:+.4f}")
    print(f"  improved on {improved} of 3 lines")

    validity_by_population(strength, preseason, outcomes)
    return improved


def _per_season(keys: pd.Series, base: np.ndarray, variant: np.ndarray) -> int:
    """Per-season deltas, and how many seasons did not get worse.

    The paired bootstrap resamples fixtures as if independent, but matches in a
    season share a model fit, so its effective sample is nearer the season
    count. A season no arm can move is reported and excluded from the
    denominator rather than handed a free pass.
    """
    print("\nRPS delta by season (negative = better)")
    better = 0
    movable = 0
    for season in sorted(keys.unique()):
        mask = (keys == season).to_numpy()
        delta = float((variant[mask] - base[mask]).mean())
        tag = ""
        if delta == 0.0:
            tag = "   (cannot move)"
        else:
            movable += 1
            better += delta <= 0
        print(f"  {season:6s} {int(mask.sum()):6d}  {delta:+12.5f}{tag}")
    print(f"  neutral-or-better in {better} of {movable} seasons that can move")
    return better


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="2223 on, for a smoke test.")
    parser.add_argument("--start-season", type=int, default=1516)
    parser.add_argument(
        "--validity-only",
        action="store_true",
        help="Endpoint 3 alone. It needs no model fit, so it answers in seconds.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Writes nothing. Both arms' features live in memory; the only files touched
    # are read-only inputs.
    tmh = load_team_match_history(TMH)
    strength = load_team_strength()
    preseason = load_squad_strength()
    odds = load_raw_odds()
    if strength.empty or preseason.empty:
        raise SystemExit("need both team_strength.csv and squad_strength.csv")

    print(
        f"measured strength: {len(strength)} team-seasons | "
        f"pre-season ratings: {len(preseason)} team-seasons"
    )

    if args.validity_only:
        endpoint_three(tmh, strength, preseason)
        return 0

    results: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    season_of: pd.Series | None = None
    filled: np.ndarray | None = None

    for name, table in arms(preseason):
        logger.info("--- %s ---", name)
        started = time.monotonic()
        features = build_match_features(tmh, strength=strength, preseason_strength=table)
        _, preds = run_backtest(
            features,
            start_season=2223 if args.quick else args.start_season,
            use_dixon_coles=True,
            odds=odds if not odds.empty else None,
            calibrate=True,
        )
        rps, accuracy = _score(preds)
        vectors[name] = rps
        if season_of is None:
            season_of = preds["Season"].reset_index(drop=True).astype(str)
        if table is not None:
            # The targeted mask, carried from the features onto the scored
            # fixtures by identifier rather than by position -- the two frames
            # are differently filtered and an index-aligned mask would quietly
            # point at the wrong matches.
            flags = features[[*_KEY, f"home_{PRESEASON_FLAG}", f"away_{PRESEASON_FLAG}"]].copy()
            flags["Season"] = flags["Season"].astype(str)
            marked = (
                preds[_KEY]
                .assign(Season=preds["Season"].astype(str))
                .merge(flags, on=_KEY, how="left", validate="1:1")
            )
            filled = (
                marked[f"home_{PRESEASON_FLAG}"].fillna(0.0)
                + marked[f"away_{PRESEASON_FLAG}"].fillna(0.0)
            ).to_numpy() > 0
        results.append(
            {
                "arm": name,
                "n": len(preds),
                "rps": float(rps.mean()),
                "accuracy": accuracy,
                "seconds": round(time.monotonic() - started),
            }
        )

    lengths = {name: len(v) for name, v in vectors.items()}
    assert len(set(lengths.values())) == 1, f"arms scored different fixture counts: {lengths}"
    assert season_of is not None and filled is not None

    print("\n=== Determinism control ===")
    drift = float(np.abs(vectors["control"] - vectors["baseline"]).max())
    print(f"  max per-match |control - baseline|: {drift:.3e}")
    print(
        "  bit-identical, so any movement below is the change"
        if drift == 0.0
        else "  NOT identical -- a delta this size cannot be attributed to the change"
    )

    base = vectors["baseline"]
    variant = vectors["fallback"]

    print("\n=== Arms ===")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== Endpoint 1: pooled RPS ===")
    pooled = paired_bootstrap(base, variant)
    print(
        f"  n {int(pooled['n'])}  delta {pooled['delta']:+.5f}  "
        f"95% CI [{pooled['lo']:+.5f}, {pooled['hi']:+.5f}]  "
        f"{'excludes zero' if pooled['excludes_zero'] else 'straddles zero'}"
    )

    print("\n=== Endpoint 2: targeted RPS, fixtures with a filled side ===")
    print(
        f"  {int(filled.sum())} of {len(filled)} scored fixtures "
        f"({100 * filled.mean():.1f}%) had at least one side filled"
    )
    targeted = paired_bootstrap(base[filled], variant[filled])
    print(
        f"  n {int(targeted['n'])}  delta {targeted['delta']:+.5f}  "
        f"95% CI [{targeted['lo']:+.5f}, {targeted['hi']:+.5f}]  "
        f"{'excludes zero' if targeted['excludes_zero'] else 'straddles zero'}"
    )
    untouched = paired_bootstrap(base[~filled], variant[~filled])
    print(
        f"  and on the {int((~filled).sum())} fixtures it cannot touch directly: "
        f"delta {untouched['delta']:+.5f} "
        f"[{untouched['lo']:+.5f}, {untouched['hi']:+.5f}] -- not zero, because "
        "both arms retrain on a different feature set"
    )

    _per_season(season_of, base, variant)

    improved = endpoint_three(tmh, strength, preseason)

    print("\n=== Verdict against the criteria declared before the run ===")
    better = [b for b in (pooled, targeted) if b["excludes_zero"] and float(b["delta"]) < 0]
    worse = [b for b in (pooled, targeted) if b["excludes_zero"] and float(b["delta"]) > 0]
    no_harm = float(pooled["hi"]) < MARGIN and float(targeted["hi"]) < MARGIN
    print(f"  an RPS interval excludes zero, better side: {'yes' if better else 'NO'}")
    print(f"  an RPS interval excludes zero, worse side:  {'yes' if worse else 'no'}")
    print(f"  no detectable harm (both upper CIs < {MARGIN}): {'yes' if no_harm else 'NO'}")
    print(f"  validity improved on all three lines:      {'yes' if improved == 3 else 'NO'}")

    if worse:
        verdict = "DO NOT SHIP -- RPS got significantly worse"
    elif better:
        verdict = "SHIP -- RPS resolves it in the change's favour"
    elif no_harm and improved == 3:
        verdict = (
            "SHIP on the secondary criterion -- RPS could not resolve it, "
            "no detectable harm, validity improves on all three lines"
        )
    else:
        verdict = "DO NOT SHIP -- inconclusive, and the default is the status quo"
    print(f"\n  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
