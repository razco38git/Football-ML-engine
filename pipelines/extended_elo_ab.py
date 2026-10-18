"""Should Elo learn from the European results the model cannot predict?

Elo is the only feature that crosses a league boundary -- its updates are
zero-sum, so without European ties the five leagues are five pools each anchored
at 1500. Production feeds it `european_matches.csv`: **829 ties where both clubs
are in the big five**. On disk sit another **3,178** it ignores, involving 180
clubs that have no squad rating, no form and no xG, and which are dropped for
that reason by `big_five_only`.

Elo needs none of those things. It needs results, and the dropped file has kept
`FTHG`/`FTAG` all along. Measured separately (`pipelines/european_elo_reach.py`,
written up in EXPERIMENTS.md), feeding them in improves an Elo-only predictor on
the 827 ties where the full model can also be scored: **0.21760 to 0.21621,
-0.00139 [-0.00274, -0.00008], better in 8 of 10 seasons**.

That is the upside. This measures the risk, which is much larger and entirely
domestic: Elo is one pool, so changing what it learns from changes the `elo`
feature on all 20,074 walk-forward fixtures, not just the European ones.

**So this is a non-inferiority test, not a search for a gain.** The acceptance
criteria are fixed here, before the run:

    ship         upper CI bound on the domestic RPS delta < +0.0005, and
                 neutral-or-better in >= 6 of the 11 movable seasons
    no-ship      the interval excludes zero on the wrong side, or most
                 seasons worsen
    inconclusive -> no-ship; the default is the status quo

+0.0005 is not arbitrary: it is the measurement floor established in
EXPERIMENTS.md, below which this pipeline cannot resolve a difference at all.

The dropped ties are mostly **not** Champions League -- 1,616 Europa, 723
Conference, 839 UCL -- so a third arm excludes the Conference League, the
newest and weakest competition and the likeliest source of noise. It is declared
here rather than sliced out afterwards if the headline disappoints.

Run with::

    python -m pipelines.extended_elo_ab
    python -m pipelines.extended_elo_ab --quick   # 1819 on, for a smoke test
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
    load_european_matches,
    load_team_match_history,
)
from footballml.features.build import build_match_features  # noqa: E402
from footballml.models.evaluate import paired_bootstrap, rps_per_match  # noqa: E402
from pipelines.backtest import load_raw_odds, run_backtest  # noqa: E402

logger = logging.getLogger("extended_elo_ab")

TMH = PROCESSED_DIR / "team_match_history_all.csv"
DROPPED = PROCESSED_DIR / "european_dropped.csv"

#: Competitions excluded from the third arm.
#:
#: The Conference League is the newest and weakest of the three and contributes
#: 723 of the dropped ties. If the extension helps overall but this arm helps
#: more, the extra data is carrying noise as well as signal -- a diagnosis, and
#: the reason this is declared before the run rather than after it.
WEAKEST = ("UECL",)

#: The floor below which this pipeline cannot resolve a difference.
#:
#: Measured in EXPERIMENTS.md: nine arms of the defending sweep spanned 0.00012,
#: the squad-size sweep 0.00018, and neither produced a significant result on
#: 20,074 fixtures. A non-inferiority margin tighter than this would be claiming
#: a precision the harness does not have.
MARGIN = 0.0005

#: Seasons that can actually move. The earliest test fold trains with the
#: strength features absent, so its delta is exactly zero for any arm.
MOVABLE_SEASONS = 11


def arms(dropped: pd.DataFrame, kept: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """What each arm's Elo walk is allowed to learn from.

    `production` is what `add_elo(extra=...)` receives today, named so the table
    cannot be misread as a hypothetical.
    """
    without_weakest = dropped[~dropped["League"].isin(WEAKEST)]
    return [
        ("production", kept),
        ("extended", pd.concat([kept, dropped], ignore_index=True)),
        ("extended-no-uecl", pd.concat([kept, without_weakest], ignore_index=True)),
    ]


def _score(preds: pd.DataFrame) -> tuple[np.ndarray, float]:
    # The raw H/D/A column. Mapping it to integers first is what made an earlier
    # sweep score a function of its own forecast; `_one_hot` now raises on it.
    truth = preds["FTR"]
    probs = preds[["prob_H", "prob_D", "prob_A"]].to_numpy()
    actual = truth.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return rps_per_match(truth, probs), float((probs.argmax(axis=1) == actual).mean())


def _breakdown(label: str, keys: pd.Series, base: np.ndarray, arms_: dict[str, np.ndarray],
               minimum: int = 100) -> dict[str, int]:
    """Per-season or per-league deltas, and how many did not get worse.

    The paired bootstrap resamples fixtures as if independent, but matches in a
    season share a model fit, so its effective sample is nearer the season count.
    Counting neutral-or-better rather than strictly better, because the
    hypothesis is non-inferiority.
    """
    names = [n for n in arms_ if n != "production"]
    print(f"\nDomestic RPS delta vs production, by {label} (negative = better)")
    print(f"  {label:8s} {'n':>6s}  " + "  ".join(f"{n:>18s}" for n in names))
    ok = dict.fromkeys(names, 0)
    total = 0
    for key in sorted(keys.unique()):
        mask = (keys == key).to_numpy()
        if mask.sum() < minimum:
            continue
        deltas = {n: float((arms_[n][mask] - base[mask]).mean()) for n in names}
        # A group no arm can move is not evidence that no arm made it worse.
        # The earliest test fold trains with the strength features absent, so
        # every arm scores it identically and `delta <= 0` would hand each of
        # them a free pass -- counted against a denominator that already
        # excluded it, which is how a 5-of-11 record printed as 6.
        if all(delta == 0.0 for delta in deltas.values()):
            print(f"  {str(key):8s} {int(mask.sum()):6d}  " +
                  "  ".join(f"{deltas[n]:+18.5f}" for n in names) + "   (cannot move)")
            continue
        total += 1
        for name in names:
            ok[name] += deltas[name] <= 0
        print(f"  {str(key):8s} {int(mask.sum()):6d}  " +
              "  ".join(f"{deltas[n]:+18.5f}" for n in names))
    print(f"  neutral-or-better in, of {total} {label}s that can move:")
    for name in names:
        print(f"    {name:18s} {ok[name]}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Four seasons, for a smoke test.")
    parser.add_argument("--start-season", type=int, default=1516)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("soccerdata").setLevel(logging.ERROR)

    # Writes nothing. `build_match_features` takes the European frame as an
    # argument, so each arm's features live in memory and no file on disk is
    # touched -- the property `tests/test_extended_elo_ab.py` asserts, and the
    # one an interrupted run depends on.
    tmh = load_team_match_history(TMH)
    kept = load_european_matches()
    dropped = pd.read_csv(DROPPED, low_memory=False).dropna(subset=["FTHG", "FTAG"])
    odds = load_raw_odds()

    print(f"European ties available to the walk: {len(kept)} kept, {len(dropped)} dropped")
    print("dropped by competition:")
    for comp, n in dropped["League"].value_counts().items():
        print(f"  {comp:6s} {n}")

    results: list[dict] = []
    vectors: dict[str, np.ndarray] = {}
    season_of: pd.Series | None = None
    league_of: pd.Series | None = None

    for name, european in arms(dropped, kept):
        logger.info("--- %s: Elo learns from %d European ties ---", name, len(european))
        started = time.monotonic()
        features = build_match_features(tmh, european=european)
        _, preds = run_backtest(
            features,
            start_season=1819 if args.quick else args.start_season,
            use_dixon_coles=True,
            odds=odds if not odds.empty else None,
            calibrate=True,
        )
        rps, accuracy = _score(preds)
        vectors[name] = rps
        if season_of is None:
            season_of = preds["Season"].reset_index(drop=True).astype(str)
            league_of = preds["League"].reset_index(drop=True).astype(str)
        results.append({
            "arm": name,
            "european_ties": len(european),
            "n": len(preds),
            "rps": float(rps.mean()),
            "accuracy": accuracy,
            "seconds": round(time.monotonic() - started),
        })

    # Every arm must have predicted the identical fixture list in identical
    # order, or the pairing is meaningless. Asserted rather than trusted, as
    # `align` in ablation_ladder puts it.
    lengths = {name: len(v) for name, v in vectors.items()}
    assert len(set(lengths.values())) == 1, f"arms scored different fixture counts: {lengths}"

    table = pd.DataFrame(results)
    base = vectors["production"]
    rows = []
    for name in table["arm"]:
        if name == "production":
            rows.append({"arm": name, "delta": 0.0, "lo": 0.0, "hi": 0.0, "sig": False})
            continue
        boot = paired_bootstrap(base, vectors[name])
        rows.append({
            "arm": name, "delta": boot["delta"], "lo": boot["lo"],
            "hi": boot["hi"], "sig": boot["excludes_zero"],
        })
    table = table.merge(pd.DataFrame(rows), on="arm")

    print("\n=== Extended Elo: domestic walk-forward ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    assert season_of is not None and league_of is not None
    by_season = _breakdown("season", season_of, base, vectors)
    _breakdown("league", league_of, base, vectors)

    print("\n=== Verdict against the criteria declared before the run ===")
    for name in vectors:
        if name == "production":
            continue
        hi = float(table.loc[table["arm"] == name, "hi"].iloc[0])
        seasons_ok = by_season[name]
        passes = hi < MARGIN and seasons_ok >= 6
        print(
            f"  {name:18s} upper CI {hi:+.5f} (< {MARGIN:+.5f}? "
            f"{'yes' if hi < MARGIN else 'NO'})  "
            f"neutral-or-better in {seasons_ok} of {MOVABLE_SEASONS} movable seasons "
            f"(>= 6? {'yes' if seasons_ok >= 6 else 'NO'})  "
            f"-> {'MEETS the bar' if passes else 'does NOT meet the bar'}"
        )
    print(
        "\nMeeting the bar means no detectable domestic harm, not a domestic gain."
        "\nThe case for shipping rests on the European result measured separately;"
        "\nthis run exists to say whether it costs anything."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
