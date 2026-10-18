"""How many players should a team rating be built from?

Production aggregates each club's **top 15 players by rating**
(``pipelines/build_players.py --squad-size``, feeding
:func:`footballml.players.team_strength.team_strength` with ``method="best_n"``).
That 15 has never been measured. Every other tuned number in this project
carries its evidence in `config/player_rating.yaml` -- `peak_weight` has a
six-row table, the defending weight has two -- and this one has nothing. The
library default is ``best_n: int = 14`` while production passes ``15``, which is
the tell: two different unexplained numbers, and nothing calls the 14.

So this sweeps it. The question is not academic: `best_n` decides which players
reach the per-line aggregates, so it moves `strength_defence`, `strength_attack`
and `strength_overall` -- twelve features the match model reads, and the numbers
the site publishes as team ratings.

**The gate is declared here, before the run, on purpose.** Two previous sweeps
in this project produced a "significant" winner on a paired bootstrap that then
failed to replicate season by season, so a bootstrap interval alone does not
decide anything:

1. *Primary* -- next-season validity. Attack against the goals a club scores the
   following season, defence against the goals it concedes, overall against the
   points it takes. This is the `atk>GF / ovr>pts / def>GA` triple that
   `config/player_rating.yaml` already reports for `peak_weight`.
2. *Secondary* -- walk-forward RPS, paired bootstrap against the shipped 15,
   **and** the per-season and per-league breakdown.
3. *Ship only if* a setting beats 15 on the correlations, does not regress RPS
   beyond its interval, and improves in most of the eleven seasons that can
   move. Otherwise 15 stays and this is recorded as inconclusive.

That last outcome is the likely one and is a real result: it would mean the
number nobody measured was fine, which is worth knowing with evidence.

Note the tension this design respects rather than resolves. `peak_weight`'s
comment says the correlations are "inside noise ... the backtest does [pick the
number]", while the defending weight shipped on the correlations precisely
because RPS could not resolve a difference that small. Both criteria are weak at
this scale, which is why both are printed and the gate was fixed in advance.

Run with::

    python -m pipelines.squad_size_sweep
    python -m pipelines.squad_size_sweep --quick   # 1819 on, for a smoke test
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

from footballml.data import PROCESSED_DIR, load_team_match_history  # noqa: E402
from footballml.features.build import AUTO, _next_season, build_match_features  # noqa: E402
from footballml.ingest.understat import season_labels  # noqa: E402
from footballml.models.evaluate import paired_bootstrap, rps_per_match  # noqa: E402
from footballml.players.ingest import fetch_player_seasons  # noqa: E402
from footballml.players.rating import rate_players  # noqa: E402
from footballml.players.team_strength import team_strength  # noqa: E402
from pipelines.backtest import load_raw_odds, run_backtest  # noqa: E402

logger = logging.getLogger("squad_size_sweep")

TMH = PROCESSED_DIR / "team_match_history_all.csv"

#: What production ships today, and the arm every delta is measured against.
SHIPPED = 15

#: Squad sizes to try, in the order they are reported.
#:
#: 13 and 17 bracket the shipped number rather than extending in one direction,
#: because a monotone trend across a bracket is the thing that would make a
#: result believable -- a single better point next door to 15 would not.
SIZES = (13, 14, SHIPPED, 16, 17)

#: A minimum season length before a club's outcome counts toward the
#: correlations. The big five play 34 or 38; anything shorter is a season still
#: in progress, whose goals-per-match is fine but whose *points total* is not
#: comparable with a finished one.
MIN_PLAYED = 20


def variants() -> list[tuple[str, int]]:
    """The arms, baseline first, with a determinism control second.

    `control-15` is the shipped setting a second time. It must come back
    bit-identical to `baseline`, which is the guard against reading a
    difference that is really run-to-run drift: this pipeline is exactly
    deterministic within one data vintage and demonstrably *not* between
    vintages, and a sweep with no such arm cannot tell the two apart.
    """
    built = [("baseline", SHIPPED), ("control-15", SHIPPED)]
    built += [(f"n-{n}", n) for n in SIZES if n != SHIPPED]
    return built


def _season_outcomes(history: pd.DataFrame) -> pd.DataFrame:
    """Goals for, goals against and points per club-season.

    Points are derived from the goals rather than read from ``Result``, so the
    figure does not depend on how that column happens to be encoded.
    """
    frame = history.assign(Season=history["Season"].astype(str))
    won = (frame["GoalsFor"] > frame["GoalsAgainst"]).astype(int) * 3
    drawn = (frame["GoalsFor"] == frame["GoalsAgainst"]).astype(int)
    frame = frame.assign(points=won + drawn)
    return (
        frame.groupby(["League", "Season", "Team"], observed=True)
        .agg(
            gf=("GoalsFor", "mean"),
            ga=("GoalsAgainst", "mean"),
            pts=("points", "sum"),
            played=("GoalsFor", "size"),
        )
        .reset_index()
        .query(f"played >= {MIN_PLAYED}")
    )


def next_season_validity(
    strength: pd.DataFrame, outcomes: pd.DataFrame, min_rows: int = 50
) -> dict[str, float]:
    """Each strength line against what the club does the *following* season.

    Next season, not this one, for the reason `add_team_strength` exists:
    season S strength is joined onto season S+1 matches, so that is the only
    direction the model ever reads it. A same-season correlation would also
    flatter any rating built from that season's own results.

    Generalised from `next_season_defence` in `pipelines/defending_sweep.py`,
    which does this for the back line alone.

    Returns:
        ``atk_vs_gf``, ``def_vs_ga`` and ``ovr_vs_pts``. Attack and overall
        should come back **positive** (a better side scores more and takes more
        points) and defence **negative** (a better back line concedes fewer),
        so the signs differ by design and are not a bug.
    """
    pairs = (
        ("strength_attack", "gf", "atk_vs_gf"),
        ("strength_defence", "ga", "def_vs_ga"),
        ("strength_overall", "pts", "ovr_vs_pts"),
    )
    out: dict[str, float] = {}
    for column, outcome, name in pairs:
        if column not in strength.columns:
            out[name] = float("nan")
            continue
        lhs = strength.assign(
            _next=strength["Season"].astype(str).map(_next_season),
            value=strength[column],
        )[["League", "Team", "_next", "value"]].dropna(subset=["value"])
        merged = lhs.merge(
            outcomes.rename(columns={"Season": "_next"}),
            on=["League", "Team", "_next"],
            how="inner",
        )
        # Too few club-seasons to read a correlation from. Reporting one off a
        # handful of clubs would be worse than reporting nothing.
        out[name] = (
            float(merged["value"].corr(merged[outcome]))
            if len(merged) >= min_rows
            else float("nan")
        )
    return out


def _score(preds: pd.DataFrame) -> tuple[np.ndarray, float]:
    # The raw H/D/A column, not an integer encoding of it: `_one_hot` compares
    # against the strings, so integers silently one-hot to all zeros and RPS
    # stops reading the result at all. It now raises instead, and this is the
    # call that taught it to.
    truth = preds["FTR"]
    probs = preds[["prob_H", "prob_D", "prob_A"]].to_numpy()
    picked = probs.argmax(axis=1)
    actual = truth.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return rps_per_match(truth, probs), float((picked == actual).mean())


def _breakdown(
    label: str, keys: pd.Series, base: np.ndarray, arms: dict[str, np.ndarray], minimum: int = 100
) -> None:
    """Print the RPS delta per season or per league, and how many improved.

    The paired bootstrap resamples fixtures as if they were independent, but
    matches inside a season share a model fit -- so its effective sample is
    nearer the number of seasons than the number of matches. A delta that holds
    in two seasons and reverses in two others is not a finding, and only this
    table shows that.
    """
    names = [n for n in arms if n != "baseline"]
    print(f"\nRPS delta vs baseline, by {label} (negative = the variant is better)")
    print(f"  {label:8s} {'n':>6s}  " + "  ".join(f"{n:>12s}" for n in names))
    wins = dict.fromkeys(names, 0)
    total = 0
    for key in sorted(keys.unique()):
        mask = (keys == key).to_numpy()
        if mask.sum() < minimum:
            continue
        total += 1
        cells = []
        for name in names:
            delta = float((arms[name][mask] - base[mask]).mean())
            wins[name] += delta < 0
            cells.append(f"{delta:+12.5f}")
        print(f"  {str(key):8s} {int(mask.sum()):6d}  " + "  ".join(cells))
    print(f"  improved in, of {total} {label}s:")
    for name in names:
        print(f"    {name:12s} {wins[name]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Four seasons, for a smoke test.")
    parser.add_argument("--start-season", type=int, default=1516)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("soccerdata").setLevel(logging.ERROR)

    # This module writes nothing, and that is deliberate rather than incidental.
    #
    # An earlier sweep in this project swapped the production `team_strength.csv`
    # per variant and restored it in a `finally`. A `finally` does not survive
    # the process being killed, and a timeout left the shipped file holding a
    # variant's numbers -- 1,160 of 1,170 rows wrong, by up to 5.3 points, with
    # the API serving them.
    #
    # It was never needed: `build_match_features` takes the strength *frame*, so
    # every arm lives in memory and the file on disk stays exactly as shipped.
    # `tests/test_squad_size_sweep.py` asserts there is no write to interrupt.
    seasons = season_labels("1415", None)
    logger.info("Ingesting players across %d seasons", len(seasons))
    started = time.monotonic()
    players = fetch_player_seasons(None, seasons)
    logger.info(
        "Ingest done in %.0fs (%d player-seasons)", time.monotonic() - started, len(players)
    )

    # Rated once, outside the loop. `best_n` selects *from* the ratings and
    # changes nothing upstream of them, so re-rating per arm would burn minutes
    # to reproduce an identical frame -- the one real saving this sweep has over
    # `defending_sweep`, which cannot hoist this because it varies the weights.
    rated = rate_players(players)
    logger.info("Rated %d player-seasons once for every arm", len(rated))

    tmh = load_team_match_history(TMH)
    history = pd.read_csv(TMH, low_memory=False)
    outcomes = _season_outcomes(history)
    odds = load_raw_odds()

    results: list[dict] = []
    vectors: dict[str, np.ndarray] = {}
    season_of: pd.Series | None = None
    league_of: pd.Series | None = None

    for name, size in variants():
        logger.info("--- %s: best_n=%d ---", name, size)
        strength = team_strength(rated, method="best_n", best_n=size)
        features = build_match_features(tmh, strength=strength, european=AUTO)
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

        validity = next_season_validity(strength, outcomes)
        results.append(
            {
                "variant": name,
                "best_n": size,
                "n": len(preds),
                "rps": float(rps.mean()),
                "accuracy": accuracy,
                **validity,
                "team_seasons": len(strength),
                "mean_players": float(strength["n_players"].mean()),
            }
        )

    table = pd.DataFrame(results)
    base = vectors["baseline"]
    deltas = []
    for name in table["variant"]:
        if name == "baseline":
            deltas.append({"variant": name, "delta": 0.0, "lo": 0.0, "hi": 0.0, "sig": False})
            continue
        boot = paired_bootstrap(base, vectors[name])
        deltas.append(
            {
                "variant": name,
                "delta": boot["delta"],
                "lo": boot["lo"],
                "hi": boot["hi"],
                "sig": boot["excludes_zero"],
            }
        )

    table = table.merge(pd.DataFrame(deltas), on="variant")
    print("\n=== Squad size sweep ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    identical = np.array_equal(vectors["baseline"], vectors["control-15"])
    print(
        f"\ndeterminism control: baseline and control-15 are "
        f"{'bit-identical' if identical else '**DIFFERENT -- do not read this table**'}"
    )

    assert season_of is not None and league_of is not None
    _breakdown("season", season_of, base, vectors)
    _breakdown("league", league_of, base, vectors)

    print(
        "\nThe gate, declared before this ran: a setting ships only if it beats"
        "\nbest_n=15 on the correlations, does not regress RPS beyond its"
        "\ninterval, and improves in most of the seasons that can move."
        "\natk_vs_gf and ovr_vs_pts should be positive, def_vs_ga negative."
        "\nNothing here writes to data/processed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
