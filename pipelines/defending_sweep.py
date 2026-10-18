"""Two ways to fix a backwards defending sub-rating, measured against each other.

Across 859 team-seasons the back line's ``sub_defending`` correlates **+0.10**
with the goals its team conceded. Not weak -- backwards. Teams whose defenders
rank higher on "defending" concede more, because every input to it counts
defensive *actions* and a defender makes more of those when his side defends
worse. It is 62% of a centre-back's composite, and the real signal in a
defender's rating comes from EA's overall (-0.52) and from involvement (-0.50).

Two fixes are possible without data we cannot get, and this runs both:

``share``
    Cut the weight on a backwards signal. CB ``defending`` from 0.62 down to 0,
    the freed weight redistributed across involvement, creation and finishing in
    their existing proportions. Untested: the sweep recorded in
    ``config/player_rating.yaml`` moved the *full-back* share upward, 0.38 to
    0.80, and lost at every step. Downward has never been tried.

``outcome``
    Give the defenders what the goalkeeper already has. ``GK`` is rated 97% on
    outcomes -- ``goals_against_per90: -2.0``, ``clean_sheet_pct`` -- while the
    centre-backs in front of him are rated 0% on them. Adds ``team_defence`` at
    0.15, 0.30 and 0.45, taken out of ``defending``.

Judged on both numbers that matter, for every variant:

1. Walk-forward RPS over the same fixtures, paired bootstrap against the
   shipped configuration.
2. The back line against the **next** season's goals conceded -- next, not
   same, because that is how the model consumes strength. The shipped number is
   -0.489, and the ceiling is known: a team's *actual* previous conceding
   predicts next season's at only -0.555.

Expect small RPS effects whichever wins. ``strength_defence`` ranks 93rd of 125
features in the match model, so this is mostly a question about whether the
published ratings are defensible, not about prediction.

Run with::

    python -m pipelines.defending_sweep
    python -m pipelines.defending_sweep --quick   # 1516-1819 only, for a smoke test
"""

from __future__ import annotations

import argparse
import copy
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
from footballml.players.rating import load_config, rate_players  # noqa: E402
from footballml.players.team_strength import team_strength  # noqa: E402
from pipelines.backtest import load_raw_odds, run_backtest  # noqa: E402

logger = logging.getLogger("defending_sweep")

TMH = PROCESSED_DIR / "team_match_history_all.csv"

#: Sub-ratings the freed weight is returned to, in the `share` variant.
#:
#: Proportionally, so the balance between them is untouched and the only thing
#: the sweep changes is how much of the composite rests on defending.
REDISTRIBUTE_TO = ("involvement", "creation", "finishing")


def _pool(weights: dict) -> float:
    """The weight the two defensive groups share between them.

    Both variants move weight *within* this pool, never into or out of it, so
    they describe the same experiment whichever setting happens to be shipped.
    Without that the builders drift: once `team_defence: 0.30` shipped,
    subtracting another 0.30 from `defending` left the weights summing to 0.70.
    """
    return weights["defending"] + weights.get("team_defence", 0.0)


def _share_variant(config: dict, defending: float) -> dict:
    """CB defending at `defending`, the remainder shared out as it is now."""
    out = copy.deepcopy(config)
    weights = out["positions"]["CB"]["sub_rating_weights"]
    freed = _pool(weights) - defending
    weights["team_defence"] = 0.0
    others = {k: weights[k] for k in REDISTRIBUTE_TO if k in weights}
    total = sum(others.values())
    weights["defending"] = defending
    for name, value in others.items():
        weights[name] = value + freed * (value / total)
    return out


def _outcome_variant(config: dict, share: float) -> dict:
    """`team_defence` at `share`, taken out of defending and nothing else."""
    out = copy.deepcopy(config)
    weights = out["positions"]["CB"]["sub_rating_weights"]
    weights["defending"] = _pool(weights) - share
    weights["team_defence"] = share
    return out


#: The centre-back setting the first stage chose, which the second builds on.
#:
#: `outcome-0.30` won both criteria: RPS 0.36463 against the baseline's 0.36508
#: (delta -0.00045, CI [-0.00063, -0.00027]) and the back line against next
#: season's goals conceded moving from -0.452 to -0.528. `outcome-0.45` reached
#: -0.548, nearly the -0.555 ceiling, and was significantly *worse* at
#: predicting, so there is a real optimum rather than "more is better".
CB_WINNER = 0.30


def _fb_variant(config: dict, cb_share: float, fb_share: float) -> dict:
    """The chosen centre-back setting, plus the same move on full backs."""
    out = _outcome_variant(config, cb_share)
    weights = out["positions"]["FB"]["sub_rating_weights"]
    weights["defending"] = _pool(weights) - fb_share
    weights["team_defence"] = fb_share
    return out


def variants(
    config: dict, stage: str = "cb", cb_share: float | None = None
) -> list[tuple[str, dict]]:
    """The runs for one stage. ``cb`` chooses a centre-back setting, ``fb`` extends it.

    Two stages rather than one grid, because the full-back question only makes
    sense once the centre-back one is settled: they share the defending
    definition exactly, so they share its defect, but a full back is judged 38%
    on defending against a centre back's 62% and the right amount of team
    outcome need not be the same.

    Args:
        config: The shipped configuration, which every variant is built from.
        stage: ``cb`` chooses a centre-back setting, ``fb`` extends it.
        cb_share: The centre-back setting the ``fb`` stage builds on. Defaults
            to :data:`CB_WINNER`. Passed explicitly when a re-run of the first
            stage chooses differently, so the second stage never silently
            extends a setting the current numbers did not pick.
    """
    if stage == "fb":
        chosen = CB_WINNER if cb_share is None else cb_share
        built = [("baseline", copy.deepcopy(config))]
        built.append((f"cb-{chosen:.2f}-only", _outcome_variant(config, chosen)))
        built += [
            (f"cb-{chosen:.2f}+fb-{f:.2f}", _fb_variant(config, chosen, f))
            for f in (0.15, 0.30)
        ]
        return built

    built = [("baseline", copy.deepcopy(config))]
    # 0.62 is the whole pool on defending: the setting that shipped before any
    # of this, measured explicitly so the table stands on its own.
    built += [
        (f"share-{d:.2f}", _share_variant(config, d)) for d in (0.62, 0.45, 0.30, 0.15, 0.00)
    ]
    built += [(f"outcome-{s:.2f}", _outcome_variant(config, s)) for s in (0.15, 0.30, 0.45)]
    return built


def next_season_defence(
    strength: pd.DataFrame, history: pd.DataFrame, min_rows: int = 50
) -> float:
    """Back-line strength against the goals its club concedes the season after.

    Next season rather than this one, because that is the only direction the
    model ever reads strength in -- `add_team_strength` joins season S onto
    season S+1 -- and because a same-season figure flatters any rating built
    from that season's own results, which is exactly what the `outcome`
    variant is.
    """
    outcomes = (
        history.assign(Season=history["Season"].astype(str))
        .groupby(["League", "Season", "Team"], observed=True)
        .agg(ga=("GoalsAgainst", "mean"), played=("GoalsAgainst", "size"))
        .reset_index()
    )
    outcomes = outcomes[outcomes["played"] >= 20]

    lhs = strength.assign(
        _next=strength["Season"].astype(str).map(_next_season),
        value=strength["strength_defence"],
    )[["League", "Team", "_next", "value"]].dropna(subset=["value"])

    merged = lhs.merge(
        outcomes.rename(columns={"Season": "_next"}), on=["League", "Team", "_next"], how="inner"
    )
    if len(merged) < min_rows:
        # Too few team-seasons to read a correlation from. A sweep that
        # silently reported one off a handful of clubs would be worse than
        # reporting nothing.
        return float("nan")
    return float(merged["value"].corr(merged["ga"]))


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Four seasons, for a smoke test.")
    parser.add_argument("--start-season", type=int, default=1516)
    parser.add_argument(
        "--stage", choices=("cb", "fb"), default="cb",
        help="cb chooses a centre-back setting; fb extends the winner to full backs.",
    )
    parser.add_argument(
        "--cb-share", type=float, default=None,
        help=f"The centre-back team_defence the fb stage builds on (default {CB_WINNER}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("soccerdata").setLevel(logging.ERROR)

    # Nothing here writes `team_strength.csv`, and that is deliberate.
    #
    # The first version of this swapped the production file per variant and
    # restored it in a `finally`. A `finally` does not survive the process
    # being killed, and when a timeout cut the smoke test short the shipped
    # file was left holding a variant's numbers -- 1,160 of 1,170 rows wrong,
    # by up to 5.3 points, with the API serving them.
    #
    # It was never needed. `build_match_features` takes the strength *frame*,
    # so the experiment can hold every variant in memory and the file on disk
    # can stay exactly as shipped.
    seasons = season_labels("1415", None)
    logger.info("Ingesting players across %d seasons", len(seasons))
    started = time.monotonic()
    players = fetch_player_seasons(None, seasons)
    logger.info(
        "Ingest done in %.0fs (%d player-seasons)", time.monotonic() - started, len(players)
    )

    tmh = load_team_match_history(TMH)
    history = pd.read_csv(TMH, low_memory=False)
    odds = load_raw_odds()
    config = load_config()

    results: list[dict] = []
    per_variant: dict[str, pd.DataFrame] = {}
    for name, variant in variants(config, stage=args.stage, cb_share=args.cb_share):
        weights = variant["positions"]["CB"]["sub_rating_weights"]
        logger.info(
            "--- %s: CB defending=%.2f team_defence=%.2f | FB team_defence=%.2f ---",
            name, weights["defending"], weights.get("team_defence", 0.0),
            variant["positions"]["FB"]["sub_rating_weights"].get("team_defence", 0.0),
        )
        rated = rate_players(players, config=variant)
        strength = team_strength(rated, method="best_n", best_n=15)

        features = build_match_features(tmh, strength=strength, european=AUTO)
        _, preds = run_backtest(
            features, start_season=1819 if args.quick else args.start_season,
            use_dixon_coles=True, odds=odds if not odds.empty else None, calibrate=True,
        )
        rps, accuracy = _score(preds)
        defence_r = next_season_defence(strength, history)

        cb = rated[(rated["role"] == "CB") & rated["rated"]]
        results.append({
            "variant": name,
            "cb_defending_weight": weights["defending"],
            "team_defence_weight": weights.get("team_defence", 0.0),
            "fb_team_defence": variant["positions"]["FB"]["sub_rating_weights"].get(
                "team_defence", 0.0
            ),
            "n": len(preds),
            "rps": float(rps.mean()),
            "accuracy": accuracy,
            "defence_vs_next_ga": defence_r,
            "mean_cb_rating": float(cb["rating"].mean()),
        })
        per_variant[name] = preds
    table = pd.DataFrame(results)
    base_rps, _ = _score(per_variant["baseline"])
    deltas = []
    for name in table["variant"]:
        if name == "baseline":
            deltas.append(
                {"variant": name, "delta": 0.0, "lo": 0.0, "hi": 0.0, "excludes_zero": False}
            )
            continue
        rps, _ = _score(per_variant[name])
        boot = paired_bootstrap(base_rps, rps)
        deltas.append({
            "variant": name, "delta": boot["delta"],
            "lo": boot["lo"], "hi": boot["hi"], "excludes_zero": boot["excludes_zero"],
        })

    table = table.merge(pd.DataFrame(deltas), on="variant")
    print("\n=== Defending sweep ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(
        "\nrps delta is variant minus baseline; negative means the variant predicts better."
        "\ndefence_vs_next_ga is the back line against the next season's goals conceded,"
        "\nso more negative is better. Shipped is about -0.489; the ceiling, a team's own"
        "\nprevious conceding, is -0.555."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
