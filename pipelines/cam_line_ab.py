"""Does counting an attacking midfielder as midfield describe a team better?

The question and the first measurement are in
`docs/PROPOSAL-cam-line-mapping.md`. This harness is the second half of it: the
proposal measured the line arrangement with ratings held fixed, and left two
things unmeasured that decide whether the change can ship at all.

**What the proposal missed.** ``role`` is not only the line a player is
reported under -- it is the *pool he is rated in*. `rate_players` computes
every percentile within ``(role, Season)``, so moving `CAM` out of `AMW`
re-ranks both the players who move and the wingers they leave behind. A
measurement that relabels the line and holds the ratings fixed answers a
narrower question than the change actually poses.

So five arms, in two pairs that must not be crossed:

``production`` / ``line_only``
    Shipped ratings, read from `player_ratings.csv` exactly as the site serves
    them. ``line_only`` moves `CAM`-primary players to the midfield line and
    changes nothing else. This is the proposal's arm, re-run through the
    production selection and the production feature pipeline.

``rerated`` / ``cam_to_mid`` / ``cam_new_role``
    Ratings recomputed from the same file, so the baseline carries the same
    percentile jitter as the arms and the pairing is exact. ``cam_to_mid``
    maps `CAM` onto the existing `MID` role -- MID's pool, MID's yardstick.
    ``cam_new_role`` gives it a role of its own with `AMW`'s yardstick, which
    is the precedent `DM` set when it was split out of `MID`: *"the same
    metrics and the same weights as MID, deliberately ... Separate the
    players, not the yardstick."*

`cam_new_role` is what the proposal proposed. `cam_to_mid` is the only version
that needs no new configuration, so it is the cheap one to ship if it does as
well.

## Why it reads the built file instead of re-ingesting

Every other A/B here calls `fetch_player_seasons` and rates from scratch.
Understat has served a holding page since 2 November, so that path cannot run,
and `player_ratings.csv` is frozen at the 2 November vintage -- which is the
vintage the proposal measured, so using it keeps the two documents comparable.

The file is an input, never an output: this writes nothing to
`data/processed`. Two fidelity checks make that safe to lean on, both asserted
at runtime rather than claimed:

1. Re-deriving ``role`` from ``fifa_positions`` through the production chain
   (`primary_role` -> `_carry_role_across_seasons` -> `FALLBACK_ROLE` ->
   `_settle_reordered_roles`) must reproduce the shipped ``role`` column for
   every one of the 34,856 rows. If it does not, the arms differ from
   production by something other than the mapping and nothing below is
   readable.
2. The ``production`` arm must reproduce `team_strength.csv` on disk.

Re-rating is faithful but not bit-exact: 10 of 24,669 ratings come back one
point different, from percentile ties resolving the other way after a CSV
round-trip. That is why `rerated` exists as its own baseline -- the three
re-rated arms share the jitter, so their deltas do not contain it.

## The bar, fixed here before the run

`team_strength.csv` is a model input (`features/build.py`, `add_team_strength`),
so this is not a purely published number and RPS is not a tiebreak:

    ship         mid>pts improves by >= 0.003 (the correlation floor), and
                 neither atk>GF nor ovr>pts degrades by more than 0.005, and
                 the pooled RPS interval does not exclude zero on the worse
                 side
    no-ship      anything else, including inconclusive

The squad family is 12 of 235 features and removing it *entirely* is worth
+0.00015 RPS [-0.00001, +0.00032], which does not clear zero. So RPS cannot be
expected to improve here, and a movement comfortably past the 0.0005 floor
would be evidence of a mistake rather than a result.

Run with::

    python -m pipelines.cam_line_ab
    python -m pipelines.cam_line_ab --quick
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import PROCESSED_DIR, load_team_match_history  # noqa: E402
from footballml.features.build import AUTO, _next_season, build_match_features  # noqa: E402
from footballml.models.evaluate import paired_bootstrap, rps_per_match  # noqa: E402
from footballml.players import fifa as F  # noqa: E402
from footballml.players.rating import load_config, rate_players  # noqa: E402
from footballml.players.team_strength import (  # noqa: E402
    DEFAULT_BEST_N,
    LINE_BY_ROLE,
    team_strength,
)
from pipelines.backtest import load_raw_odds, run_backtest  # noqa: E402
from pipelines.squad_size_sweep import _season_outcomes, next_season_validity  # noqa: E402

logger = logging.getLogger("cam_line_ab")

RATINGS = PROCESSED_DIR / "player_ratings.csv"
STRENGTH = PROCESSED_DIR / "team_strength.csv"
TMH = PROCESSED_DIR / "team_match_history_all.csv"
KEYS = ["League", "Season", "Team"]

#: The role a new attacking-midfield pool would carry.
NEW_ROLE = "AM"

#: Columns `rate_players` produces. Dropped before re-rating so nothing is
#: carried over from the shipped file; everything else in it is an input.
_RATING_OUTPUTS = (
    "rated",
    "unrated_reason",
    "composite_raw",
    "composite",
    "performance_rating",
    "fifa_on_our_scale",
    "measured_share",
    "fifa_weight_used",
    "rating",
    "listed_position",
)

#: How much `mid > points` must gain to count. The measured floor on these
#: correlations, below which this comparison cannot resolve a difference.
MID_GAIN = 0.003

#: How much `atk > GF` or `ovr > points` may degrade.
OTHER_TOLERANCE = 0.005

#: The floor below which the backtest cannot resolve an RPS difference.
MARGIN = 0.0005


def primary_position(positions: object) -> str | None:
    """EA's first-listed position, which is how `primary_role` reads them."""
    if not isinstance(positions, str):
        return None
    head = positions.replace("|", ",").split(",")[0].strip().upper()
    return head or None


def derive_roles(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    """``role`` as production derives it, under an arbitrary position mapping.

    Replicates `attach_fifa`'s four steps in its order. Each is production's
    own function, so the only thing that varies between arms is ``mapping``.
    """

    def primary(positions: object) -> str | None:
        if not isinstance(positions, str):
            return None
        for token in positions.replace("|", ",").split(","):
            role = mapping.get(token.strip().upper())
            if role:
                return role
        return None

    work = frame.copy()
    work["role"] = work["fifa_positions"].map(primary)
    work = F._carry_role_across_seasons(work)
    fallback = work["position_group"]
    if "season_position" in work.columns:
        fallback = work["season_position"].fillna(fallback)
    work["role"] = work["role"].fillna(fallback.map(F.FALLBACK_ROLE))
    work = F._settle_reordered_roles(work)
    return work["role"]


@contextmanager
def role_registered(role: str, line: str):
    """Make `role` a first-class outfield role for the duration.

    Three registrations, because production reads three different globals and
    omitting any one of them fails quietly rather than loudly:

    ``LINE_BY_ROLE``
        Which line it is reported and aggregated under. Missing: every player
        in the role is dropped by `team_strength`'s ``dropna(subset=["line"])``.
    ``_OUTFIELD`` / ``COMPATIBLE_ROLES``
        Whether an EA entry in this role can describe the same person as an
        Understat row. Missing: `contradicts_position` rejects the role for
        every outfield group, which is the 92%-to-87% failure recorded beside
        the set itself.
    ``ROLE_LABELS``
        What a reader is shown.
    """
    saved = (
        dict(LINE_BY_ROLE),
        set(F._OUTFIELD),
        copy.deepcopy(F.COMPATIBLE_ROLES),
        dict(F.ROLE_LABELS),
    )
    LINE_BY_ROLE[role] = line
    F._OUTFIELD.add(role)
    F.COMPATIBLE_ROLES["D"] = F._OUTFIELD
    F.COMPATIBLE_ROLES["M"] = F._OUTFIELD
    F.COMPATIBLE_ROLES["F"] = F._OUTFIELD - {"CB"}
    F.ROLE_LABELS[role] = "Attacking midfielder"
    try:
        yield
    finally:
        LINE_BY_ROLE.clear()
        LINE_BY_ROLE.update(saved[0])
        F._OUTFIELD.clear()
        F._OUTFIELD.update(saved[1])
        F.COMPATIBLE_ROLES.clear()
        F.COMPATIBLE_ROLES.update(saved[2])
        F.ROLE_LABELS.clear()
        F.ROLE_LABELS.update(saved[3])


def config_with_new_role(role: str, like: str = "AMW") -> dict:
    """The rating config plus `role`, scored on `like`'s yardstick.

    `DM` is the precedent and it is explicit about this: it was split out of
    `MID` with *"the same metrics and the same weights ... Separate the
    players, not the yardstick."* Inventing weights for a new pool is a
    separate change with its own sweep.
    """
    config = copy.deepcopy(load_config())
    spec = copy.deepcopy(config["positions"][like])
    spec["label"] = "Attacking midfielder"
    config["positions"][role] = spec
    return config


def strength_of(rated: pd.DataFrame) -> pd.DataFrame:
    """Production's team rating: top 15 by rating, aggregated by line."""
    return team_strength(rated, method="best_n", best_n=DEFAULT_BEST_N)


def mid_validity(strength: pd.DataFrame, outcomes: pd.DataFrame) -> float:
    """The midfield line against the club's points the *following* season.

    `next_season_validity` covers attack, defence and overall. Midfield has no
    outcome column of its own -- there is no "midfield goals" -- so points is
    the only thing it plausibly predicts, and this is the metric the proposal
    nominated for itself.
    """
    if "strength_midfield" not in strength.columns:
        return float("nan")
    lhs = strength.assign(
        _next=strength["Season"].astype(str).map(_next_season),
        value=strength["strength_midfield"],
    )[["League", "Team", "_next", "value"]].dropna(subset=["value"])
    merged = lhs.merge(
        outcomes.rename(columns={"Season": "_next"}),
        on=["League", "Team", "_next"],
        how="inner",
    )
    return float(merged["value"].corr(merged["pts"])) if len(merged) >= 50 else float("nan")


def _score(preds: pd.DataFrame) -> tuple[np.ndarray, float]:
    # The raw H/D/A column, not an integer encoding: `_one_hot` compares against
    # the strings, so integers silently one-hot to all zeros.
    truth = preds["FTR"]
    probs = preds[["prob_H", "prob_D", "prob_A"]].to_numpy()
    actual = truth.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return rps_per_match(truth, probs), float((probs.argmax(axis=1) == actual).mean())


def _movement(base: pd.DataFrame, arm: pd.DataFrame) -> pd.DataFrame:
    """How far each line's rating moves, over the team-seasons both rate."""
    lines = ["strength_midfield", "strength_attack", "strength_defence", "strength_overall"]
    merged = base.merge(arm, on=KEYS, suffixes=("_b", "_a"))
    rows = []
    for line in lines:
        if f"{line}_b" not in merged.columns:
            continue
        delta = (merged[f"{line}_a"] - merged[f"{line}_b"]).dropna()
        moved = delta[delta.abs() > 1e-9]
        rows.append(
            {
                "line": line.replace("strength_", ""),
                "rated": len(delta),
                "moved": len(moved),
                "pct_moved": round(len(moved) / len(delta), 3) if len(delta) else float("nan"),
                "mean": round(float(delta.mean()), 3),
                "mean_abs": round(float(delta.abs().mean()), 3),
                "max_up": round(float(delta.max()), 2),
                "max_down": round(float(delta.min()), 2),
            }
        )
    return pd.DataFrame(rows)


def _biggest_movers(
    base: pd.DataFrame, arm: pd.DataFrame, season: str, n: int = 12
) -> pd.DataFrame:
    merged = base.merge(arm, on=KEYS, suffixes=("_b", "_a"))
    merged = merged[merged["Season"].astype(str) == season].copy()
    merged["mid_delta"] = merged["strength_midfield_a"] - merged["strength_midfield_b"]
    merged["atk_delta"] = merged["strength_attack_a"] - merged["strength_attack_b"]
    merged["ovr_delta"] = merged["strength_overall_a"] - merged["strength_overall_b"]
    cols = [
        "Team",
        "League",
        "strength_midfield_b",
        "strength_midfield_a",
        "mid_delta",
        "atk_delta",
        "ovr_delta",
    ]
    order = merged["mid_delta"].abs().sort_values(ascending=False).index
    return merged.reindex(order)[cols].head(n)


def _seasons_ok(keys: pd.Series, base: np.ndarray, arm: np.ndarray, minimum: int = 100) -> int:
    """Seasons where an arm is neutral-or-better, ignoring those that cannot move.

    A fold every arm scores identically is not evidence that none made it
    worse; counting it once turned a failing arm into a passing one.
    """
    ok = 0
    for key in sorted(keys.unique()):
        mask = (keys == key).to_numpy()
        if mask.sum() < minimum:
            continue
        delta = float((arm[mask] - base[mask]).mean())
        if delta == 0.0:
            continue
        ok += delta <= 0
    return ok


def build_arms(shipped: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every arm's rated-player frame, built from the one shipped file."""
    inputs = shipped.drop(columns=[c for c in _RATING_OUTPUTS if c in shipped.columns])

    # Fidelity check 1: the role chain reproduces what shipped, exactly.
    reproduced = derive_roles(inputs, F.ROLE_BY_POSITION)
    disagree = int((reproduced.fillna("~") != shipped["role"].fillna("~")).sum())
    assert disagree == 0, f"role chain does not reproduce production: {disagree} rows differ"
    logger.info("Role chain reproduces the shipped column for all %d rows", len(shipped))

    arms: dict[str, pd.DataFrame] = {"production": shipped}

    # Arm: the line moves, the ratings do not. A proxy role carrying the same
    # line is enough -- `team_strength` reads only the line, and re-pooling is
    # exactly what this arm is built to exclude.
    primary = shipped["fifa_positions"].map(primary_position)
    cam = (shipped["role"] == "AMW") & (primary == "CAM")
    logger.info(
        "CAM-primary AMW: %d player-seasons, %d of them rated",
        int(cam.sum()),
        int((cam & shipped["rated"]).sum()),
    )
    line_only = shipped.copy()
    line_only.loc[cam, "role"] = NEW_ROLE
    arms["line_only"] = line_only

    # Arms that re-rate. The baseline re-rates under the unmodified mapping, so
    # it carries the same percentile jitter as the two treatments.
    arms["rerated"] = rate_players(inputs.assign(role=reproduced))

    to_mid = dict(F.ROLE_BY_POSITION)
    to_mid["CAM"] = "MID"
    arms["cam_to_mid"] = rate_players(inputs.assign(role=derive_roles(inputs, to_mid)))

    new_role = dict(F.ROLE_BY_POSITION)
    new_role["CAM"] = NEW_ROLE
    arms["cam_new_role"] = rate_players(
        inputs.assign(role=derive_roles(inputs, new_role)),
        config=config_with_new_role(NEW_ROLE),
    )
    return arms


#: Each arm's baseline, for the paired comparisons. Crossing the pairs would
#: fold the re-rating jitter into the line-only delta.
PAIRS = {
    "line_only": "production",
    "cam_to_mid": "rerated",
    "cam_new_role": "rerated",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Four seasons, for a smoke test.")
    parser.add_argument("--start-season", type=int, default=1516)
    parser.add_argument("--no-backtest", action="store_true", help="Ratings and correlations only.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("footballml.players.rating", "footballml.players.fifa"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    shipped = pd.read_csv(RATINGS, low_memory=False)
    logger.info("Read %d player-seasons from %s", len(shipped), RATINGS.name)

    # The new role has to be registered for every arm, not only its own: the
    # line lookup and the compatibility set are read at call time, and a
    # registration that lapsed between building the arm and aggregating it
    # would silently drop every CAM from the team rating.
    with role_registered(NEW_ROLE, "midfield"):
        arms = build_arms(shipped)
        strengths = {name: strength_of(frame) for name, frame in arms.items()}

    # Fidelity check 2: the production arm is what the site serves.
    on_disk = pd.read_csv(STRENGTH)
    on_disk["Season"] = on_disk["Season"].astype(str)
    mine = strengths["production"].copy()
    mine["Season"] = mine["Season"].astype(str)
    merged = on_disk.merge(mine, on=KEYS, suffixes=("_d", "_m"))
    assert len(merged) == len(on_disk), f"team-season mismatch: {len(merged)} of {len(on_disk)}"
    for column in ("strength_overall", "strength_attack", "strength_defence", "strength_midfield"):
        gap = (merged[f"{column}_d"] - merged[f"{column}_m"]).abs().max()
        assert gap < 1e-9, f"production arm does not reproduce {column}: off by {gap}"
    print(f"\nProduction arm reproduces team_strength.csv exactly ({len(merged)} team-seasons)")

    print("\n=== How the ratings move, against each arm's own baseline ===")
    for arm, base in PAIRS.items():
        print(f"\n{arm} vs {base}")
        print(_movement(strengths[base], strengths[arm]).to_string(index=False))

    latest = str(on_disk["Season"].max())
    for arm, base in PAIRS.items():
        print(f"\n=== Largest midfield movers, {latest}: {arm} vs {base} ===")
        print(
            _biggest_movers(strengths[base], strengths[arm], latest).to_string(
                index=False, float_format=lambda v: f"{v:+.2f}"
            )
        )

    outcomes = _season_outcomes(pd.read_csv(TMH, low_memory=False))
    rows = []
    for name, strength in strengths.items():
        rows.append(
            {
                "arm": name,
                "team_seasons": len(strength),
                "mid_vs_pts": mid_validity(strength, outcomes),
                **next_season_validity(strength, outcomes),
            }
        )
    correlations = pd.DataFrame(rows)
    print("\n=== Against the following season's outcome ===")
    print(correlations.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.no_backtest:
        print("\n--no-backtest: RPS not measured.")
        return 0

    tmh = load_team_match_history(TMH)
    odds = load_raw_odds()
    vectors: dict[str, np.ndarray] = {}
    season_of: pd.Series | None = None
    scores: dict[str, tuple[float, float, int]] = {}
    for name, strength in strengths.items():
        started = time.monotonic()
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
        scores[name] = (float(rps.mean()), accuracy, round(time.monotonic() - started))
        if season_of is None:
            season_of = preds["Season"].reset_index(drop=True).astype(str)
        logger.info("%s: RPS %.5f over %d fixtures", name, rps.mean(), len(rps))

    assert season_of is not None
    table = []
    for name in strengths:
        rps, accuracy, seconds = scores[name]
        row = {"arm": name, "rps": rps, "accuracy": accuracy, "seconds": seconds}
        base = PAIRS.get(name)
        if base is None:
            row.update({"vs": "-", "delta": 0.0, "lo": 0.0, "hi": 0.0, "sig": False})
        else:
            boot = paired_bootstrap(vectors[base], vectors[name])
            row.update(
                {
                    "vs": base,
                    "delta": boot["delta"],
                    "lo": boot["lo"],
                    "hi": boot["hi"],
                    "sig": boot["excludes_zero"],
                }
            )
        table.append(row)
    print(
        f"\n=== Walk-forward RPS, {len(vectors['production'])} fixtures "
        "(negative delta = better) ==="
    )
    print(pd.DataFrame(table).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== Verdict against the bar declared before the run ===")
    for arm, base in PAIRS.items():
        ref = correlations[correlations["arm"] == base].iloc[0]
        row = correlations[correlations["arm"] == arm].iloc[0]
        mid_gain = float(row["mid_vs_pts"]) - float(ref["mid_vs_pts"])
        atk_cost = float(ref["atk_vs_gf"]) - float(row["atk_vs_gf"])
        ovr_cost = float(ref["ovr_vs_pts"]) - float(row["ovr_vs_pts"])
        boot = paired_bootstrap(vectors[base], vectors[arm])
        worse = boot["lo"] > 0
        ok_seasons = _seasons_ok(season_of, vectors[base], vectors[arm])
        passes = (
            mid_gain >= MID_GAIN
            and atk_cost <= OTHER_TOLERANCE
            and ovr_cost <= OTHER_TOLERANCE
            and not worse
        )
        print(
            f"  {arm:13s} vs {base:11s} mid {mid_gain:+.4f} (>= {MID_GAIN}? "
            f"{'yes' if mid_gain >= MID_GAIN else 'NO'})  "
            f"atk cost {atk_cost:+.4f} ovr cost {ovr_cost:+.4f} "
            f"(<= {OTHER_TOLERANCE}? "
            f"{'yes' if max(atk_cost, ovr_cost) <= OTHER_TOLERANCE else 'NO'})  "
            f"RPS {boot['delta']:+.5f} [{boot['lo']:+.5f}, {boot['hi']:+.5f}]"
            f"{' WORSE' if worse else ''}  seasons {ok_seasons}  "
            f"-> {'MEETS the bar' if passes else 'does NOT meet the bar'}"
        )
    print(
        f"\nThe RPS floor is {MARGIN}; the squad family is worth +0.00015 entire, so a"
        "\nmovement well past the floor is a bug signal rather than a result."
        "\nNothing here writes to data/processed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
