"""Does picking a squad with no positional constraint distort team strength?

Production takes each club's **top 15 players by rating** and nothing else
(`method="best_n"`). There is no formation and no per-line minimum, so nothing
stops a club's fifteen being attack-heavy, leaving `strength_defence` to be
computed from whoever happens to remain.

The squad-size sweep (EXPERIMENTS.md) is what pointed here. Across 13-17 the
three next-season correlations disagreed in a specific way:

    best_n   atk>GF    def>GA    ovr>pts
    13       0.7325    -0.5196   0.7546
    15       0.7408    -0.5232   0.7562   <- shipped
    17       0.7331    -0.5316   0.7555

Attack peaks at 15 and falls away either side; defence keeps improving with
squad size and had not levelled off at 17. The natural reading is dilution: a
larger N adds marginal attackers to a line that only has a few real ones, while
the back line -- which clubs carry more of -- keeps gaining useful players. If
that is right, a per-line floor should buy the defensive gain *without* paying
the attacking cost.

**It may well not be right**, and that is the point of measuring. The opposite
result is just as plausible: forcing a club to field its fifth-best defender
over a better attacker makes the aggregate describe a team nobody picks.

## How the arms are built, and why it is safe

Selection happens *outside* `team_strength`, and the result is handed back to it
as ``method="squad"`` -- which aggregates whatever it is given. So every arm
runs production's exact line aggregation, weighting and gating, and the only
thing that varies is which fifteen players go in. Nothing in
`src/footballml/players/` is touched.

The baseline selects the top fifteen by rating, which must reproduce
``team_strength(method="best_n", best_n=15)`` byte for byte. That equality is
asserted at runtime: if it fails, the harness is wrong and no arm is readable.

## The bar, fixed here before the run

    ship        def>GA improves by >= 0.010, and neither atk>GF nor ovr>pts
                degrades by more than 0.005, and the domestic RPS upper CI
                bound is < +0.0005 with >= 6 of the 11 movable seasons
                neutral-or-better
    no-ship     anything else, including inconclusive

0.010 on def>GA is the whole 13-to-17 range from the squad sweep, so it asks a
quota to be worth as much as the entire squad-size axis. 0.005 is about half the
range of the other two. +0.0005 is the measurement floor established in
EXPERIMENTS.md, below which this pipeline cannot resolve a difference at all.

Run with::

    python -m pipelines.positional_quota_ab
    python -m pipelines.positional_quota_ab --quick
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
from footballml.features.build import AUTO, build_match_features  # noqa: E402
from footballml.ingest.understat import season_labels  # noqa: E402
from footballml.models.evaluate import paired_bootstrap, rps_per_match  # noqa: E402
from footballml.players.ingest import fetch_player_seasons  # noqa: E402
from footballml.players.rating import rate_players  # noqa: E402
from footballml.players.team_strength import (  # noqa: E402
    DEFAULT_BEST_N,
    LINE_BY_ROLE,
    team_strength,
)
from pipelines.backtest import load_raw_odds, run_backtest  # noqa: E402
from pipelines.squad_size_sweep import _season_outcomes, next_season_validity  # noqa: E402

logger = logging.getLogger("positional_quota_ab")

TMH = PROCESSED_DIR / "team_match_history_all.csv"
KEYS = ["League", "Season", "Team"]

#: Minimum players per line, by arm. ``None`` is production: no constraint.
#:
#: Three quotas, each with a reason, declared before the run:
#:
#: `formation` reserves a 4-3-3 and leaves four places free, so a club still
#: fields its four best remaining players whoever they are. It is the mildest
#: thing that can be called a quota, and it is the shape the documentation
#: wrongly described as shipped until 2026-10-18.
#:
#: `balanced` reserves nine and frees six -- half the squad constrained, half
#: not -- to separate "a floor helps" from "the 4-3-3 shape specifically helps".
#:
#: `full` allocates all fifteen, a notional squad with two keepers and cover in
#: every line. It is the strongest version, and the one most likely to force a
#: club to field a defender over a better attacker.
QUOTAS: dict[str, dict[str, int] | None] = {
    "production": None,
    "formation": {"goalkeeper": 1, "defence": 4, "midfield": 3, "attack": 3},
    "balanced": {"goalkeeper": 1, "defence": 3, "midfield": 3, "attack": 2},
    "full": {"goalkeeper": 2, "defence": 5, "midfield": 5, "attack": 3},
}

#: How much `def>GA` must improve to be worth a quota.
#:
#: The squad sweep moved it 0.012 across 13 to 17, so this asks a quota to be
#: worth about as much as the entire squad-size axis.
DEF_GAIN = 0.010

#: How much the other two correlations may degrade. About half their observed
#: range across the same sweep.
OTHER_TOLERANCE = 0.005

#: The floor below which this harness cannot resolve an RPS difference.
MARGIN = 0.0005

#: Test seasons that can move. The earliest trains with the strength features
#: absent and scores identically for every arm.
MOVABLE_SEASONS = 11


def select_squad(
    players: pd.DataFrame, quota: dict[str, int] | None, total: int = DEFAULT_BEST_N
) -> pd.DataFrame:
    """The ``total`` players a club contributes, honouring a per-line floor.

    Each line takes its quota of its own best first; the remaining places go to
    the best players left regardless of line. So a quota is a *floor*, never a
    cap -- a club whose six best players are defenders still fields them.

    A line with fewer rated players than its quota simply contributes what it
    has. Forcing the number would mean inventing players, and the thin-squad
    case is already handled downstream by `MIN_RATED_PLAYERS`.

    With ``quota=None`` this is the top ``total`` by rating, which is exactly
    what ``method="best_n"`` does.
    """
    ordered = players.sort_values("rating", ascending=False)
    if quota is None:
        return ordered.groupby(KEYS, observed=True).head(total)

    # Each line's own best first, up to its floor. `cumcount` on a frame already
    # sorted by rating gives each player's rank within his club's line, so the
    # whole reservation is one comparison rather than a per-group apply.
    rank_in_line = ordered.groupby([*KEYS, "line"], observed=True).cumcount()
    floor = ordered["line"].map(quota).fillna(0).astype(int)
    reserved_mask = rank_in_line < floor
    reserved = ordered[reserved_mask]

    # Then the best of whoever is left, regardless of line, until the squad is
    # full. A club that has fewer players in a line than its floor simply
    # contributes fewer; inventing them is not an option and `MIN_RATED_PLAYERS`
    # already handles a squad too thin to rate.
    rest = ordered[~reserved_mask]
    taken = reserved.groupby(KEYS, observed=True).size()
    taken_per_row = pd.Series(
        taken.reindex(pd.MultiIndex.from_frame(rest[KEYS])).fillna(0).to_numpy(),
        index=rest.index,
    )
    rank_in_club = rest.groupby(KEYS, observed=True).cumcount()
    extra = rest[rank_in_club < (total - taken_per_row)]
    return pd.concat([reserved, extra]).sort_index()


def with_lines(rated: pd.DataFrame) -> pd.DataFrame:
    """Rated players with the `line` column `team_strength` derives internally."""
    players = rated[rated["rated"] & rated["rating"].notna()].copy()
    group_col = "role" if "role" in players.columns else "position_group"
    players["line"] = players[group_col].map(LINE_BY_ROLE)
    return players.dropna(subset=["line"])


def _score(preds: pd.DataFrame) -> tuple[np.ndarray, float]:
    truth = preds["FTR"]
    probs = preds[["prob_H", "prob_D", "prob_A"]].to_numpy()
    actual = truth.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return rps_per_match(truth, probs), float((probs.argmax(axis=1) == actual).mean())


def _seasons_ok(keys: pd.Series, base: np.ndarray, arm: np.ndarray, minimum: int = 100) -> int:
    """Seasons where an arm is neutral-or-better, excluding those that cannot move.

    A fold every arm scores identically is not evidence that none made it worse.
    Counting it was a real bug in the extended-Elo run and turned a failing arm
    into a passing one.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Four seasons, for a smoke test.")
    parser.add_argument("--start-season", type=int, default=1516)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("soccerdata").setLevel(logging.ERROR)

    # Writes nothing: selection happens in memory and `build_match_features`
    # takes the strength frame, so no file on disk is touched.
    seasons = season_labels("1415", None)
    logger.info("Ingesting players across %d seasons", len(seasons))
    players_raw = fetch_player_seasons(None, seasons)
    rated = rate_players(players_raw)
    players = with_lines(rated)
    logger.info("Rated once for every arm: %d players carry a line", len(players))

    tmh = load_team_match_history(TMH)
    outcomes = _season_outcomes(pd.read_csv(TMH, low_memory=False))
    odds = load_raw_odds()

    print("\nLine composition of the shipped selection (top 15, no quota):")
    base_pick = select_squad(players, None)
    comp = base_pick.groupby("line", observed=True).size() / base_pick.groupby(
        KEYS, observed=True
    ).ngroups
    print(comp.round(2).to_string())

    results: list[dict] = []
    vectors: dict[str, np.ndarray] = {}
    season_of: pd.Series | None = None

    for name, quota in QUOTAS.items():
        logger.info("--- %s: quota=%s ---", name, quota)
        started = time.monotonic()
        picked = select_squad(players, quota)
        strength = team_strength(picked, method="squad")

        if name == "production":
            # The harness must reproduce what ships, or nothing below is
            # readable. Selecting the top fifteen and aggregating them as a
            # "squad" has to equal `best_n` exactly.
            shipped = team_strength(players, method="best_n", best_n=DEFAULT_BEST_N)
            merged = shipped.merge(strength, on=KEYS, suffixes=("_s", "_h"))
            for column in ("strength_overall", "strength_defence", "strength_attack"):
                gap = (merged[f"{column}_s"] - merged[f"{column}_h"]).abs().max()
                assert gap < 1e-9, f"harness does not reproduce production: {column} off by {gap}"
            print(f"  harness reproduces method='best_n' exactly ({len(merged)} team-seasons)")

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

        lines = picked.groupby("line", observed=True).size()
        per_club = picked.groupby(KEYS, observed=True).size()
        results.append({
            "arm": name,
            "mean_squad": round(float(per_club.mean()), 2),
            "pct_defence": round(float(lines.get("defence", 0) / len(picked)), 3),
            "pct_attack": round(float(lines.get("attack", 0) / len(picked)), 3),
            "rps": float(rps.mean()),
            "accuracy": accuracy,
            **next_season_validity(strength, outcomes),
            "seconds": round(time.monotonic() - started),
        })

    table = pd.DataFrame(results)
    base = vectors["production"]
    rows = []
    for name in table["arm"]:
        if name == "production":
            rows.append({"arm": name, "delta": 0.0, "lo": 0.0, "hi": 0.0, "sig": False})
            continue
        boot = paired_bootstrap(base, vectors[name])
        rows.append({"arm": name, "delta": boot["delta"], "lo": boot["lo"],
                     "hi": boot["hi"], "sig": boot["excludes_zero"]})
    table = table.merge(pd.DataFrame(rows), on="arm")

    print("\n=== Positional quotas ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    assert season_of is not None
    ref = table[table["arm"] == "production"].iloc[0]
    print("\n=== Verdict against the criteria declared before the run ===")
    for _, row in table.iterrows():
        if row["arm"] == "production":
            continue
        # def>GA is negative, so an improvement is a *decrease*.
        def_gain = float(ref["def_vs_ga"]) - float(row["def_vs_ga"])
        atk_drop = float(ref["atk_vs_gf"]) - float(row["atk_vs_gf"])
        ovr_drop = float(ref["ovr_vs_pts"]) - float(row["ovr_vs_pts"])
        ok_seasons = _seasons_ok(season_of, base, vectors[row["arm"]])
        passes = (
            def_gain >= DEF_GAIN
            and atk_drop <= OTHER_TOLERANCE
            and ovr_drop <= OTHER_TOLERANCE
            and float(row["hi"]) < MARGIN
            and ok_seasons >= 6
        )
        print(
            f"  {row['arm']:11s} def gain {def_gain:+.4f} (>= {DEF_GAIN}? "
            f"{'yes' if def_gain >= DEF_GAIN else 'NO'})  "
            f"atk cost {atk_drop:+.4f} ovr cost {ovr_drop:+.4f} "
            f"(<= {OTHER_TOLERANCE}? "
            f"{'yes' if max(atk_drop, ovr_drop) <= OTHER_TOLERANCE else 'NO'})  "
            f"RPS hi {float(row['hi']):+.5f}  seasons {ok_seasons}/{MOVABLE_SEASONS}  "
            f"-> {'MEETS the bar' if passes else 'does NOT meet the bar'}"
        )
    print(
        "\natk_vs_gf and ovr_vs_pts are positive (higher is better); def_vs_ga is"
        "\nnegative (lower is better), so its 'gain' is a decrease. Nothing here"
        "\nwrites to data/processed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
