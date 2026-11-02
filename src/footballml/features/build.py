"""Assemble the match-level, pre-kickoff feature table.

Form is computed on the long team-match shape (see
:mod:`footballml.features.rolling`), then pivoted here into one row per match
with ``home_``/``away_`` prefixed columns plus a handful of differences.

Differences matter more than you'd expect: tree models can in principle learn
``home_x - away_x`` themselves, but handing it over directly consistently helps,
because the *relative* strength of the two sides is what actually decides a
football match.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from footballml.data import load_european_matches
from footballml.entities import load_aliases
from footballml.features.elo import add_elo
from footballml.features.rolling import (
    add_rest_features,
    add_team_form,
    prepare_team_match,
)

logger = logging.getLogger(__name__)

#: Form stems differenced per window. ``_venue`` variants are split out because
#: only some metrics are computed home/away as well as overall.
_WINDOWED_STEMS: tuple[str, ...] = (
    "points_last_{w}",
    "points_last_{w}_venue",
    "goal_diff_last_{w}",
    "goal_diff_last_{w}_venue",
    "goals_for_last_{w}",
    "goals_against_last_{w}",
    "shots_on_target_for_last_{w}",
    "shot_accuracy_last_{w}",
    "xg_for_last_{w}",
    "xg_against_last_{w}",
    "xg_diff_last_{w}",
    "xg_diff_last_{w}_venue",
    "xg_overperformance_last_{w}",
    "npxg_diff_last_{w}",
    "ppda_last_{w}",
)

#: Stems that do not depend on a rolling window.
_FIXED_STEMS: tuple[str, ...] = (
    "days_since_last_match",
    "strength_overall",
    "strength_attack",
    "strength_defence",
    "strength_goalkeeper",
    "elo",
)

#: Rolling windows, in matches.
#:
#: Five alone made the model timid. It could not tell a genuinely elite side
#: from one that had won three of five, so it shrank every prediction toward
#: 50%: where the market said 80%+, it said 78% and the home team won 87%. Over
#: a season that compounds -- simulated tables spread 9.7 points against a real
#: 17.4, with the champion on 72 where reality averages 89.
#:
#: 19 is half a season: long enough to express persistent quality, short enough
#: to move when a team genuinely changes. Windows roll over a team's matches in
#: date order and so carry across the summer, which is deliberate -- a side's
#: level is fairly stable year to year, and without it August predictions would
#: have no history at all. The model is free to weight the long window down.
DEFAULT_WINDOWS: tuple[int, ...] = (5, 19)


def diff_stems(windows: Sequence[int] = DEFAULT_WINDOWS) -> tuple[str, ...]:
    """Stems to difference, for these rolling windows.

    Generated rather than hardcoded: a new window otherwise produces
    ``home_points_last_19`` with no ``points_last_19_diff`` beside it, and the
    relative strength of the two sides is what actually decides a match.
    """
    windowed = [stem.format(w=w) for w in windows for stem in _WINDOWED_STEMS]
    return tuple(windowed) + _FIXED_STEMS


#: Differenced stems for the default windows, kept for callers that want the
#: module-level constant.
DIFF_STEMS: tuple[str, ...] = diff_stems()

#: Squad-strength columns joined onto each match.
#:
#: The goalkeeper is carried separately rather than folded into the defence
#: line, because he is the best single defensive signal we have: across 1,052
#: completed team-seasons his rating tracks goals conceded at -0.62, against
#: -0.52 for the whole back line. Blending him into `strength_defence` would
#: also double-count him, since he already contributes to `strength_overall`.
STRENGTH_COLUMNS = (
    "strength_overall",
    "strength_attack",
    "strength_defence",
    "strength_goalkeeper",
)

#: `squad_strength.csv`'s columns under the names the measured table publishes.
#:
#: Both tables rate the same four lines on the same 0-99 scale -- the pre-season
#: rating is built by handing EA's September roster to the same
#: `team_strength` aggregation, with each player carrying his most recent rating
#: from *before* this season -- so one can fill a hole in the other without
#: rescaling. `squad_midfield` has no measured counterpart in
#: :data:`STRENGTH_COLUMNS` and is therefore not carried.
PRESEASON_COLUMNS: dict[str, str] = {
    "squad_overall": "strength_overall",
    "squad_attack": "strength_attack",
    "squad_defence": "strength_defence",
    "squad_goalkeeper": "strength_goalkeeper",
}

#: Stem of the flag that records where a side's strength came from.
#:
#: ``1.0`` where it is the pre-season rating, ``0.0`` where it is the previous
#: season's measured rating *or* where there is nothing at all. The value column
#: already separates those last two -- one is a number, the other NaN -- so the
#: flag does not need a third state, and a NaN in a boolean would only cost the
#: booster a split.
#:
#: Why a flag rather than two separate families of columns: the two ratings
#: answer the same question on the same scale but with different error -- one is
#: measured from a season of football, the other projected from a roster -- so
#: a single split threshold does mean slightly different things on either side
#: of it. The flag is what lets the booster find that out. Keeping them apart
#: instead was the alternative: it would leave the measured columns NaN on 14%
#: of team-sides and add four columns populated on only 14%, which is the shape
#: a gradient booster handles worst -- too few rows in any leaf to earn a split,
#: for exactly the clubs the fallback exists to cover.
PRESEASON_FLAG = "strength_is_preseason"

#: Internal marker column: a side matched a row in the table being joined.
#: Dropped before the frame is returned.
_MATCHED = "_matched"

#: Columns identifying a match rather than describing it.
_ID_COLS = ("League", "Season", "Date", "HomeTeam", "AwayTeam")

#: Stable numeric codes for the league feature. Scoring rates and home advantage
#: genuinely differ between leagues -- the Bundesliga is higher scoring than
#: Serie A -- so the model is told which competition it is looking at. Codes are
#: fixed rather than derived from the data so that a model trained on four
#: leagues still interprets the feature correctly when scoring a fifth.
LEAGUE_CODES = {"E0": 0, "SP1": 1, "D1": 2, "I1": 3, "F1": 4}


def add_team_strength(
    matches: pd.DataFrame,
    strength: pd.DataFrame,
    previous_season: bool = True,
    preseason: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach squad strength to each match, from the player ratings.

    Three numbers per side -- overall, attack and defence -- which map onto the
    model's two goal rates directly: a side's attack is set against the
    opposition's defence.

    Args:
        matches: Wide match table with ``League``, ``Season`` and both teams.
        strength: ``team_strength.csv``, one row per team-season.
        previous_season: Join the *preceding* season's strength rather than the
            match's own. This is the leakage guard and the default.
        preseason: Optional ``squad_strength.csv``. Where a side has no measured
            strength, its **own** season's pre-season rating fills the hole and
            ``home_``/``away_strength_is_preseason`` records that it did. Off by
            default: the columns are absent unless a caller passes this, so the
            feature-name guard on a trained artifact stays honest. See
            :data:`PRESEASON_COLUMNS` and the note below.

    Returns:
        ``matches`` with ``home_``/``away_`` strength columns. Teams with no
        strength row get NaN, which the gradient booster handles natively.

    .. warning::
        Strength is computed from **whole-season** player ratings, so joining a
        match to its own season lets end-of-season information predict an
        October fixture. That leak is invisible to the truncation test in
        ``tests/test_leakage.py``: it truncates *match* data, and strength comes
        from a separate file that would not change.

        ``previous_season=True`` is therefore correct for training and
        backtesting -- and for live prediction too, for two reasons that each
        suffice on their own:

        - **Current-season strength does not exist when it is needed.** Ratings
          require a minimum number of minutes, so four weeks into 2026/27 not
          one player was rated and ``team_strength.csv`` had no rows for it.
          ``False`` would hand every live fixture all-NaN strength, silently
          identical to not using the feature.
        - **Train/serve consistency.** The model learns "this match, given last
          season's squad". Serving it this season's squad is a different
          quantity, and skew even where the data exists.

        ``False`` remains for analysis only. A promoted side has no
        previous-season row in its new league and gets NaN; the model was
        trained with exactly those gaps, so it degrades rather than breaks.

    .. note:: ``preseason`` was measured and does **not** ship. Off by default.

        The short version: the gap is real and the fallback fills 99.3% of it
        with a value that genuinely carries signal, and the match model still
        cannot use it. Pooled RPS moved **-0.00001 [-0.00022, +0.00021]** on
        20,194 walk-forward fixtures; on the 5,375 fixtures it can actually
        touch, **-0.00020 [-0.00067, +0.00027]**. Both straddle zero, it was
        neutral-or-better in 5 of 12 seasons, and the secondary criterion
        improved on 2 of 3 lines. The default is the status quo, so it stays
        off. `pipelines/preseason_fallback_ab.py` and the EXPERIMENTS.md
        section "Should the model fall back to the pre-season squad rating?"
        carry the full numbers and the criteria, which were fixed before the
        run. The parameter remains because the alternative had to be measured,
        and a reader who wonders about this gap should find the answer rather
        than re-run it.

        The rest of this note is why it was worth measuring, and why the
        measurement is trustworthy.

        The gap it fills is not small: across 1516-2627, counting team-sides
        whose season *and* its predecessor are inside Understat's coverage,
        **5,674 of 40,334 (14.1%)** have no previous-season measured strength,
        steady at 13-15% every season. They are not random -- Metz, Frosinone,
        Troyes, Norwich, Burnley, Fulham -- they are promoted and yo-yo sides,
        which the measured rating *structurally* cannot reach, because a club
        that was in Ligue 2 last season has no top-flight season to measure.
        The pre-season rating covers **5,636 of them, 99.3%**.

        And the values are not noise. On the 186 team-seasons it fills that
        have a comparable outcome, the pre-season rating points the right way
        on all three lines -- attack against goals for **+0.63**, defence
        against goals against **-0.51**, overall against points **+0.69**.
        Weaker than the measured rating on its own population (+0.74, -0.53,
        +0.76), which is what a promoted side should look like, but clearly
        informative. That is also why the *pooled* attack correlation falls by
        0.0018 when the fallback is switched on: the rows it adds are a harder
        population, not a wrong one, and the pooled number cannot tell those
        two apart. So the honest reading of the result is not "the pre-season
        rating is bad" but "the model already knows this from Elo and from the
        19-match form window, which carries across the summer."

        Both objections above are against joining the match's **own season's
        measured** strength, and the pre-season rating fails neither:

        - **It exists when it is needed.** Measured strength needs a minimum
          number of minutes, so it does not exist in August. The pre-season
          rating needs none: it is EA's September roster with each player
          carrying his most recent rating from a season *strictly before* this
          one (``squad_rating._latest_before``). It is on disk for 2026/27 now.
        - **Train and serve read the same quantity.** The rule is the same in
          both: previous season's measured, else this season's pre-season.
          Nothing is joined at serve time that was not joined in training.

        The leak the warning guards against is also untouched, and that is the
        whole reason this is safe: the pre-season rating is joined at the
        match's own season, but contains nothing from it. Every rated player in
        it was rated strictly before, and an unrated one is mapped from EA's
        ``overall`` through a curve that is a rank of EA overalls within
        (role, season) -- EA's numbers only, no results.
        ``tests/test_strength_leakage.py`` pins both halves.

        One real anachronism remains, and it is roster composition rather than
        results: EA ships an edition in September, after one or two matchdays
        and after the window shuts, so for those matches the roster is a few
        weeks ahead of what was knowable. It is **269 of the 5,674 filled
        team-sides (4.7%)** within a fortnight of a season's first match, 638
        (11.2%) within five weeks, and it carries no information about any
        result. It is noted rather than corrected because correcting it means
        withholding the fallback from precisely the fixtures where a promoted
        side has no form either.
    """
    out = matches.copy()
    available = [c for c in STRENGTH_COLUMNS if c in strength.columns]
    if strength.empty or not available:
        return out

    # Match on a temporary string key rather than casting `Season` in place.
    # Callers compare seasons numerically (`run_backtest` filters `s >= start`),
    # so silently turning the column into strings breaks them well downstream of
    # here, with an error that points nowhere near this function.
    #
    # Keyed on season and team alone, *not* league -- see `_strength_lookup`,
    # which also translates the club names. A club plays in exactly one league
    # per season -- checked across all 1,072 team-seasons: no team name appears
    # in two leagues, and no (season, team) pair repeats -- so for a real
    # fixture, where both sides share the fixture's league, this joins exactly
    # what including `League` did.
    #
    # It differs only for a pairing the fixture list never contains: `/predict`
    # scores a hypothetical tie in the *home* side's league, which left the away
    # side matching nothing and being scored with no squad strength at all --
    # about a fifth of the model's influence on expected goals, silently absent.
    lookup = _strength_lookup(strength, available)
    out["_season_key"] = out["Season"].astype(str)

    if previous_season:
        # Shift the strength forward a season so a 2425 squad rating is what a
        # 2526 match sees. Season labels are "2425"-style, so the successor of
        # season YYZZ is ZZ(ZZ+1).
        lookup["_season_key"] = lookup["_season_key"].map(_next_season)

    for side in ("Home", "Away"):
        prefixed = lookup.rename(
            columns={
                "Team": f"{side}Team",
                _MATCHED: f"{side.lower()}{_MATCHED}",
                **{c: f"{side.lower()}_{c}" for c in available},
            }
        )
        out = out.merge(prefixed, on=["_season_key", f"{side}Team"], how="left", validate="m:1")

    if preseason is not None:
        out = _fill_from_preseason(out, preseason, available)

    return out.drop(columns=["_season_key", f"home{_MATCHED}", f"away{_MATCHED}"])


def _fill_from_preseason(
    matches: pd.DataFrame, preseason: pd.DataFrame, available: Sequence[str]
) -> pd.DataFrame:
    """Fill missing measured strength from the match's **own** season's roster.

    Own season deliberately, and it is the one line in this module where that is
    not a leak: see the note on :func:`add_team_strength`. ``_season_key`` is
    already the match's own season here -- :func:`add_team_strength` shifts the
    *lookup* forward a season, never the matches -- so this merge needs no
    shift, and adding one would hand a 2025/26 fixture the 2024/25 roster, which
    is the measured rating again by a longer route.

    Only holes are filled. A side with a measured rating keeps it untouched,
    which is what makes this strictly additive: every value the model saw before
    is still the value it sees.
    """
    out = matches
    renamed = preseason.rename(columns=PRESEASON_COLUMNS)
    usable = [c for c in available if c in renamed.columns]
    if preseason.empty or not usable:
        # The flag is still published, and false everywhere. Silently omitting
        # it would change the feature *names* depending on whether a file
        # happened to be on disk, which is how a trained artifact comes to
        # disagree with the frame it is served.
        for side in ("Home", "Away"):
            out[f"{side.lower()}_{PRESEASON_FLAG}"] = 0.0
        return out

    lookup = _strength_lookup(renamed, usable)
    for side in ("Home", "Away"):
        prefix = side.lower()
        temporary = {c: f"_pre_{c}" for c in usable} | {_MATCHED: f"_pre{_MATCHED}"}
        pre = lookup.rename(columns={"Team": f"{side}Team", **temporary})
        out = out.merge(pre, on=["_season_key", f"{side}Team"], how="left", validate="m:1")

        # A whole row at a time, never a column at a time. Patching a measured
        # side's missing goalkeeper from EA would mix two populations inside one
        # row -- the thing the flag exists to let the booster avoid -- and it is
        # not the gap this fallback is for.
        use = out[f"{prefix}{_MATCHED}"].isna() & out[f"_pre{_MATCHED}"].notna()
        for column in usable:
            out.loc[use, f"{prefix}_{column}"] = out.loc[use, f"_pre_{column}"]
        out[f"{prefix}_{PRESEASON_FLAG}"] = use.astype("float64")
        out = out.drop(columns=list(temporary.values()))
    return out


def _strength_lookup(strength: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """A ``(_season_key, Team, *columns)`` frame ready to merge onto matches.

    Factored out of :func:`add_team_strength` so the pre-season table is keyed
    exactly as the measured one is -- same alias translation, same string
    season key, same dropped ``League``. A second hand-rolled copy of this is
    how one of the two tables comes to reach Dortmund and the other not.
    """
    lookup = strength[["League", "Season", "Team", *columns]].copy()

    # Strength is built from player data, which carries Understat's club names
    # ("Borussia Dortmund", "Atletico Madrid"); matches carry football-data's
    # ("Dortmund", "Ath Madrid"). Without translating, the join fails silently --
    # a left merge just leaves NaN -- and before this line it reached only 40% of
    # Bundesliga and 52% of La Liga team-seasons, missing Dortmund, Leverkusen,
    # Leipzig and both Madrid clubs. Resolved here rather than in the CSV so the
    # Team Strength page keeps the fuller display names. Canonical names are not
    # alias keys, so this is safe on an already-resolved frame.
    lookup["Team"] = lookup["Team"].replace(load_aliases("understat"))
    lookup["_season_key"] = lookup["Season"].astype(str)

    # Did this side match a row at all? Asked with a marker rather than by
    # testing the values, because a measured row can legitimately carry a NaN
    # goalkeeper -- 1,113 team-sides do, mostly 2015/16, where FBref served no
    # `keeper` table -- and "no rating for the keeper" is a different gap from
    # "no season to measure", with a different answer.
    lookup[_MATCHED] = 1.0
    return lookup.drop(columns=["Season", "League"])


def _next_season(label: str) -> str:
    """``"2425"`` -> ``"2526"``. Returns the input unchanged if unparseable."""
    text = str(label)
    if len(text) != 4 or not text.isdigit():
        return text
    start = int(text[:2]) + 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


#: Sentinel for "load the European matches from disk". A plain ``None`` default
#: cannot express this, because ``None`` has to keep meaning "no European data".
class _Auto:
    """Type of the `AUTO` sentinel below.

    A bare `object()` would do at runtime, but it forces the parameter's
    annotation to `object` -- which accepts anything and narrows to nothing, so
    the `european` frame stayed untyped all the way into `european_elo_rows`.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "AUTO"


#: "Load the European ties yourself." Distinct from `None`, which means "do not
#: use them at all" -- see `build_match_features`.
AUTO = _Auto()


def european_elo_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Wide UEFA results to the minimal long shape the Elo walk consumes.

    Only the columns ``_walk`` reads. These rows are never returned to the
    caller and never reach a rolling window, so the match statistics and xG
    that the domestic history carries are neither needed nor available.
    """
    if matches.empty:
        return pd.DataFrame()

    neutral = matches.get("Round", pd.Series(index=matches.index, dtype="object")) == "Final"
    sides = []
    for own, opp, venue in (("HomeTeam", "AwayTeam", "Home"), ("AwayTeam", "HomeTeam", "Away")):
        goals_for, goals_against = ("FTHG", "FTAG") if venue == "Home" else ("FTAG", "FTHG")
        sides.append(
            pd.DataFrame(
                {
                    "League": matches["League"],
                    # The competition's own season label, so a UEFA tie counts
                    # as the same season as the domestic matches around it and
                    # does not trigger a spurious between-season regression.
                    "Season": matches["Season"].astype(str),
                    "Date": pd.to_datetime(matches["Date"]),
                    "Team": matches[own],
                    "Opponent": matches[opp],
                    "Venue": venue,
                    "GoalsFor": matches[goals_for],
                    "GoalsAgainst": matches[goals_against],
                    "Neutral": neutral,
                }
            )
        )
    return pd.concat(sides, ignore_index=True)


def build_team_features(
    tmh: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    congestion_days: int = 14,
    european: pd.DataFrame | None | _Auto = AUTO,
) -> pd.DataFrame:
    """Run the full long-shape feature build: prepare, roll, venue-split, rest.

    Args:
        european: UEFA ties for Elo to learn from. Defaults to loading them
            from disk rather than taking them from the caller, and that is
            deliberate: nine call sites build features, and one of them
            forgetting to pass these would serve predictions from a *different*
            Elo than the model was trained on. The reload guard compares
            feature **names**, so it would not notice. Pass ``None`` to opt out
            explicitly, which tests do.
    """
    df = prepare_team_match(tmh)
    if isinstance(european, _Auto):
        european = load_european_matches()
    extra = None if european is None else european_elo_rows(european)
    # Elo before the rolling features, so it is just another team-level column
    # the pivot picks up. It answers what the windows cannot: how good a side is
    # over years rather than over its last five or nineteen matches -- and,
    # once `extra` connects the leagues, how good it is against a side it has
    # never played.
    df = add_elo(df, extra=extra)
    for window in windows:
        df = add_team_form(df, window=window, venue_split=False)
        df = add_team_form(df, window=window, venue_split=True)
    return add_rest_features(df, congestion_days=congestion_days)


def build_match_features(
    tmh: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    congestion_days: int = 14,
    strength: pd.DataFrame | None = None,
    previous_season_strength: bool = True,
    preseason_strength: pd.DataFrame | None = None,
    european: pd.DataFrame | None | _Auto = AUTO,
) -> pd.DataFrame:
    """Build the wide, model-ready match feature table from long team-match rows.

    Args:
        tmh: Raw long team-match history (one row per team per match, with a
            ``Venue`` column of ``"Home"``/``"Away"``).
        windows: Rolling window sizes to compute form over.
        congestion_days: Lookback for the fixture-congestion count.
        european: UEFA ties for Elo to learn from; see
            :func:`build_team_features`. Loaded from disk by default.
        strength: Optional ``team_strength.csv`` to join squad quality from.
        previous_season_strength: Join the preceding season's strength. Leave
            True for training and backtesting; pass False only for live
            prediction, where the current squad is the right one.
        preseason_strength: Optional ``squad_strength.csv``, to fill the sides
            with no measured strength. Adds two columns, so training and
            serving must agree on whether it is passed -- the reload guard
            compares feature names. See :func:`add_team_strength`.

    Returns:
        One row per match: identifiers, the actual result (``FTHG``, ``FTAG``,
        ``FTR``) as targets, and ``home_*``/``away_*``/``*_diff`` features that
        are all strictly knowable before kickoff.
    """
    if "League" not in tmh.columns:
        # Single-league inputs (the original EPL-only files) predate the League
        # column. Default rather than fail, so older data still builds.
        tmh = tmh.assign(League="E0")

    long_df = build_team_features(
        tmh, windows=windows, congestion_days=congestion_days, european=european
    )

    feature_cols = _feature_columns(long_df, tmh)

    home = long_df[long_df["Venue"] == "Home"].copy()
    away = long_df[long_df["Venue"] == "Away"].copy()

    home = home.rename(columns={"Team": "HomeTeam", "Opponent": "AwayTeam"})
    away = away.rename(columns={"Team": "AwayTeam", "Opponent": "HomeTeam"})

    # Targets come from the home row, where GoalsFor/Against are already
    # oriented home-first.
    targets = home[[*_ID_COLS, "GoalsFor", "GoalsAgainst"]].rename(
        columns={"GoalsFor": "FTHG", "GoalsAgainst": "FTAG"}
    )
    # Unplayed fixtures have null goals and must keep a null result. Without the
    # final `.where`, NaN comparisons are all False and every upcoming match
    # would be silently labelled an away win.
    played = targets["FTHG"].notna() & targets["FTAG"].notna()
    targets["FTR"] = (
        pd.Series("D", index=targets.index)
        .where(targets["FTHG"] == targets["FTAG"])
        .fillna(pd.Series("H", index=targets.index).where(targets["FTHG"] > targets["FTAG"]))
        .fillna("A")
        .where(played)
    )

    home_feats = home[[*_ID_COLS, *feature_cols]].add_prefix("home_")
    away_feats = away[[*_ID_COLS, *feature_cols]].add_prefix("away_")
    home_feats = home_feats.rename(columns={f"home_{c}": c for c in _ID_COLS})
    away_feats = away_feats.rename(columns={f"away_{c}": c for c in _ID_COLS})

    matches = targets.merge(home_feats, on=list(_ID_COLS), how="inner", validate="1:1")
    matches = matches.merge(away_feats, on=list(_ID_COLS), how="inner", validate="1:1")

    if strength is not None:
        matches = add_team_strength(
            matches,
            strength,
            previous_season=previous_season_strength,
            preseason=preseason_strength,
        )

    matches = _add_diffs(matches, windows)
    matches["league_code"] = matches["League"].map(LEAGUE_CODES).astype("float64")
    return matches.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)


def build_upcoming_features(
    tmh: pd.DataFrame, fixtures: pd.DataFrame, **kwargs: object
) -> pd.DataFrame:
    """Attach pre-kickoff features to fixtures that have not been played.

    The trick is that no separate code path is needed. Each fixture is appended
    to the history as a pair of team-match rows with every statistic missing,
    then the normal builder runs over the combined frame. Because rolling form
    only ever looks *backwards*, those placeholder rows pick up each side's real
    recent form while contributing nothing themselves -- which is precisely the
    leak-safe behaviour the tests already guarantee.

    Args:
        tmh: Long team-match history of played matches.
        fixtures: ``League``, ``Season``, ``Date``, ``HomeTeam``, ``AwayTeam``.
        **kwargs: Passed through to :func:`build_match_features`.

    Returns:
        One row per fixture with features populated and ``FTHG``/``FTAG`` null.
    """
    if fixtures.empty:
        return pd.DataFrame()

    # Drop fixtures the history already contains. football-data's fixture list
    # keeps publishing a round after it has been played, so once results are
    # refreshed every "upcoming" fixture can already be in `tmh` -- appending a
    # placeholder for one then duplicates a real match, and the 1:1 merge in
    # `build_match_features` fails with a MergeError that points nowhere near
    # here. Predicting a played match through this path is meaningless anyway;
    # `/matches` scores those retrospectively.
    played = tmh[tmh["Venue"] == "Home"][["League", "Date", "Team", "Opponent"]]
    played = played.rename(columns={"Team": "HomeTeam", "Opponent": "AwayTeam"})
    played = played.assign(Date=pd.to_datetime(played["Date"]), _played=True)

    key = ["League", "Date", "HomeTeam", "AwayTeam"]
    fixtures = fixtures.assign(Date=pd.to_datetime(fixtures["Date"]))
    marked = fixtures.merge(played.drop_duplicates(key), on=key, how="left")
    already = marked["_played"].notna()
    if already.any():
        logger.info(
            "Skipping %d of %d fixture(s) already played and in the history",
            int(already.sum()),
            len(fixtures),
        )
        fixtures = fixtures[~already.to_numpy()]
    if fixtures.empty:
        return pd.DataFrame()

    # A team may appear at most once per batch, and this is not a style rule.
    # Rolling form looks back over a team's previous rows; when a team has
    # several placeholders they sit next to each other in the frame, so each
    # one's window fills with the *other placeholders* -- every statistic NaN --
    # instead of real matches. The frame still comes back the right shape, which
    # is what made this expensive: the season projection scored 330 fixtures on
    # one date and lost 31 of 232 features to silent NaN, halving how far apart
    # it placed teams. Callers that legitimately need a whole season at once
    # want `build_frozen_form_features`.
    appearances = pd.concat([fixtures["HomeTeam"], fixtures["AwayTeam"]]).value_counts()
    repeated = appearances[appearances > 1]
    if not repeated.empty:
        logger.warning(
            "%d team(s) appear more than once in this fixture batch (worst: %s x%d). "
            "Their rolling form will be computed over the other placeholder rows "
            "and come back NaN. Use build_frozen_form_features for a whole season.",
            len(repeated),
            repeated.index[0],
            int(repeated.iloc[0]),
        )

    placeholder = pd.concat(
        [
            _fixture_side(fixtures, "HomeTeam", "AwayTeam", "Home"),
            _fixture_side(fixtures, "AwayTeam", "HomeTeam", "Away"),
        ],
        ignore_index=True,
    )
    # Match the history's columns so the concat lines up. Missing statistics must
    # be float NaN rather than pd.NA: a pd.NA leaves the column as object dtype,
    # and the arithmetic in prepare_team_match then fails on it.
    for col in tmh.columns:
        if col not in placeholder.columns:
            placeholder[col] = np.nan if pd.api.types.is_numeric_dtype(tmh[col]) else None

    combined = pd.concat([tmh, placeholder[tmh.columns]], ignore_index=True)
    for col in tmh.columns:
        if pd.api.types.is_numeric_dtype(tmh[col]):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    built = build_match_features(combined, **kwargs)  # type: ignore[arg-type]

    key = ["League", "Date", "HomeTeam", "AwayTeam"]
    wanted = fixtures[key].assign(Date=pd.to_datetime(fixtures["Date"]))
    return built.merge(wanted, on=key, how="inner").reset_index(drop=True)


def build_frozen_form_features(
    tmh: pd.DataFrame,
    fixtures: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    **kwargs: object,
) -> pd.DataFrame:
    """Features for a whole season's remaining fixtures, form held constant.

    A season projection scores every unplayed fixture as of today and holds form
    there until May, so a team's feature vector is *the same* in all of its
    remaining matches. That makes the obvious approach -- hand the whole fixture
    list to :func:`build_upcoming_features` -- both wrong and wasteful: wrong
    because a team's many placeholder rows then roll their form over each other
    and come back NaN, wasteful because the identical vector is recomputed once
    per fixture.

    So each team's form is built once, in two synthetic rounds where everyone
    appears exactly once, venues swapped between them. Each fixture is then
    assembled from the home side's home-context block and the away side's
    away-context block, and the differences recomputed.

    Checked against the slow, obviously-correct alternative -- partitioning the
    fixtures into 40 rounds and building each separately -- the two agree on
    expected points per team to within 1.5%, at roughly a fortieth of the cost.

    Args:
        tmh: Long team-match history of played matches.
        fixtures: ``League``, ``Season``, ``HomeTeam``, ``AwayTeam``. Any
            ``Date`` is ignored: every fixture is scored as of today by design.
        windows: Rolling windows, passed on and used to rebuild the differences.
        **kwargs: Passed through to :func:`build_upcoming_features`.

    Returns:
        One row per fixture, with the same columns a normal feature build
        produces. Empty if no fixture has both sides' form available.
    """
    if fixtures.empty:
        return pd.DataFrame()

    teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
    league = fixtures["League"].iloc[0]
    season = str(fixtures["Season"].iloc[0])
    as_of = pd.Timestamp(pd.to_datetime(tmh["Date"]).max())

    # Pair the teams up arbitrarily; the opponent does not matter, because every
    # feature taken from these rounds describes the team itself. An odd team out
    # simply has no row, and its fixtures are dropped below rather than guessed.
    pairs = [(teams[i], teams[i + 1]) for i in range(0, len(teams) - 1, 2)]
    blocks: dict[str, dict[str, pd.Series]] = {}
    for offset, swap in enumerate((False, True), start=1):
        round_fixtures = pd.DataFrame(
            [
                {
                    "League": league,
                    "Season": season,
                    "HomeTeam": b if swap else a,
                    "AwayTeam": a if swap else b,
                    "Date": as_of + pd.Timedelta(days=7 * offset),
                }
                for a, b in pairs
            ]
        )
        built = build_upcoming_features(tmh, round_fixtures, windows=windows, **kwargs)
        for _, built_row in built.iterrows():
            blocks.setdefault(str(built_row["HomeTeam"]), {})["home"] = built_row
            blocks.setdefault(str(built_row["AwayTeam"]), {})["away"] = built_row

    if not blocks:
        return pd.DataFrame()

    template = next(iter(blocks.values()))
    # Every block got one side or the other filled in above, so one of these two
    # keys is present; indexing rather than `.get` chaining says so.
    sample = template["home"] if "home" in template else template["away"]
    home_cols = [c for c in sample.index if c.startswith("home_")]
    away_cols = [c for c in sample.index if c.startswith("away_")]

    rows = []
    for fixture in fixtures.itertuples():
        home_team, away_team = str(fixture.HomeTeam), str(fixture.AwayTeam)
        home = blocks.get(home_team, {}).get("home")
        away = blocks.get(away_team, {}).get("away")
        if home is None or away is None:
            continue
        row = {
            "League": league,
            "Season": season,
            "Date": as_of + pd.Timedelta(days=7),
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "FTHG": np.nan,
            "FTAG": np.nan,
            "FTR": None,
        }
        row.update({c: home[c] for c in home_cols})
        row.update({c: away[c] for c in away_cols})
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    # Differences pair a home column with its away twin, so they only become
    # correct once the two blocks are side by side.
    matches = _add_diffs(pd.DataFrame(rows), windows)
    matches["league_code"] = matches["League"].map(LEAGUE_CODES).astype("float64")
    return matches.reset_index(drop=True)


def _fixture_side(fixtures: pd.DataFrame, own: str, opp: str, venue: str) -> pd.DataFrame:
    """One side of each unplayed fixture, with no match statistics."""
    return pd.DataFrame(
        {
            "League": fixtures["League"],
            "Season": fixtures["Season"].astype(str),
            "Date": pd.to_datetime(fixtures["Date"]),
            "Team": fixtures[own],
            "Opponent": fixtures[opp],
            "Venue": venue,
            "Result": pd.NA,
        }
    )


def _feature_columns(long_df: pd.DataFrame, original: pd.DataFrame) -> list[str]:
    """Columns added by the feature build -- i.e. everything not in the input.

    Derived per-match quantities (``Points``, ``GoalDiff``, ...) are excluded:
    they describe the match being predicted, so carrying them forward would leak
    the result.
    """
    derived_in_match = {
        "Points",
        "Win",
        "Draw",
        "Loss",
        "GoalDiff",
        "ShotAccuracy",
        "xGDiff",
        "xGOverperformance",
        "xGPerShot",
        "npxGDiff",
    }
    original_cols = set(original.columns) | derived_in_match
    return [c for c in long_df.columns if c not in original_cols]


def _add_diffs(matches: pd.DataFrame, windows: Sequence[int] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Add ``{stem}_diff`` columns for every stem present on both sides."""
    diffs = {
        f"{stem}_diff": matches[f"home_{stem}"] - matches[f"away_{stem}"]
        for stem in diff_stems(windows)
        if f"home_{stem}" in matches.columns and f"away_{stem}" in matches.columns
    }
    return matches.assign(**diffs)
