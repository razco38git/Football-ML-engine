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
    windowed = [
        stem.format(w=w) for w in windows for stem in _WINDOWED_STEMS
    ]
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
    """
    out = matches.copy()
    available = [c for c in STRENGTH_COLUMNS if c in strength.columns]
    if strength.empty or not available:
        return out

    # Match on a temporary string key rather than casting `Season` in place.
    # Callers compare seasons numerically (`run_backtest` filters `s >= start`),
    # so silently turning the column into strings breaks them well downstream of
    # here, with an error that points nowhere near this function.
    lookup = strength[["League", "Season", "Team", *available]].copy()

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

    # Keyed on season and team alone, *not* league. A club plays in exactly one
    # league per season -- checked across all 1,072 team-seasons: no team name
    # appears in two leagues, and no (season, team) pair repeats -- so for a
    # real fixture, where both sides share the fixture's league, this joins
    # exactly what including `League` did.
    #
    # It differs only for a pairing the fixture list never contains: `/predict`
    # scores a hypothetical tie in the *home* side's league, which left the away
    # side matching nothing and being scored with no squad strength at all --
    # about a fifth of the model's influence on expected goals, silently absent.
    lookup = lookup.drop(columns=["Season", "League"])
    out["_season_key"] = out["Season"].astype(str)

    if previous_season:
        # Shift the strength forward a season so a 2425 squad rating is what a
        # 2526 match sees. Season labels are "2425"-style, so the successor of
        # season YYZZ is ZZ(ZZ+1).
        lookup["_season_key"] = lookup["_season_key"].map(_next_season)

    for side in ("Home", "Away"):
        prefixed = lookup.rename(
            columns={"Team": f"{side}Team", **{c: f"{side.lower()}_{c}" for c in available}}
        )
        out = out.merge(prefixed, on=["_season_key", f"{side}Team"], how="left")

    return out.drop(columns=["_season_key"])


def _next_season(label: str) -> str:
    """``"2425"`` -> ``"2526"``. Returns the input unchanged if unparseable."""
    text = str(label)
    if len(text) != 4 or not text.isdigit():
        return text
    start = int(text[:2]) + 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


#: Sentinel for "load the European matches from disk". A plain ``None`` default
#: cannot express this, because ``None`` has to keep meaning "no European data".
AUTO = object()


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
    european: pd.DataFrame | None | object = AUTO,
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
    if european is AUTO:
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
    european: pd.DataFrame | None | object = AUTO,
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
            matches, strength, previous_season=previous_season_strength
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
            int(already.sum()), len(fixtures),
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
            len(repeated), repeated.index[0], int(repeated.iloc[0]),
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
            placeholder[col] = (
                np.nan if pd.api.types.is_numeric_dtype(tmh[col]) else None
            )

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
        built = build_upcoming_features(
            tmh, round_fixtures, windows=windows, **kwargs
        )
        for _, row in built.iterrows():
            blocks.setdefault(row["HomeTeam"], {})["home"] = row
            blocks.setdefault(row["AwayTeam"], {})["away"] = row

    if not blocks:
        return pd.DataFrame()

    template = next(iter(blocks.values()))
    sample = template.get("home", template.get("away"))
    home_cols = [c for c in sample.index if c.startswith("home_")]
    away_cols = [c for c in sample.index if c.startswith("away_")]

    rows = []
    for fixture in fixtures.itertuples():
        home = blocks.get(fixture.HomeTeam, {}).get("home")
        away = blocks.get(fixture.AwayTeam, {}).get("away")
        if home is None or away is None:
            continue
        row = {
            "League": league,
            "Season": season,
            "Date": as_of + pd.Timedelta(days=7),
            "HomeTeam": fixture.HomeTeam,
            "AwayTeam": fixture.AwayTeam,
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


def _fixture_side(
    fixtures: pd.DataFrame, own: str, opp: str, venue: str
) -> pd.DataFrame:
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
        "Points", "Win", "Draw", "Loss", "GoalDiff", "ShotAccuracy",
        "xGDiff", "xGOverperformance", "xGPerShot", "npxGDiff",
    }
    original_cols = set(original.columns) | derived_in_match
    return [c for c in long_df.columns if c not in original_cols]


def _add_diffs(
    matches: pd.DataFrame, windows: Sequence[int] = DEFAULT_WINDOWS
) -> pd.DataFrame:
    """Add ``{stem}_diff`` columns for every stem present on both sides."""
    diffs = {
        f"{stem}_diff": matches[f"home_{stem}"] - matches[f"away_{stem}"]
        for stem in diff_stems(windows)
        if f"home_{stem}" in matches.columns and f"away_{stem}" in matches.columns
    }
    return matches.assign(**diffs)
