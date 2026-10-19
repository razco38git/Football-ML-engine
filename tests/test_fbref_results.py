"""Topping the match history up from FBref when football-data lags.

football-data is the results source and it falls behind -- three weeks, in
October 2026 -- which froze the projected tables, because `project_season`
simulates from the match history and will not simulate from rows missing half
their columns.

The gap is narrower than it looks. Of what football-data brings, only goals,
shots and shots on target ever become features; fouls, corners and cards are
stored and read by nothing. FBref has all three and Understat has the xG family,
so a topped-up row is complete in every column the model uses.

The guarantee worth pinning is that this only ever *adds*: football-data's own
rows must win every contested match, so a week when it is current produces
exactly the frame it would have produced without any of this.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.ingest import fbref_results as fb


def _existing(dates: list[str]) -> pd.DataFrame:
    """Rows as `to_team_match_history` produces them, football-data shaped."""
    return pd.DataFrame({
        "League": ["E0"] * len(dates),
        "Season": ["2627"] * len(dates),
        "Date": pd.to_datetime(dates),
        "Team": ["Arsenal"] * len(dates),
        "Opponent": ["Chelsea"] * len(dates),
        "Venue": ["Home"] * len(dates),
        "GoalsFor": [2] * len(dates),
        "GoalsAgainst": [1] * len(dates),
        "Result": ["W"] * len(dates),
        "ShotsFor": [14] * len(dates),
        "FoulsFor": [9] * len(dates),
    })


def _incoming(dates: list[str], team: str = "Arsenal") -> pd.DataFrame:
    return pd.DataFrame({
        "League": ["E0"] * len(dates),
        "Season": ["2627"] * len(dates),
        "Date": pd.to_datetime(dates),
        "Team": [team] * len(dates),
        "Opponent": ["Chelsea"] * len(dates),
        "Venue": ["Home"] * len(dates),
        "GoalsFor": [3] * len(dates),
        "GoalsAgainst": [0] * len(dates),
        "Result": ["W"] * len(dates),
        "ShotsFor": [20] * len(dates),
        "ShotsAgainst": [5] * len(dates),
        "ShotsOnTargetFor": [8] * len(dates),
        "ShotsOnTargetAgainst": [2] * len(dates),
    })


def test_only_missing_matches_are_added() -> None:
    have = _existing(["2026-09-20"])
    out = fb.top_up(have, _incoming(["2026-09-20", "2026-10-11"]))
    assert len(out) == 2
    assert sorted(str(d.date()) for d in out["Date"]) == ["2026-09-20", "2026-10-11"]


def test_the_primary_source_wins_a_contested_row() -> None:
    """The same match from both sources must keep football-data's version."""
    have = _existing(["2026-09-20"])
    out = fb.top_up(have, _incoming(["2026-09-20"]))

    kept = out[out["Date"] == pd.Timestamp("2026-09-20")]
    assert len(kept) == 1
    # football-data said 2-1 with 14 shots; FBref said 3-0 with 20.
    assert kept.iloc[0]["GoalsFor"] == 2
    assert kept.iloc[0]["ShotsFor"] == 14
    assert kept.iloc[0]["source"] == fb.FOOTBALL_DATA


def test_a_current_primary_source_changes_nothing() -> None:
    """The property that makes this safe to leave switched on."""
    have = _existing(["2026-09-19", "2026-09-20"])
    out = fb.top_up(have, _incoming(["2026-09-19", "2026-09-20"]))
    pd.testing.assert_frame_equal(
        out.drop(columns="source").reset_index(drop=True),
        have.reset_index(drop=True),
        check_dtype=False,
    )


def test_nothing_to_add_is_not_an_error() -> None:
    have = _existing(["2026-09-20"])
    assert len(fb.top_up(have, pd.DataFrame())) == 1
    assert len(fb.top_up(have, _incoming([]))) == 1


def test_every_row_is_labelled_with_its_source() -> None:
    """A silent mix would be the thing to avoid: these rows carry no fouls."""
    out = fb.top_up(_existing(["2026-09-20"]), _incoming(["2026-10-11"]))
    by_source = out.set_index(out["Date"].dt.date)["source"].to_dict()
    assert by_source[pd.Timestamp("2026-09-20").date()] == fb.FOOTBALL_DATA
    assert by_source[pd.Timestamp("2026-10-11").date()] == fb.FBREF


def test_matching_is_per_team_not_per_date() -> None:
    """Two clubs play on the same day; only the missing one may be added."""
    have = _existing(["2026-10-11"])  # Arsenal v Chelsea
    out = fb.top_up(have, _incoming(["2026-10-11"], team="Liverpool"))
    assert len(out) == 2
    assert set(out["Team"]) == {"Arsenal", "Liverpool"}


def test_a_source_column_is_added_even_with_no_top_up() -> None:
    """So a reader never has to wonder whether an unlabelled row is labelled."""
    out = fb.top_up(_existing(["2026-09-20"]), pd.DataFrame())
    assert set(out["source"]) == {fb.FOOTBALL_DATA}


def test_an_existing_source_column_is_respected() -> None:
    """Re-running the build must not relabel rows it already labelled."""
    have = _existing(["2026-09-20"]).assign(source=fb.FBREF)
    out = fb.top_up(have, pd.DataFrame())
    assert set(out["source"]) == {fb.FBREF}


def test_a_row_missing_a_feature_column_is_refused() -> None:
    """The guarantee that lets these rows near the training data.

    soccerdata drives FBref through a headless browser and that browser can die
    mid-run: on 2026-10-12 England and France came back with shots while Germany
    and Italy came back with none, and 36 of 88 rows went in with NaN. A league
    that fails this waits for a run where the scrape worked.
    """
    assert set(fb.FEATURE_COLUMNS) == {
        "GoalsFor", "GoalsAgainst",
        "ShotsFor", "ShotsAgainst",
        "ShotsOnTargetFor", "ShotsOnTargetAgainst",
    }
    # Every feature column is one the top-up promises to deliver.
    for column in fb.FEATURE_COLUMNS:
        assert column in fb.COLUMNS


# --- the fetcher's contract ---------------------------------------------------

def test_unknown_division_is_rejected() -> None:
    """A UEFA code here would quietly put European ties in the match history."""
    with pytest.raises(KeyError, match="unknown division"):
        fb.fetch_fbref_results(["UCL"])


def test_a_scrape_failure_returns_empty_rather_than_raising(monkeypatch) -> None:
    """A failed top-up must never cost the rows football-data did supply."""
    def boom(*_a, **_k):
        raise ConnectionError("FBref is down")

    monkeypatch.setattr("footballml.ingest.schedule.played_results", boom)
    out = fb.fetch_fbref_results(["E0"])
    assert out.empty
    assert list(out.columns) == fb.COLUMNS


def test_the_columns_it_promises_are_the_ones_features_need() -> None:
    """Fouls, corners and cards are absent on purpose -- nothing reads them."""
    assert "FoulsFor" not in fb.COLUMNS
    assert "CornersFor" not in fb.COLUMNS
    for needed in (
        "GoalsFor", "GoalsAgainst",
        "ShotsFor", "ShotsAgainst",
        "ShotsOnTargetFor", "ShotsOnTargetAgainst",
    ):
        assert needed in fb.COLUMNS


# --- the gate's memory --------------------------------------------------------
#
# The gate compares the schedule to the history and scrapes if anything is
# missing. With no memory, a match *neither* source can supply keeps its
# division behind for good: on 2026-10-13 La Liga was behind by one match --
# Vallecano 0-0 Ath Bilbao, 10 October -- that football-data had skipped and
# whose FBref shooting line did not exist, so the completeness check refused the
# row, correctly, and `build_dataset` paid 6m 43s to be told so again every
# morning.
#
# What is pinned here is the asymmetry that makes parking safe. A league that
# comes back with shots for every match but one has said something true about
# that one. A league that comes back with nothing has a dead browser, and
# parking on that would hide a real gap for a week.

def _schedule(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Played matches as `played_results` shapes them."""
    return pd.DataFrame({
        "League": [r[0] for r in rows],
        "Season": ["2627"] * len(rows),
        "Date": pd.to_datetime([r[1] for r in rows]),
        "HomeTeam": [r[2] for r in rows],
        "AwayTeam": [r[3] for r in rows],
        "FTHG": [0] * len(rows),
        "FTAG": [0] * len(rows),
        "FTR": ["D"] * len(rows),
    })


def _history(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """The same matches as the history holds them: one row per team."""
    sides = []
    for league, date, home, away in rows:
        for team, opponent, venue in ((home, away, "Home"), (away, home, "Away")):
            sides.append({
                "League": league, "Season": "2627", "Date": pd.Timestamp(date),
                "Team": team, "Opponent": opponent, "Venue": venue,
                "GoalsFor": 0, "GoalsAgainst": 0, "Result": "D",
            })
    return pd.DataFrame(sides)


def test_a_match_the_history_lacks_is_missing() -> None:
    sched = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao"),
                       ("SP1", "2026-10-10", "Barcelona", "Getafe")])
    known = _history([("SP1", "2026-10-10", "Barcelona", "Getafe")])
    out = fb._missing_matches(sched, known)
    assert list(out["HomeTeam"]) == ["Vallecano"]


def test_no_history_means_everything_is_missing() -> None:
    """Nothing to compare must not read as nothing to do."""
    sched = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])
    assert len(fb._missing_matches(sched, None)) == 1
    assert len(fb._missing_matches(sched, pd.DataFrame())) == 1


def test_a_parked_match_is_not_retried_within_the_window(tmp_path) -> None:
    sched = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])
    path = tmp_path / "deferred.csv"
    now = pd.Timestamp("2026-10-14")

    fb._record_deferred(sched, pd.DataFrame(columns=fb.DEFERRED_COLUMNS), path, now)
    deferred = fb._load_deferred(path)
    assert fb._parked(deferred, now) == fb._keys(sched)
    # Still inside the window the next morning.
    assert fb._parked(deferred, now + pd.Timedelta(days=1)) == fb._keys(sched)


def test_a_parked_match_is_retried_once_the_window_passes(tmp_path) -> None:
    """Parking must not be permanent: FBref does backfill match reports."""
    sched = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])
    path = tmp_path / "deferred.csv"
    now = pd.Timestamp("2026-10-14")

    fb._record_deferred(sched, pd.DataFrame(columns=fb.DEFERRED_COLUMNS), path, now)
    deferred = fb._load_deferred(path)
    assert fb._parked(deferred, now + fb.RETRY_AFTER) == set()


def test_attempts_accumulate_across_runs(tmp_path) -> None:
    """A count of four on a six-week-old hole is how a reader learns it is permanent."""
    sched = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])
    path = tmp_path / "deferred.csv"

    for day in ("2026-10-14", "2026-10-21", "2026-10-28"):
        fb._record_deferred(sched, fb._load_deferred(path), path, pd.Timestamp(day))

    out = fb._load_deferred(path)
    assert len(out) == 1
    assert int(out.iloc[0]["attempts"]) == 3
    assert out.iloc[0]["last_attempt"] == pd.Timestamp("2026-10-28")


def test_a_parked_match_is_forgotten_once_it_arrives(tmp_path) -> None:
    """Otherwise the file grows a permanent row for every hole ever seen."""
    sched = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])
    path = tmp_path / "deferred.csv"
    fb._record_deferred(sched, pd.DataFrame(columns=fb.DEFERRED_COLUMNS), path,
                        pd.Timestamp("2026-10-14"))

    arrived = _history([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])
    assert fb._load_deferred(path, seen=fb._seen(arrived)).empty


def test_other_divisions_keep_their_records(tmp_path) -> None:
    """A run that scrapes only Spain must not drop a parked German match."""
    path = tmp_path / "deferred.csv"
    german = _schedule([("D1", "2026-10-03", "Mainz", "Leverkusen")])
    fb._record_deferred(german, pd.DataFrame(columns=fb.DEFERRED_COLUMNS), path,
                        pd.Timestamp("2026-10-07"))

    spanish = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])
    fb._record_deferred(spanish, fb._load_deferred(path), path, pd.Timestamp("2026-10-14"))

    out = fb._load_deferred(path)
    assert sorted(out["League"]) == ["D1", "SP1"]


def test_an_unreadable_record_makes_everything_eligible(tmp_path) -> None:
    """Failing open: a corrupt cache costs a scrape, not a day of results."""
    path = tmp_path / "deferred.csv"
    path.write_text("not,a,valid\nrecord\n", encoding="utf-8")
    assert fb._load_deferred(path).empty

    path.write_text("League,Season\nSP1,2627\n", encoding="utf-8")
    assert fb._load_deferred(path).empty


# --- what gets parked, and what must not --------------------------------------

def _merged(rows: list[tuple[str, str, str, str, object]]) -> pd.DataFrame:
    """`wide` as the merge leaves it: the schedule plus HS/AS, NaN when unresolved."""
    return pd.DataFrame({
        "League": [r[0] for r in rows],
        "Season": ["2627"] * len(rows),
        "Date": pd.to_datetime([r[1] for r in rows]),
        "HomeTeam": [r[2] for r in rows],
        "AwayTeam": [r[3] for r in rows],
        "HS": [r[4] for r in rows],
        "AS": [r[4] for r in rows],
    })


def test_a_match_with_no_shot_line_is_parked(tmp_path, monkeypatch) -> None:
    path = tmp_path / "deferred.csv"
    monkeypatch.setattr(fb, "DEFERRED_PATH", path)
    wide = _merged([
        ("SP1", "2026-10-10", "Barcelona", "Getafe", 23.0),
        ("SP1", "2026-10-10", "Vallecano", "Ath Bilbao", float("nan")),
    ])
    eligible = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])

    fb._park_what_failed(wide, eligible, fb._load_deferred(path),
                         pd.Timestamp("2026-10-14"))

    out = fb._load_deferred(path)
    assert list(out["HomeTeam"]) == ["Vallecano"]


def test_a_dead_browser_parks_nothing(tmp_path, monkeypatch) -> None:
    """The load-bearing distinction.

    soccerdata drives FBref through a headless browser that can die mid-run,
    and a league then comes back with no shots at all. Parking on that would
    hide a genuine gap for a week -- the opposite of the point.
    """
    path = tmp_path / "deferred.csv"
    monkeypatch.setattr(fb, "DEFERRED_PATH", path)
    wide = _merged([
        ("SP1", "2026-10-10", "Barcelona", "Getafe", float("nan")),
        ("SP1", "2026-10-10", "Vallecano", "Ath Bilbao", float("nan")),
    ])
    eligible = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])

    fb._park_what_failed(wide, eligible, fb._load_deferred(path),
                         pd.Timestamp("2026-10-14"))
    assert fb._load_deferred(path).empty


def test_only_what_the_run_was_trying_to_add_is_parked(tmp_path, monkeypatch) -> None:
    """`wide` carries the whole season; most of it the history already has."""
    path = tmp_path / "deferred.csv"
    monkeypatch.setattr(fb, "DEFERRED_PATH", path)
    wide = _merged([
        ("SP1", "2026-08-15", "Alaves", "Getafe", float("nan")),  # old, already stored
        ("SP1", "2026-10-10", "Barcelona", "Getafe", 23.0),
        ("SP1", "2026-10-10", "Vallecano", "Ath Bilbao", float("nan")),
    ])
    eligible = _schedule([("SP1", "2026-10-10", "Vallecano", "Ath Bilbao")])

    fb._park_what_failed(wide, eligible, fb._load_deferred(path),
                         pd.Timestamp("2026-10-14"))
    assert list(fb._load_deferred(path)["HomeTeam"]) == ["Vallecano"]


# --- FBref names a competition two different ways ----------------------------
#
# soccerdata's league table knows only the spelling FBref uses on its
# competition index. `read_team_match_stats` takes each row's league from the
# match log's `Comp` column, which spells two of them differently, and
# `_translate_league` maps anything it does not recognise to NaN before a final
# `.loc[self.leagues]` drops it. Every German and Italian row was discarded
# inside soccerdata, for as long as the top-up has existed: measured on the
# live table, `fbref` had supplied 18 rows each for E0, F1 and SP1 and zero for
# D1 and I1, ever.


def _translated(values: list[str]) -> list:
    reader = fb._matchlog_reader(
        ["GER-Bundesliga", "ITA-Serie A", "ENG-Premier League"], ["2627"]
    )
    frame = pd.DataFrame({"league": values})
    return list(type(reader)._translate_league(frame)["league"])


def test_the_matchlog_names_are_declared() -> None:
    """Readable in the source, not discovered from a failure again."""
    assert fb.MATCHLOG_LEAGUE_ALIASES == {
        "Bundesliga": "GER-Bundesliga",
        "Serie A": "ITA-Serie A",
    }


def test_the_two_matchlog_spellings_translate() -> None:
    assert _translated(["Bundesliga", "Serie A"]) == ["GER-Bundesliga", "ITA-Serie A"]


def test_the_other_three_leagues_are_untouched() -> None:
    """They are spelled identically in both places, which is why they worked."""
    assert _translated(["Premier League", "Ligue 1", "La Liga"]) == [
        "ENG-Premier League", "FRA-Ligue 1", "ESP-La Liga",
    ]


def test_the_index_spellings_still_translate() -> None:
    """The alias is an addition, not a replacement: league *discovery* reads
    the index spelling, and overriding it finds no German teams at all."""
    assert _translated(["Fußball-Bundesliga", "Serie A (M)"]) == [
        "GER-Bundesliga", "ITA-Serie A",
    ]


def test_a_competition_we_do_not_follow_is_still_dropped() -> None:
    """A cup tie in the same match log must not be mistaken for league play."""
    out = _translated(["DFL-Supercup", "Coppa Italia", "FA Community Shield"])
    assert all(pd.isna(v) for v in out), out


def test_the_source_names_soccerdata_uses_for_discovery_are_unchanged() -> None:
    """The fix must not reach the league table every other reader shares."""
    import soccerdata as sd

    assert sd.FBref._all_leagues()["GER-Bundesliga"] == "Fußball-Bundesliga"
    assert sd.FBref._all_leagues()["ITA-Serie A"] == "Serie A (M)"


def test_the_subclass_can_still_see_its_leagues() -> None:
    """soccerdata keys its league table on `cls.__name__`, so a subclass of its
    own finds nothing and every league is rejected as invalid."""
    reader = fb._matchlog_reader(["GER-Bundesliga"], ["2627"])
    leagues = type(reader)._all_leagues()
    assert "GER-Bundesliga" in leagues
    assert len(leagues) > 5
