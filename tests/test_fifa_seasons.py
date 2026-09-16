"""Tests for per-season EA FC rating matching.

The rest of the suite cannot catch a season-matching error, because every other
fixture holds a single edition. Yet matching the wrong edition is exactly the
failure that matters: blending one snapshot into every season gives a player the
same rating for his whole career, which is future knowledge -- it flatters a
backtest and is worthless in production.

Two things are asserted here that nothing else can:

- a season is matched against *its own* edition, and against no other
- the loose matcher's weaker keys are rejected when they are ambiguous

The off-by-one guard earns its place too. EA editions are named for the year
*after* release -- FIFA 16 shipped in September 2015 and describes 2015/16 -- so
naming a file by its edition rather than its season invites a one-season shift
that looks exactly like leakage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.players.fifa import (
    _strip_foreign_script,
    attach_fifa,
    load_fifa,
)


def _export(rows: list[dict]) -> pd.DataFrame:
    """A loaded-and-normalised FIFA frame, as :func:`load_fifa` returns one."""
    frame = pd.DataFrame(rows)
    frame["_norm"] = frame["fifa_name"].str.lower()
    if "role" not in frame.columns:
        frame["role"] = "MID"
    return frame


@pytest.fixture
def two_editions() -> pd.DataFrame:
    """One player, two seasons, deliberately different overalls.

    The values double as season labels: 70 can only have come from 1516 and 90
    can only have come from 2526, so a wrong-season match is visible in the
    assertion rather than hidden behind a plausible number.
    """
    return _export(
        [
            {"fifa_name": "jude bellingham", "fifa_overall": 70.0, "Season": "1516"},
            {"fifa_name": "jude bellingham", "fifa_overall": 90.0, "Season": "2526"},
        ]
    )


@pytest.fixture
def players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Player": ["jude bellingham", "jude bellingham"],
            "Season": ["1516", "2526"],
            "position_group": ["M", "M"],
        }
    )


def test_each_season_gets_its_own_edition(players, two_editions):
    out = attach_fifa(players, two_editions)
    by_season = dict(zip(out["Season"].astype(str), out["fifa_overall"], strict=True))
    assert by_season["1516"] == 70.0
    assert by_season["2526"] == 90.0


def test_a_rating_is_not_carried_across_seasons(players, two_editions):
    """The regression this whole arrangement exists to prevent."""
    out = attach_fifa(players, two_editions)
    assert out["fifa_overall"].nunique() == 2, (
        "both seasons got the same rating -- one edition is being applied to all"
    )


def test_season_with_no_edition_gets_nan_not_another_year(two_editions):
    """A missing edition must leave a hole, never borrow a neighbouring year."""
    players = pd.DataFrame(
        {
            "Player": ["jude bellingham"],
            "Season": ["1920"],
            "position_group": ["M"],
        }
    )
    out = attach_fifa(players, two_editions)
    assert out["fifa_overall"].isna().all()


def test_season_is_read_from_the_filename(tmp_path):
    """``fifa_1516.csv`` is season 1516 -- not edition 16, and not 1617."""
    (tmp_path / "fifa_1516.csv").write_text(
        "name,overall,positions\nJude Bellingham,70,CM\n", encoding="utf-8"
    )
    (tmp_path / "fifa_2526.csv").write_text(
        "name,overall,positions\nJude Bellingham,90,CM\n", encoding="utf-8"
    )
    loaded = load_fifa(tmp_path)
    assert set(loaded["Season"]) == {"1516", "2526"}
    pairing = dict(zip(loaded["Season"], loaded["fifa_overall"], strict=True))
    assert pairing["1516"] == 70.0


def test_a_stray_csv_does_not_become_a_season(tmp_path):
    """Only ``fifa_<season>.csv`` files are treated as per-season editions."""
    (tmp_path / "fifa_1516.csv").write_text(
        "name,overall,positions\nJude Bellingham,70,CM\n", encoding="utf-8"
    )
    (tmp_path / "players.csv").write_text(
        "name,overall,positions\nSomeone Else,99,ST\n", encoding="utf-8"
    )
    loaded = load_fifa(tmp_path)
    assert set(loaded["Season"]) == {"1516"}
    assert "Someone Else" not in set(loaded["fifa_name"])


def test_trailing_family_names_are_matched():
    """``"Kylian Mbappe"`` must find ``"Kylian Mbappe Lottin"``.

    First-plus-last builds ``"kylian lottin"`` and misses; the leading pair is
    what resolves it.
    """
    export = _export(
        [{"fifa_name": "kylian mbappe lottin", "fifa_overall": 91.0, "Season": "2526"}]
    )
    players = pd.DataFrame(
        {"Player": ["kylian mbappe"], "Season": ["2526"], "position_group": ["F"]}
    )
    assert attach_fifa(players, export)["fifa_overall"].iloc[0] == 91.0


def test_an_ambiguous_leading_pair_is_rejected():
    """Two players sharing a leading pair resolve to neither."""
    export = _export(
        [
            {"fifa_name": "jose maria gimenez", "fifa_overall": 84.0, "Season": "2526"},
            {"fifa_name": "jose maria callejon", "fifa_overall": 80.0, "Season": "2526"},
        ]
    )
    players = pd.DataFrame(
        {"Player": ["jose maria"], "Season": ["2526"], "position_group": ["M"]}
    )
    assert attach_fifa(players, export)["fifa_overall"].isna().all()


def test_foreign_script_is_stripped():
    """The FC26 export welds a second script on with no separator."""
    assert _strip_foreign_script("Mohamed Salah Hamed Ghalyمحمد صلاح") == (
        "Mohamed Salah Hamed Ghaly"
    )


@pytest.mark.parametrize(
    "name",
    ["Willum Þór Willumsson", "Tjaš Begić", "Kylian Mbappé", "Anđelo Šetka"],
)
def test_accented_latin_names_survive_stripping(name):
    """Accents are not foreign script; damaging them would lose real matches."""
    assert _strip_foreign_script(name) == name


def test_a_purely_foreign_name_does_not_become_whitespace():
    """A name with no Latin part collapses to empty, not to a stray space."""
    assert _strip_foreign_script("香川 真司") == ""


def _club_fixture(extra_players: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An edition where each club's EA name can be learned from exact matches.

    Understat and EA spell clubs differently on purpose here, so the test
    fails if the tiebreaker ever starts comparing club strings directly.
    """
    teammates = {
        "Las Palmas": ("UD Las Palmas", ["ana uno", "ana dos", "ana tres"]),
        "Osasuna": ("CA Osasuna", ["oso uno", "oso dos", "oso tres"]),
        "Granada": ("Granada CF", ["gra uno", "gra dos", "gra tres"]),
    }
    rows, players = [], []
    for team, (club, names) in teammates.items():
        for name in names:
            rows.append({"fifa_name": name, "fifa_overall": 70.0,
                         "fifa_club": club, "Season": "1516"})
            players.append({"Player": name, "Team": team})
    rows += [
        {"fifa_name": "david garcia zubiria", "fifa_overall": 75.0,
         "fifa_club": "CA Osasuna", "Season": "1516"},
        {"fifa_name": "david garcia santana", "fifa_overall": 72.0,
         "fifa_club": "UD Las Palmas", "Season": "1516"},
        {"fifa_name": "hugo miguel almeida costa lopes", "fifa_overall": 71.0,
         "fifa_club": "Granada CF", "Season": "1516"},
        {"fifa_name": "rui lopes", "fifa_overall": 65.0,
         "fifa_club": "CA Osasuna", "Season": "1516"},
    ]
    frame = pd.DataFrame(players + extra_players)
    frame["Season"] = "1516"
    frame["position_group"] = "D"
    return frame, _export(rows)


def _overall(matched: pd.DataFrame, player: str, team: str) -> float:
    row = matched[(matched["Player"] == player) & (matched["Team"] == team)]
    return row["fifa_overall"].iloc[0]


def test_club_breaks_an_ambiguous_leading_pair():
    """Two David Garcías, one at each club: each row takes his own."""
    players, export = _club_fixture(
        [{"Player": "david garcia", "Team": "Las Palmas"},
         {"Player": "david garcia", "Team": "Osasuna"}]
    )
    matched = attach_fifa(players, export)
    assert _overall(matched, "david garcia", "Las Palmas") == 72.0
    assert _overall(matched, "david garcia", "Osasuna") == 75.0


def test_club_breaks_a_shared_surname_only_with_the_given_name():
    """``"Miguel Lopes"`` finds Hugo Miguel ... Lopes at his club; a stranger does not."""
    players, export = _club_fixture(
        [{"Player": "miguel lopes", "Team": "Granada"},
         {"Player": "pedro lopes", "Team": "Granada"}]
    )
    matched = attach_fifa(players, export)
    assert _overall(matched, "miguel lopes", "Granada") == 71.0
    assert pd.isna(_overall(matched, "pedro lopes", "Granada"))


def test_club_does_not_resolve_when_neither_candidate_is_there():
    """An ambiguous key at a third club stays unmatched rather than guessing."""
    players, export = _club_fixture([{"Player": "david garcia", "Team": "Granada"}])
    assert pd.isna(_overall(attach_fifa(players, export), "david garcia", "Granada"))


def test_club_is_not_required_for_an_unambiguous_name():
    """A player EA lists at his old club still matches: club only breaks ties."""
    players, export = _club_fixture([{"Player": "hugo lopes", "Team": "Osasuna"}])
    assert _overall(attach_fifa(players, export), "hugo lopes", "Osasuna") == 71.0
