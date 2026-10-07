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
    primary_role,
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


def test_unmatched_role_uses_the_season_not_the_career():
    """A player's role must describe what he played *that season*.

    Moisés Caicedo has no EA entry, a season position of M and a career
    position of D. Falling back to the career value made him a centre-back
    rated 90, which propped up Chelsea's defence rating.
    """
    export = _export(
        [{"fifa_name": "someone else", "fifa_overall": 70.0, "Season": "2526"}]
    )
    players = pd.DataFrame(
        {
            "Player": ["moises caicedo"],
            "Season": ["2526"],
            "season_position": ["M"],
            "position_group": ["D"],
        }
    )
    assert attach_fifa(players, export)["role"].iloc[0] == "MID"


def test_career_position_is_still_the_backstop():
    """With no season position, the career one is better than nothing."""
    export = _export(
        [{"fifa_name": "someone else", "fifa_overall": 70.0, "Season": "2526"}]
    )
    players = pd.DataFrame(
        {
            "Player": ["unknown player"],
            "Season": ["2526"],
            "season_position": [None],
            "position_group": ["F"],
        }
    )
    assert attach_fifa(players, export)["role"].iloc[0] == "FWD"


def test_extra_given_and_family_names_are_matched():
    """EA carries both extra given *and* extra family names for some players.

    "Moises Caicedo" against "Moises Isaac Caicedo Corozo": first-plus-last
    gives "moises corozo", the leading pair "moises isaac", and the surname
    tier looks for "corozo" — so he went unmatched and, with no EA rating to
    temper him, was rated 90 off a misassigned role.
    """
    export = _export(
        [{"fifa_name": "moises isaac caicedo corozo", "fifa_overall": 87.0,
          "Season": "2526"}]
    )
    players = pd.DataFrame(
        {"Player": ["moises caicedo"], "Season": ["2526"], "position_group": ["M"]}
    )
    assert attach_fifa(players, export)["fifa_overall"].iloc[0] == 87.0


def test_containment_needs_every_token():
    """A partial overlap is not a match: "david silva" must not take "david luiz"."""
    export = _export(
        [{"fifa_name": "david luiz moreira marinho", "fifa_overall": 82.0,
          "Season": "2526"}]
    )
    players = pd.DataFrame(
        {"Player": ["david silva"], "Season": ["2526"], "position_group": ["M"]}
    )
    assert attach_fifa(players, export)["fifa_overall"].isna().all()


def test_nicknames_match_through_the_familiar_name():
    """EA's legal name can share no token at all with the common one.

    Vitinha is "Vitor Machado Ferreira", Casemiro "Carlos Henrique Venancio
    Casimiro". No key built from the legal name can ever reach them, so the
    export's short name is indexed alongside it.
    """
    export = _export(
        [{"fifa_name": "vitor machado ferreira", "fifa_alt_name": "vitinha",
          "fifa_overall": 89.0, "Season": "2526"}]
    )
    players = pd.DataFrame(
        {"Player": ["vitinha"], "Season": ["2526"], "position_group": ["M"]}
    )
    assert attach_fifa(players, export)["fifa_overall"].iloc[0] == 89.0


def test_two_names_for_one_player_do_not_read_as_two_candidates():
    """Indexing both names must not defeat the uniqueness guards.

    "Fabian Ruiz Pena" and "Fabian Ruiz" both end in "ruiz" and both contain
    {fabian, ruiz}; counted twice, the surname and containment tiers would see
    an ambiguity that does not exist and refuse a correct match.
    """
    export = _export(
        [{"fifa_name": "fabian ruiz pena", "fifa_alt_name": "fabian ruiz",
          "fifa_overall": 85.0, "Season": "2526"}]
    )
    players = pd.DataFrame(
        {"Player": ["fabian ruiz"], "Season": ["2526"], "position_group": ["M"]}
    )
    assert attach_fifa(players, export)["fifa_overall"].iloc[0] == 85.0


def test_genuine_ambiguity_is_still_refused_with_alt_names():
    """Two different players sharing a surname must still resolve to neither."""
    export = _export(
        [
            {"fifa_name": "carlos silva santos", "fifa_alt_name": "carlinhos",
             "fifa_overall": 80.0, "Season": "2526"},
            {"fifa_name": "pedro silva costa", "fifa_alt_name": "pedrinho",
             "fifa_overall": 78.0, "Season": "2526"},
        ]
    )
    players = pd.DataFrame(
        {"Player": ["silva"], "Season": ["2526"], "position_group": ["M"]}
    )
    assert attach_fifa(players, export)["fifa_overall"].isna().all()


def test_hyphenated_surnames_split_into_tokens():
    """Understat spells Mbappé "Mbappe-Lottin"; EA writes "Mbappé Lottin".

    Deleting the hyphen welds it into "mbappelottin", which matches nothing.
    """
    from footballml.players.fbref import normalise_name

    assert normalise_name("Kylian Mbappe-Lottin") == "kylian mbappe lottin"
    # Apostrophes are still stripped rather than split: one name, not two.
    assert normalise_name("N'Golo Kante") == "ngolo kante"


def _with_clubs(ea_rows: list[dict], players: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A fixture where every club's EA name is learnable from exact matches.

    `at_club` deliberately never compares club strings -- it learns what EA
    calls each of our teams from players that matched by name -- so a club-aware
    test needs at least `CLUB_EVIDENCE_MIN` exact matches per club.
    """
    clubs = {"Aston Villa": "Aston Villa", "Kilmarnock": "Kilmarnock FC",
             "Bournemouth": "AFC Bournemouth", "Barcelona": "FC Barcelona"}
    rows, squad = list(ea_rows), list(players)
    for team, club in clubs.items():
        for i in range(3):
            name = f"{team.lower().replace(' ', '')} filler {i}"
            rows.append({"fifa_name": name, "fifa_overall": 70.0,
                         "fifa_club": club, "Season": "2526"})
            squad.append({"Player": name, "Team": team})
    frame = pd.DataFrame(squad)
    frame["Season"] = "2526"
    # Only where a case has not said otherwise: the position-contradiction
    # tests turn on this column, so it cannot be overwritten for everyone.
    if "position_group" not in frame.columns:
        frame["position_group"] = "M"
    frame["position_group"] = frame["position_group"].fillna("M")
    return frame, _export(rows)


def test_a_diminutive_matches_on_initial_and_surname():
    """"Ollie Watkins" against EA's "Oliver George Arthur Watkins"."""
    players, export = _with_clubs(
        [
            {"fifa_name": "oliver george arthur watkins", "fifa_club": "Aston Villa",
             "fifa_overall": 84.0, "Season": "2526"},
            {"fifa_name": "marley joseph watkins", "fifa_club": "Kilmarnock FC",
             "fifa_overall": 65.0, "Season": "2526"},
        ],
        [{"Player": "ollie watkins", "Team": "Aston Villa"},
         {"Player": "marley watkins", "Team": "Kilmarnock"}],
    )
    matched = attach_fifa(players, export)
    assert _overall(matched, "ollie watkins", "Aston Villa") == 84.0
    assert _overall(matched, "marley watkins", "Kilmarnock") == 65.0


def test_transliterated_given_names_share_an_initial():
    """"Djordje Petrovic" against "Đorđe Petrović" -- both initials give d."""
    players, export = _with_clubs(
        [{"fifa_name": "đorđe petrović", "fifa_club": "AFC Bournemouth",
          "fifa_overall": 80.0, "Season": "2526"},
         {"fifa_name": "rasmus niklasson petrovic", "fifa_club": "GAIS",
          "fifa_overall": 63.0, "Season": "2526"}],
        [{"Player": "djordje petrovic", "Team": "Bournemouth"}],
    )
    assert _overall(attach_fifa(players, export), "djordje petrovic", "Bournemouth") == 80.0


def test_initial_and_surname_alone_never_decide_it():
    """The club is required, because the key itself is weak evidence.

    EA files Barcelona's Alex Balde as "Alejandro Balde Martinez" -- the last
    token is the maternal surname -- so the only entry keyed "a balde" was
    Aliou Balde of St. Gallen. Taken unopposed that swapped an 83-rated
    starter for a 66-rated stranger, dropping him to 54.
    """
    players, export = _with_clubs(
        [{"fifa_name": "aliou badara balde", "fifa_club": "FC St.Gallen 1879",
          "fifa_overall": 66.0, "Season": "2526"},
         {"fifa_name": "mama samba balde", "fifa_club": "Stade Brestois 29",
          "fifa_overall": 73.0, "Season": "2526"}],
        [{"Player": "alex balde", "Team": "Barcelona"}],
    )
    assert pd.isna(_overall(attach_fifa(players, export), "alex balde", "Barcelona"))


# --- carrying a confirmed match into an edition that renamed the player -----
#
# EA does not spell a player the same way twice, and the two failure shapes are
# opposites. "Vini Jr." shares no token at all with "Vinícius Júnior", so every
# key in the loose matcher is looking for something that is not there. "Fermín"
# against "Fermín López" fails the other way: EA's name is a strict subset, and
# the containment tier only looks for supersets. Neither is guessable from the
# strings, and both are already answered in another season's edition.


def _two_edition_clubs(
    ea_rows: list[dict], players: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`_with_clubs`, but spanning two editions.

    Both the club map and the variant map are learned from matched rows, so a
    cross-edition test needs `CLUB_EVIDENCE_MIN` exact matches per club *per
    season* -- the club map is built per edition precisely because EA renames
    clubs between them.
    """
    clubs = {"Real Madrid": {"2526": "Real Madrid", "2627": "Real Madrid CF"},
             "Barcelona": {"2526": "FC Barcelona", "2627": "FC Barcelona"}}
    rows, squad = list(ea_rows), list(players)
    for team, per_season in clubs.items():
        for season, club in per_season.items():
            for i in range(3):
                name = f"{team.lower().replace(' ', '')} filler {i}"
                rows.append({"fifa_name": name, "fifa_overall": 70.0,
                             "fifa_club": club, "Season": season})
                squad.append({"Player": name, "Team": team, "Season": season})
    frame = pd.DataFrame(squad)
    frame["position_group"] = "M"
    return frame, _export(rows)


VINI = [
    {"fifa_name": "vinicius jose paixao de oliveira junior", "fifa_alt_name": "vini jr",
     "fifa_club": "Real Madrid", "fifa_overall": 89.0, "Season": "2526"},
    {"fifa_name": "vini jr", "fifa_club": "Real Madrid CF",
     "fifa_overall": 91.0, "Season": "2627"},
]


def test_a_nickname_only_edition_inherits_the_match():
    """The 2627 entry is reachable only through the 2526 entry's alt name."""
    players, export = _two_edition_clubs(
        VINI,
        [{"Player": "vinicius junior", "Team": "Real Madrid", "Season": s}
         for s in ("2526", "2627")],
    )
    matched = attach_fifa(players, export)
    got = matched.set_index("Season").loc[
        matched["Player"].eq("vinicius junior").to_numpy(), "fifa_overall"
    ]
    assert got.loc["2526"] == 89.0
    assert got.loc["2627"] == 91.0


def test_a_shortened_name_inherits_the_match():
    """EA's name is a subset of ours, which no ordered or containment key
    catches: "Fermín" against "Fermín López"."""
    players, export = _two_edition_clubs(
        [
            {"fifa_name": "fermin lopez marin", "fifa_alt_name": "fermin",
             "fifa_club": "FC Barcelona", "fifa_overall": 80.0, "Season": "2526"},
            {"fifa_name": "fermin", "fifa_club": "FC Barcelona",
             "fifa_overall": 85.0, "Season": "2627"},
        ],
        [{"Player": "fermin lopez", "Team": "Barcelona", "Season": s}
         for s in ("2526", "2627")],
    )
    matched = attach_fifa(players, export)
    got = matched.set_index("Season").loc[
        matched["Player"].eq("fermin lopez").to_numpy(), "fifa_overall"
    ]
    assert got.loc["2627"] == 85.0


def test_the_club_is_required_not_merely_a_tiebreak():
    """Two men share one performance-source name, so the variant map is
    polluted by construction.

    Understat calls both Luis Alberto Suárez Díaz and Luis Javier Suárez
    Charris "Luis Suárez". Whichever one matches first lends his EA names to
    the other, and the nickname tested here is reachable by no other route --
    so without the club requirement the second man silently takes the first
    man's rating. The names are deliberately opaque: a realistic pair would be
    caught by the surname tier and prove nothing about this one.
    """
    players, export = _two_edition_clubs(
        [
            {"fifa_name": "qoltan", "fifa_alt_name": "zevi",
             "fifa_club": "FC Barcelona", "fifa_overall": 77.0, "Season": "2526"},
            {"fifa_name": "zevi", "fifa_club": "Real Madrid CF",
             "fifa_overall": 78.0, "Season": "2627"},
        ],
        [{"Player": "qoltan", "Team": "Barcelona", "Season": "2526"},
         {"Player": "qoltan", "Team": "Barcelona", "Season": "2627"}],
    )
    matched = attach_fifa(players, export)
    later = matched[matched["Player"].eq("qoltan") & matched["Season"].eq("2627")]
    assert pd.isna(later["fifa_overall"].iloc[0]), (
        "a variant was taken at a club the player never played for"
    )


def test_a_variant_claimed_by_two_players_identifies_neither():
    """A key that names several players names none of them -- the same rule
    every other tier is held to.

    Two players carry the same familiar name in one edition, so that name
    cannot say which of them the next edition's entry is.
    """
    players, export = _two_edition_clubs(
        [
            {"fifa_name": "qoltan", "fifa_alt_name": "zevi",
             "fifa_club": "FC Barcelona", "fifa_overall": 80.0, "Season": "2526"},
            {"fifa_name": "brunex", "fifa_alt_name": "zevi",
             "fifa_club": "FC Barcelona", "fifa_overall": 79.0, "Season": "2526"},
            {"fifa_name": "zevi", "fifa_club": "FC Barcelona",
             "fifa_overall": 85.0, "Season": "2627"},
        ],
        [{"Player": "qoltan", "Team": "Barcelona", "Season": "2526"},
         {"Player": "brunex", "Team": "Barcelona", "Season": "2526"},
         {"Player": "qoltan", "Team": "Barcelona", "Season": "2627"}],
    )
    matched = attach_fifa(players, export)
    later = matched[matched["Player"].eq("qoltan") & matched["Season"].eq("2627")]
    assert pd.isna(later["fifa_overall"].iloc[0])


def test_nothing_already_matched_is_disturbed():
    """The tier only ever fills a gap; it never revises a decision the
    per-season matcher made."""
    players, export = _two_edition_clubs(
        VINI,
        [{"Player": "vinicius junior", "Team": "Real Madrid", "Season": s}
         for s in ("2526", "2627")],
    )
    before = attach_fifa(players, export)
    assert before["fifa_overall"].notna().sum() == len(before)


# --- a match to the wrong man usually disagrees about the position ---------
#
# Name keys cannot tell two people apart when they share a name, and the
# damage is not limited to the rating: `role` comes from the EA entry too, so a
# wrong match moves the player between lines of `team_strength`. Aston Villa's
# Emiliano Martínez spent two seasons and 6,500 minutes as a midfielder,
# leaving the club with no goalkeeper line at all.


def test_a_goalkeeper_is_never_matched_to_an_outfielder():
    """Both directions, and the single boundary nobody crosses.

    "Emiliano Martínez" is a goalkeeper at Aston Villa and an outfielder at
    Club Nacional, and the surname tier took the wrong one unopposed.
    """
    players, export = _with_clubs(
        [{"fifa_name": "emiliano martinez toranza", "fifa_club": "Club Nacional",
          "fifa_overall": 68.0, "role": "MID", "Season": "2526"}],
        [{"Player": "emiliano martinez", "Team": "Aston Villa", "position_group": "GK"}],
    )
    matched = attach_fifa(players, export)
    assert pd.isna(_overall(matched, "emiliano martinez", "Aston Villa"))


def test_the_right_man_is_found_once_the_wrong_one_is_ruled_out():
    """Filtering before the tiers count candidates, not after one picks.

    With the Uruguayan removed the bucket holds exactly one plausible entry,
    so the uniqueness guard resolves instead of refusing.
    """
    players, export = _with_clubs(
        [{"fifa_name": "emiliano martinez toranza", "fifa_club": "Club Nacional",
          "fifa_overall": 68.0, "role": "MID", "Season": "2526"},
         {"fifa_name": "damian emiliano martinez", "fifa_club": "Aston Villa",
          "fifa_overall": 84.0, "role": "GK", "Season": "2526"}],
        [{"Player": "emiliano martinez", "Team": "Aston Villa", "position_group": "GK"}],
    )
    matched = attach_fifa(players, export)
    assert _overall(matched, "emiliano martinez", "Aston Villa") == 84.0


def test_an_exact_name_match_is_checked_too():
    """Two different men can share a normalised name.

    Atalanta's midfielder Éderson and Manchester City's keeper Ederson both
    reduce to "ederson", and the exact merge runs before any tier that could
    have second thoughts.
    """
    players, export = _with_clubs(
        [{"fifa_name": "ederson", "fifa_club": "Manchester City",
          "fifa_overall": 88.0, "role": "GK", "Season": "2526"}],
        [{"Player": "ederson", "Team": "Barcelona", "position_group": "M"}],
    )
    matched = attach_fifa(players, export)
    assert pd.isna(_overall(matched, "ederson", "Barcelona"))


def test_a_forward_is_not_a_centre_back():
    """Espanyol's Sergio García carried Sergio Ramos' rating for three
    seasons -- 87, 90 and 91 -- on a shared surname."""
    players, export = _with_clubs(
        [{"fifa_name": "sergio ramos garcia", "fifa_club": "Real Madrid CF",
          "fifa_overall": 91.0, "role": "CB", "Season": "2526"}],
        [{"Player": "sergio garcia", "Team": "Barcelona", "position_group": "F"}],
    )
    matched = attach_fifa(players, export)
    assert pd.isna(_overall(matched, "sergio garcia", "Barcelona"))


def test_a_wing_back_filed_as_a_defender_still_matches():
    """The boundaries are narrow on purpose. Understat calls wing-backs "D"
    and EA calls them LM or RW -- 1,073 player-seasons, nearly all correct --
    so that disagreement must not be read as two different people."""
    players, export = _with_clubs(
        [{"fifa_name": "marc cucurella saseta", "fifa_club": "FC Barcelona",
          "fifa_overall": 81.0, "role": "AMW", "Season": "2526"}],
        [{"Player": "marc cucurella", "Team": "Barcelona", "position_group": "D"}],
    )
    matched = attach_fifa(players, export)
    assert _overall(matched, "marc cucurella", "Barcelona") == 81.0


def test_a_withdrawn_forward_still_matches_a_midfielder():
    """Forward-to-midfield is deliberately not a contradiction: false nines
    and withdrawn forwards cross it constantly."""
    players, export = _with_clubs(
        [{"fifa_name": "lionel andres messi cuccittini", "fifa_club": "FC Barcelona",
          "fifa_overall": 93.0, "role": "MID", "Season": "2526"}],
        [{"Player": "lionel messi", "Team": "Barcelona", "position_group": "F"}],
    )
    matched = attach_fifa(players, export)
    assert _overall(matched, "lionel messi", "Barcelona") == 93.0


def test_a_sole_surname_holder_is_not_proof_on_its_own():
    """Understat writes Kylian Mbappe as "Mbappe-Lottin" and EA's 2024/25
    export does not, so the only entry keyed "lottin" was Albert-Nicolas
    Lottin of CD Castellon. Taken unopposed, a 91-rated striker became a
    65-rated defensive midfielder."""
    players, export = _with_clubs(
        [{"fifa_name": "albert nicolas lottin", "fifa_club": "CD Castellon",
          "fifa_overall": 65.0, "role": "FWD", "Season": "2526"}],
        [{"Player": "kylian mbappe-lottin", "Team": "Barcelona", "position_group": "F"}],
    )
    matched = attach_fifa(players, export)
    assert pd.isna(_overall(matched, "kylian mbappe-lottin", "Barcelona"))


def test_a_bare_surname_at_another_club_still_matches():
    """The guard must not take the ordinary case with it. EA stores plenty of
    Spanish players as a surname alone -- "Alena", "Reguilon", "Terrats" --
    where no given name can ever agree, and those are exactly the players
    most likely to have moved, so the club cannot rescue them either. EA's
    name being a strict subset of ours is enough.
    """
    players, export = _with_clubs(
        [{"fifa_name": "alena", "fifa_club": "Getafe CF",
          "fifa_overall": 73.0, "role": "MID", "Season": "2526"}],
        [{"Player": "carles alena", "Team": "Barcelona", "position_group": "M"}],
    )
    matched = attach_fifa(players, export)
    assert _overall(matched, "carles alena", "Barcelona") == 73.0


def test_every_role_is_accounted_for_in_the_compatibility_table():
    """A role missing from the table rejects wholesale, not narrowly.

    Splitting `DM` out of `MID` and forgetting to list it here made every
    CDM-primary entry in the database contradict every player: the match rate
    fell from 92% to 87% in a single run, and the only symptom was a number in
    a log line. The two tables have to be checked against each other, not
    maintained in parallel by hand.
    """
    from footballml.players.fifa import COMPATIBLE_ROLES, ROLE_BY_POSITION

    roles = set(ROLE_BY_POSITION.values())
    for group, allowed in COMPATIBLE_ROLES.items():
        assert allowed <= roles, f"{group} allows a role that does not exist"
    outfield = roles - {"GK"}
    covered = set().union(*(a for g, a in COMPATIBLE_ROLES.items() if g != "GK"))
    assert covered == outfield, (
        f"outfield roles never allowed anywhere: {outfield - covered}"
    )


def test_the_goalkeeper_boundary_is_the_only_total_one():
    """Everything else must be crossable from somewhere, or it is a blanket
    rejection wearing the costume of a rule."""
    from footballml.players.fifa import COMPATIBLE_ROLES

    assert COMPATIBLE_ROLES["GK"] == {"GK"}
    for group, allowed in COMPATIBLE_ROLES.items():
        if group != "GK":
            assert "GK" not in allowed, f"{group} should never reach a goalkeeper"


# --- containment, the other way round ---------------------------------------
#
# Every tier asks whether our tokens appear in EA's. None asked the reverse, so
# an EA entry that is a *shorter* form of our name was unreachable however
# plainly it was the same man.


def test_an_ea_name_inside_ours_is_matched():
    """Understat writes Kylian Mbappe as "Mbappe-Lottin"; EA's FC24 and FC25
    exports say "Kylian Mbappe". Right player, right club, right position, and
    not one key reached him -- so the league's best player carried no EA rating
    in two seasons out of three.
    """
    players, export = _with_clubs(
        [{"fifa_name": "kylian mbappe", "fifa_club": "Real Madrid CF",
          "fifa_overall": 91.0, "role": "FWD", "Season": "2526"}],
        [{"Player": "kylian mbappe-lottin", "Team": "Barcelona", "position_group": "F"}],
    )
    # The club deliberately does not agree: he had just moved, which is exactly
    # when EA's snapshot is stale and the name is all there is to go on.
    matched = attach_fifa(players, export)
    assert _overall(matched, "kylian mbappe-lottin", "Barcelona") == 91.0


def test_a_one_token_ea_name_does_not_reach_through_this_tier():
    """Two tokens minimum, because one inside ours is a much weaker claim.

    A lone *given* name is the clear case: "Thiago" sits inside "Thiago Silva"
    and inside every other Thiago in the database, and unlike a lone surname it
    has no tier of its own that could weigh it. (A lone surname does -- the
    surname tier, which accepts a strict subset deliberately, so that EA's
    bare "Alena" still reaches Carles Alena.)
    """
    players, export = _with_clubs(
        [{"fifa_name": "thiago", "fifa_club": "FC St.Gallen 1879",
          "fifa_overall": 62.0, "role": "MID", "Season": "2526"}],
        [{"Player": "thiago silva", "Team": "Barcelona", "position_group": "D"}],
    )
    assert pd.isna(_overall(attach_fifa(players, export), "thiago silva", "Barcelona"))


def test_two_shorter_ea_names_inside_ours_need_the_club():
    """Ambiguity is refused here as everywhere else."""
    players, export = _with_clubs(
        [{"fifa_name": "carlos vinicius", "fifa_club": "Aston Villa",
          "fifa_overall": 74.0, "role": "FWD", "Season": "2526"},
         {"fifa_name": "vinicius santos", "fifa_club": "Kilmarnock FC",
          "fifa_overall": 66.0, "role": "FWD", "Season": "2526"}],
        [{"Player": "carlos vinicius santos", "Team": "Aston Villa",
          "position_group": "F"}],
    )
    # Both are subsets of ours; the club picks the right one.
    assert _overall(attach_fifa(players, export),
                    "carlos vinicius santos", "Aston Villa") == 74.0


def test_the_position_check_still_applies_to_this_tier():
    """A shorter name is no reason to skip the guard the other tiers use."""
    players, export = _with_clubs(
        [{"fifa_name": "kylian mbappe", "fifa_club": "Real Madrid CF",
          "fifa_overall": 91.0, "role": "GK", "Season": "2526"}],
        [{"Player": "kylian mbappe-lottin", "Team": "Barcelona", "position_group": "F"}],
    )
    assert pd.isna(_overall(attach_fifa(players, export),
                            "kylian mbappe-lottin", "Barcelona"))


# --- a role must not follow EA's position ordering --------------------------


def _kante(orderings: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One player across several seasons, with EA's position string per season."""
    export = pd.DataFrame(
        [
            {
                "fifa_name": "ngolo kante",
                "_norm": "ngolo kante",
                "fifa_overall": 80.0,
                "fifa_positions": positions,
                "role": primary_role(positions),
                "Season": season,
            }
            for season, positions in orderings.items()
        ]
    )
    players = pd.DataFrame(
        {
            "Player": ["ngolo kante"] * len(orderings),
            "Season": list(orderings),
            "season_position": ["M"] * len(orderings),
            "position_group": ["M"] * len(orderings),
        }
    )
    return players, export


def test_reordering_the_same_positions_does_not_change_the_role():
    """Kanté is a defensive midfielder for nine straight seasons at one club.

    EA writes him "CM, CDM" through FC17 and "CDM, CM" after. `primary_role`
    takes the first token, so the pool he is ranked in changed from central
    midfielders to defensive ones in the middle of his Chelsea career -- on an
    ordering EA re-sorts between releases, with no transfer and no conversion.
    640 player-seasons flipped this way.
    """
    players, export = _kante(
        {
            "1516": "CM, CDM",
            "1617": "CM, CDM",
            "1718": "CDM, CM",
            "1819": "CDM, CM",
            "1920": "CDM, CM",
        }
    )

    roles = attach_fifa(players, export)["role"]

    assert set(roles) == {"DM"}, "one position set, one role"


def test_a_genuine_position_change_still_moves_the_role():
    """The guard must be narrow enough to let a real conversion through.

    A winger who becomes a full back carries a *different* EA position set, and
    a rule keyed on the set leaves him alone. The wider rules measured against
    this one -- one role per career, or the set deciding globally -- froze
    3,490 seasons that belong to conversions like this.
    """
    players, export = _kante(
        {"1819": "LW, LM", "1920": "LW, LM", "2021": "LB, LWB", "2122": "LB, LWB"}
    )

    roles = attach_fifa(players, export).set_index("Season")["role"]

    assert roles["1819"] == "AMW"
    assert roles["2122"] == "FB"


def test_a_tie_takes_the_later_seasons_reading():
    """Two seasons each way is EA changing its mind, not noise.

    Spelled out in the code rather than left to `value_counts` tie ordering,
    which is not a documented guarantee -- so it is pinned here.
    """
    players, export = _kante(
        {"1819": "CM, CDM", "1920": "CM, CDM", "2021": "CDM, CM", "2122": "CDM, CM"}
    )

    roles = attach_fifa(players, export)["role"]

    assert set(roles) == {"DM"}


def test_a_player_with_no_ea_entry_is_left_to_the_fallback():
    """No position set, nothing to settle. The Understat fallback still rules."""
    export = _export(
        [{"fifa_name": "someone else", "fifa_overall": 70.0, "Season": "2526"}]
    )
    players = pd.DataFrame(
        {
            "Player": ["unmatched player"],
            "Season": ["2526"],
            "season_position": ["D"],
            "position_group": ["D"],
        }
    )

    assert attach_fifa(players, export)["role"].iloc[0] == "CB"


# --- the same name, written in the other order -----------------------------


def test_a_reversed_name_order_matches_the_right_player():
    """Understat writes Son Heung-Min; EA writes "Heung Min Son".

    Identical tokens, no exact string match. On its own that would cost a
    rating. What made it worth its own tier is what filled the gap: a later
    tier matched him to *Son Seung Min*, a different player rated 49, and
    blended that in -- 2024/25 published 66 against a performance rating of
    84.3, while the right entry sat in the same file at 87.
    """
    export = _export(
        [
            {"fifa_name": "Heung Min Son", "fifa_overall": 87.0, "Season": "2425"},
            {"fifa_name": "Son Seung Min", "fifa_overall": 49.0, "Season": "2425"},
        ]
    )
    players = pd.DataFrame(
        {
            "Player": ["Son Heung-Min"],
            "Season": ["2425"],
            "season_position": ["F"],
            "position_group": ["F"],
        }
    )

    out = attach_fifa(players, export)

    assert out["fifa_overall"].iloc[0] == 87.0, "matched the wrong Korean player"


def test_an_anagram_of_two_players_is_left_to_the_club_tiers():
    """Ambiguity is not a licence to guess.

    If two EA entries reduce to the same set of tokens, the set says nothing
    about which man this is, and the tiers below have the club to lean on.
    """
    export = _export(
        [
            {"fifa_name": "Lee Jae Sung", "fifa_overall": 78.0, "Season": "2425"},
            {"fifa_name": "Jae Sung Lee", "fifa_overall": 59.0, "Season": "2425"},
        ]
    )
    players = pd.DataFrame(
        {
            "Player": ["Sung Lee Jae"],
            "Season": ["2425"],
            "season_position": ["M"],
            "position_group": ["M"],
        }
    )

    assert attach_fifa(players, export)["fifa_overall"].isna().all()


def test_the_token_set_never_crosses_the_goalkeeper_boundary():
    """Same tokens, wrong man: the one boundary that is proof of a different
    person has to hold here as it does in every other tier."""
    export = _export(
        [{"fifa_name": "Min Heung Son", "fifa_overall": 87.0, "Season": "2425",
          "role": "FWD"}]
    )
    players = pd.DataFrame(
        {
            "Player": ["Son Heung Min"],
            "Season": ["2425"],
            "season_position": ["GK"],
            "position_group": ["GK"],
        }
    )

    assert attach_fifa(players, export)["fifa_overall"].isna().all()
