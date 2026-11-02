"""Filing a player-season under a club the player never appeared for.

Understat's ``team_title`` is a comma-joined list of every club a player turned
out for, alphabetically sorted, and soccerdata keeps the first name in it. For
Ligue 1 2026/27 Understat tags all 31 participants of Paris Saint-Germain 2-2
Rennes with *both* clubs, so fifteen Rennes players were published as PSG
players -- 39 player-seasons under `Paris Saint Germain` against six under
`Rennes`, where every other Ligue 1 club had 21 to 27. Both clubs' squad
strength was wrong, and that feeds the match model from 2027/28 on.

See :mod:`footballml.players.clubs`. The rule tested here is that a row may only
be filed under a club FBref also records the player at, and the cases below are
the ones that decide whether it fires.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.players.clubs import club_candidates, resolve_club_attribution


def understat(*rows: tuple[int, str, str]) -> pd.DataFrame:
    """Player-seasons as ``(player_id, player, team)``, all Ligue 1 2026/27."""
    return pd.DataFrame(
        [
            {
                "League": "F1",
                "Season": "2627",
                "player_id": pid,
                "player": name,
                "team": team,
            }
            for pid, name, team in rows
        ]
    )


def fbref(*rows: tuple[str, str, float]) -> pd.DataFrame:
    """FBref rows as ``(player, team, fouls)``, one per club per player.

    ``fouls`` and its per-90 rate are what minutes are recovered from, so a
    fixed rate of 1.0 makes the count stand in for ninetieths of a match.
    """
    return pd.DataFrame(
        [
            {
                "League": "F1",
                "Season": "2627",
                "Player": name,
                "Team": team,
                "fouls": fouls,
                "fouls_per90": 1.0,
            }
            for name, team, fouls in rows
        ]
    )


def test_a_player_is_moved_off_a_club_he_never_played_for() -> None:
    """Cresswell: tagged with both, filed at PSG, only ever a Rennes player."""
    players = understat((1, "Charlie Cresswell", "Paris Saint Germain"))
    candidates = {("F1", "2627", 1): ("Paris Saint Germain", "Rennes")}

    out = resolve_club_attribution(
        players, fbref(("Charlie Cresswell", "Rennes", 9.0)), candidates
    )

    assert out.at[0, "team"] == "Rennes"


def test_a_genuine_transfer_is_left_where_it_is() -> None:
    """The point of the rule, and the reason it is not "pick by minutes".

    Understat's row is one aggregate across both clubs, so neither answer is
    right and moving it is a judgement about data quality rather than a defect.
    FBref records him at the filed club, so the rule stays silent -- even though
    he played far more elsewhere.
    """
    players = understat((2, "Giovane", "Napoli"))
    candidates = {("F1", "2627", 2): ("Napoli", "Verona")}
    evidence = fbref(("Giovane", "Napoli", 3.0), ("Giovane", "Hellas Verona", 17.0))

    out = resolve_club_attribution(players, evidence, candidates)

    assert out.at[0, "team"] == "Napoli"


def test_a_single_club_row_is_never_touched() -> None:
    """Only a row Understat itself called ambiguous is eligible to move."""
    players = understat((3, "Anthony Rouault", "Rennes"))

    out = resolve_club_attribution(
        players, fbref(("Anthony Rouault", "Paris SG", 20.0)), candidates={}
    )

    assert out.at[0, "team"] == "Rennes"


def test_no_fbref_row_leaves_the_club_alone() -> None:
    """Absence of evidence is not evidence, and guessing would cost a right row.

    Mousa Al Tamari is the real instance: FBref spells him ``Musa Al-Taamari``,
    which no name key or containment rule reaches, so he stays at PSG. 120 of
    the 927 multi-club player-seasons have no FBref row at all.
    """
    players = understat((4, "Mousa Al Tamari", "Paris Saint Germain"))
    candidates = {("F1", "2627", 4): ("Paris Saint Germain", "Rennes")}

    out = resolve_club_attribution(
        players, fbref(("Musa Al-Taamari", "Rennes", 9.0)), candidates
    )

    assert out.at[0, "team"] == "Paris Saint Germain"


def test_the_club_names_are_translated_before_they_are_compared() -> None:
    """Understat says ``Paris Saint Germain``; FBref says ``Paris SG``.

    Compared raw, every PSG player looks absent from PSG and the whole squad
    would be refiled -- the defect, with the clubs swapped.
    """
    players = understat((5, "Achraf Hakimi", "Paris Saint Germain"))
    candidates = {("F1", "2627", 5): ("Paris Saint Germain", "Rennes")}

    out = resolve_club_attribution(
        players, fbref(("Achraf Hakimi", "Paris SG", 8.0)), candidates
    )

    assert out.at[0, "team"] == "Paris Saint Germain"


def test_a_name_one_source_prints_shorter_still_resolves() -> None:
    """``William Pacho`` against ``Willian Pacho``: a PSG player, kept at PSG.

    Four of the five PSG players FBref spells differently are recovered this
    way. Without the containment tier each would have looked absent from PSG
    and been refiled to Rennes, which is worse than the bug.
    """
    players = understat(
        (6, "William Pacho", "Paris Saint Germain"),
        (7, "Fabián", "Paris Saint Germain"),
    )
    candidates = {
        ("F1", "2627", 6): ("Paris Saint Germain", "Rennes"),
        ("F1", "2627", 7): ("Paris Saint Germain", "Rennes"),
    }
    evidence = fbref(
        ("Willian Pacho", "Paris SG", 6.0), ("Fabián Ruiz Peña", "Paris SG", 7.0)
    )

    out = resolve_club_attribution(players, evidence, candidates)

    assert list(out["team"]) == ["Paris Saint Germain"] * 2


def test_a_name_naming_two_players_decides_nothing() -> None:
    """André and Amadou Onana share the initial-plus-last key ``a onana``.

    `match_players` drops an ambiguous key rather than taking the first row,
    because resolving it by row order once made the build irreproducible. The
    same key is used here, so it must be dropped here too.
    """
    players = understat((8, "A. Onana", "Paris Saint Germain"))
    candidates = {("F1", "2627", 8): ("Paris Saint Germain", "Rennes")}
    evidence = fbref(("André Onana", "Rennes", 9.0), ("Amadou Onana", "Rennes", 9.0))

    out = resolve_club_attribution(players, evidence, candidates)

    assert out.at[0, "team"] == "Paris Saint Germain"


def test_only_a_club_understat_listed_can_win() -> None:
    """FBref may know of a third club; the candidate set is Understat's."""
    players = understat((9, "Someone Else", "Paris Saint Germain"))
    candidates = {("F1", "2627", 9): ("Paris Saint Germain", "Rennes")}
    evidence = fbref(("Someone Else", "Monaco", 30.0), ("Someone Else", "Rennes", 2.0))

    out = resolve_club_attribution(players, evidence, candidates)

    assert out.at[0, "team"] == "Rennes"


def test_an_empty_candidate_set_is_a_no_op() -> None:
    """What a soccerdata upgrade that hides the payload must degrade to."""
    players = understat((10, "Charlie Cresswell", "Paris Saint Germain"))

    out = resolve_club_attribution(
        players, fbref(("Charlie Cresswell", "Rennes", 9.0)), candidates={}
    )

    assert out.at[0, "team"] == "Paris Saint Germain"


def test_candidates_are_empty_when_the_payload_cannot_be_reached() -> None:
    """A reader without the private method must not fail the build."""

    class Reader:
        pass

    assert club_candidates(Reader()) == {}


def test_only_multi_club_players_become_candidates() -> None:
    """The comma is the whole trigger."""

    class Reader:
        def read_seasons(self):
            return pd.DataFrame(
                [{"league_id": 5, "season_id": 2026, "url": "u"}],
                index=pd.MultiIndex.from_tuples(
                    [("FRA-Ligue 1", "2627")], names=["league", "season"]
                ),
            )

        def _read_league_season(self, url, league_id, season_id, no_cache):
            return {
                "playersData": [
                    {"id": "1", "team_title": "Paris Saint Germain,Rennes"},
                    {"id": "2", "team_title": "Rennes"},
                ]
            }

    assert club_candidates(Reader()) == {
        ("F1", "2627", 1): ("Paris Saint Germain", "Rennes")
    }


def test_a_season_the_caller_did_not_ask_for_is_skipped() -> None:
    class Reader:
        def read_seasons(self):
            return pd.DataFrame(
                [{"league_id": 5, "season_id": 2026, "url": "u"}],
                index=pd.MultiIndex.from_tuples(
                    [("FRA-Ligue 1", "2627")], names=["league", "season"]
                ),
            )

        def _read_league_season(self, url, league_id, season_id, no_cache):
            raise AssertionError("must not be read")

    assert club_candidates(Reader(), seasons=["2526"]) == {}


def test_one_unreadable_season_does_not_lose_the_others() -> None:
    class Reader:
        def read_seasons(self):
            return pd.DataFrame(
                [
                    {"league_id": 5, "season_id": 2026, "url": "bad"},
                    {"league_id": 1, "season_id": 2026, "url": "good"},
                ],
                index=pd.MultiIndex.from_tuples(
                    [("FRA-Ligue 1", "2627"), ("ENG-Premier League", "2627")],
                    names=["league", "season"],
                ),
            )

        def _read_league_season(self, url, league_id, season_id, no_cache):
            if url == "bad":
                raise RuntimeError("payload moved")
            return {
                "playersData": [{"id": "3", "team_title": "Arsenal,Chelsea"}]
            }

    assert club_candidates(Reader()) == {("E0", "2627", 3): ("Arsenal", "Chelsea")}


# --- the audit on the real file ----------------------------------------------


def test_no_club_season_holds_another_clubs_squad() -> None:
    """The regression guard, run against the published player file.

    `resolve`'s own `runner_up` is the ready-made audit: it says how many of a
    club-season's rated players EA places at the *second* most popular label. A
    handful is ordinary -- players who moved in the window, or whom EA lists at
    last season's club. Four or more is a different claim, and before this was
    fixed exactly one club-season in 1,266 made it: Paris Saint-Germain 2026/27,
    with fifteen Rennes players in its squad.

    `MIN_VOTES` is the same number for the same reason, so a runner-up that
    clears it is a second squad with a vote of its own.

    The runner-up is read without comparing it to the label the club ended up
    with, and it has to be: PSG 2026/27 is the case where they *match*. Its
    first-pass vote ranked `Stade Rennais FC` above `Paris SG`, it then yielded
    Rennes' label back to Rennes and re-resolved to `Paris SG` -- so the
    runner-up records nine players at the label it finally took, while the
    fifteen that caused the trouble are no longer in the field at all. What the
    count means is that the vote was split two ways with weight behind both,
    which is the thing worth refusing.
    """
    from footballml.data import PROCESSED_DIR
    from footballml.players.ea_squads import FIFA_DIR, MIN_VOTES, resolve

    path = PROCESSED_DIR / "player_ratings.csv"
    if not path.exists():
        pytest.skip("run `python -m pipelines.build_players` first")
    if not any(FIFA_DIR.glob("fifa_*.csv")):
        pytest.skip(f"no EA editions in {FIFA_DIR}")

    players = pd.read_csv(path, low_memory=False)
    rated = players[players["rated"].fillna(False).astype(bool)]

    contaminated = [
        r
        for r in resolve(rated, FIFA_DIR)
        if r.runner_up and r.runner_up[1] > MIN_VOTES
    ]

    assert not contaminated, "\n".join(
        f"{r.season} {r.team}: vote split, {r.runner_up[1]} rated players at "
        f"{r.runner_up[0]!r} as runner-up; resolved to {r.label!r} "
        f"with {r.votes} by {r.how}"
        for r in contaminated
    )
