"""A maintenance page must never be stored over a good Understat payload.

On 2026-11-02 understat.com started answering every request with HTTP 200 and a
holding page. soccerdata writes the body to the cache before anyone parses it,
so each failed weekly replaced the first league's stored payload with that page
-- `league_3_season_2026.json`, the Bundesliga data the shipped
`player_ratings.csv` was built from, was last rewritten at 10:21 on 2026-11-23
and it was the only raw copy. The weekly then died on a `JSONDecodeError` with
nothing to say why.

These tests drive the real `soccerdata.Understat` subclass with a stub session
rather than testing the validator alone, because the failure was in the wiring:
a correct check that the write path never calls protects nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from footballml.ingest import understat_cache as uc

sd = pytest.importorskip("soccerdata")

ROOT = Path(__file__).resolve().parents[1]

#: The shape of what understat.com served from 2 November: 200 OK, HTML.
HOLDING_PAGE = (
    b"<!DOCTYPE html><html><head><title>Understat.com</title></head>"
    b"<body><p>We will come back soon.</p></body></html>"
)
URL = "https://understat.com/getLeagueData/EPL/2026"


def league_payload(*player_ids: str) -> bytes:
    body = {"dates": [], "players": [{"id": i} for i in player_ids], "teams": {}}
    return json.dumps(body).encode()


class _Response:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.content = body
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    """Stands in for the HTTP session, and records whether it was used."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls: list[str] = []

    def get(self, url: str, **_: object) -> _Response:
        self.calls.append(url)
        return _Response(self.body)


def make_reader(tmp_path: Path, body: bytes, **kwargs: object):
    reader = uc.understat_reader(["ENG-Premier League"], ["2627"], data_dir=tmp_path, **kwargs)
    reader._session = _Session(body)
    return reader


# --- the incident -------------------------------------------------------------


def test_a_maintenance_page_does_not_overwrite_a_good_payload(tmp_path: Path) -> None:
    """The exact failure: 200 OK, HTML, over a payload that was fine."""
    stored = tmp_path / "league_3_season_2026.json"
    stored.write_bytes(league_payload("1", "2", "3"))
    before = stored.read_bytes()

    reader = make_reader(tmp_path, HOLDING_PAGE)
    with pytest.raises(uc.UnusablePayloadError, match="left untouched"):
        reader._request_api(URL, stored, no_cache=True)

    assert stored.read_bytes() == before


def test_a_maintenance_page_is_not_stored_where_nothing_was(tmp_path: Path) -> None:
    stored = tmp_path / "league_3_season_2026.json"
    reader = make_reader(tmp_path, HOLDING_PAGE)

    with pytest.raises(uc.UnusablePayloadError, match="nothing was stored"):
        reader._request_api(URL, stored, no_cache=True)

    assert not stored.exists()


def test_the_error_names_what_was_served_instead_of_bare_json_noise(tmp_path: Path) -> None:
    """The old failure was `Expecting value: line 1 column 1 (char 0)`."""
    reader = make_reader(tmp_path, HOLDING_PAGE)
    with pytest.raises(uc.UnusablePayloadError) as caught:
        reader._request_api(URL, tmp_path / "league_3_season_2026.json", no_cache=True)

    message = str(caught.value)
    assert "not JSON" in message
    assert "DOCTYPE" in message, "the start of the body is what identifies the page"
    assert "league_3_season_2026.json" in message


def test_valid_json_that_is_not_a_league_payload_is_refused(tmp_path: Path) -> None:
    """A rate-limit reply parses, and would otherwise replace the data."""
    stored = tmp_path / "league_1_season_2026.json"
    stored.write_bytes(league_payload("1"))
    before = stored.read_bytes()

    reader = make_reader(tmp_path, b'{"error": "rate limited"}')
    with pytest.raises(uc.UnusablePayloadError, match="not a league-season payload"):
        reader._request_api(URL, stored, no_cache=True)

    assert stored.read_bytes() == before


def test_the_league_shape_is_only_demanded_of_league_files(tmp_path: Path) -> None:
    """Narrow on purpose: other cache files are judged on being JSON, no more."""
    reader = make_reader(tmp_path, b'{"anything": [1, 2]}')
    target = tmp_path / "match_31948.json"

    reader._request_api("https://understat.com/getMatchData/31948", target, no_cache=True)

    assert json.loads(target.read_bytes()) == {"anything": [1, 2]}


# --- the guard must not break what worked -------------------------------------


def test_a_good_payload_replaces_the_stored_one_whole(tmp_path: Path) -> None:
    stored = tmp_path / "league_1_season_2026.json"
    stored.write_bytes(league_payload("old"))
    fresh = league_payload("1", "2", "3", "4")

    reader = make_reader(tmp_path, fresh)
    served = reader._request_api(URL, stored, no_cache=True).read()

    assert served == fresh
    assert stored.read_bytes() == fresh
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != stored.name]
    assert leftovers == [], f"the temporary file was not cleaned up: {leftovers}"


def test_a_stored_file_is_served_without_touching_the_network(tmp_path: Path) -> None:
    """Finished seasons come from cache. If this ever went to the network, an
    outage would turn every historical read into a failed fetch -- the guard
    repeats soccerdata's own condition for this, so it is what notices if that
    condition changes."""
    stored = tmp_path / "league_1_season_2025.json"
    stored.write_bytes(league_payload("kept"))

    reader = make_reader(tmp_path, HOLDING_PAGE)
    served = reader._request_api(URL, stored, no_cache=False).read()

    assert served == stored.read_bytes()
    assert reader._session.calls == []


def test_the_season_in_progress_is_refetched_even_when_stored(tmp_path: Path) -> None:
    """`no_cache=True` is how soccerdata asks for a current season."""
    stored = tmp_path / "league_1_season_2026.json"
    stored.write_bytes(league_payload("old"))

    reader = make_reader(tmp_path, league_payload("new"))
    reader._request_api(URL, stored, no_cache=True)

    assert reader._session.calls == [URL]
    assert b"new" in stored.read_bytes()


def test_no_store_still_stores_nothing(tmp_path: Path) -> None:
    stored = tmp_path / "league_1_season_2026.json"
    reader = make_reader(tmp_path, league_payload("1"), no_store=True)

    served = reader._request_api(URL, stored, no_cache=True).read()

    assert served == league_payload("1")
    assert not stored.exists()


# --- the wiring ---------------------------------------------------------------


def test_the_guarded_reader_overrides_the_write_path() -> None:
    """If soccerdata renames `_request_api`, this class would quietly stop
    guarding anything, and every other test here would still read the old name."""
    cls = uc._guarded_class()
    assert "_request_api" in vars(cls), "the guard no longer overrides the write path"
    assert hasattr(sd.Understat, "_request_api")


def test_the_subclass_keeps_the_name_soccerdata_uses_to_find_leagues() -> None:
    """`_all_leagues` keys on `cls.__name__`; any other name finds no leagues."""
    cls = uc._guarded_class()
    assert cls.__name__ == "Understat"
    assert cls._all_leagues() == sd.Understat._all_leagues()
    assert "ENG-Premier League" in cls._all_leagues()


def test_nothing_builds_a_raw_understat_reader() -> None:
    """The one call site that forgets is the one that overwrites a good payload
    on the next outage. It was three, and this is how they stay one."""
    pattern = re.compile(r"\b(?:sd|soccerdata)\.Understat\(")
    offenders = [
        f"{path.relative_to(ROOT)}:{n}"
        for folder in ("src", "pipelines")
        for path in (ROOT / folder).rglob("*.py")
        if path.name != "understat_cache.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith("#")
    ]
    assert offenders == [], f"build the reader with understat_reader(): {offenders}"


# --- looking for damage that is already there ---------------------------------


def test_unusable_payloads_finds_only_the_damaged_files(tmp_path: Path) -> None:
    (tmp_path / "league_1_season_2026.json").write_bytes(league_payload("1"))
    (tmp_path / "league_3_season_2026.json").write_bytes(HOLDING_PAGE)
    (tmp_path / "league_2_season_2025.json").write_bytes(b"")

    bad = [p.name for p in uc.unusable_payloads(tmp_path, uc.LEAGUE_PAYLOADS)]

    assert bad == ["league_2_season_2025.json", "league_3_season_2026.json"]


def test_a_rebuild_from_cache_says_what_is_wrong_instead_of_a_json_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--understat-from-cache` reads whatever is stored. With a holding page in
    there it used to die on 'Expecting value: line 1 column 1 (char 0)'."""
    from footballml.players import ingest

    (tmp_path / "league_3_season_2026.json").write_bytes(HOLDING_PAGE)

    class _Reader:
        data_dir = tmp_path

        def read_player_season_stats(self, **_: object) -> None:  # pragma: no cover
            raise AssertionError("should not get as far as reading")

    monkeypatch.setattr(ingest, "understat_reader", lambda *a, **k: _Reader())

    with pytest.raises(uc.UnusablePayloadError, match=r"league_3_season_2026\.json"):
        ingest.fetch_player_seasons(["D1"], ["2627"], understat_from_cache=True)
