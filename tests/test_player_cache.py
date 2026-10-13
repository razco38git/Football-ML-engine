"""The FBref player-stat read cache.

`build_players` rebuilt every season from 2014/15 on every weekly run. Measured
on 2026-10-13: 346s total, of which 339s was the ingest and 2.1s the ratings.
Split by source, FBref was 150s of it, and a season that finished years ago
cannot read differently -- so that part is pure waste and is what this caches.

The tempting, larger win is to cache each finished season's derived frame or its
ratings. That was built, measured and reverted: it changed **103 EA matches**,
because three corrections in the ingest deliberately compare a player's seasons
against each other --

- `_career_position`, one position per player across their career by minutes,
  which stopped Szoboszlai being rated as a defender;
- `_match_across_editions`, which runs only once every season is matched so it
  has confirmed names to carry into editions that spell a player differently;
- `_settle_reordered_roles`, so Kante reading "CM, CDM" in one edition and
  "CDM, CM" in the next does not change the pool he is ranked in.

So the boundary is the read, and the tests below pin the two things that make
even that safe: a key that cannot serve the wrong data, and failing open.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from footballml.players import cache


@pytest.fixture(autouse=True)
def cache_in_tmp(tmp_path, monkeypatch):
    """Never touch the real cache, and recompute the fingerprint per test."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "fbref-players")
    monkeypatch.setattr(cache, "_fingerprint", None)
    return tmp_path


def _block(season: str = "1819", rows: int = 3) -> pd.DataFrame:
    """A block shaped like one `fetch_fbref_stats` read."""
    return pd.DataFrame({
        "League": ["E0"] * rows,
        "Season": [season] * rows,
        "Team": ["Arsenal"] * rows,
        "Player": [f"Player {i}" for i in range(rows)],
        # Nullable Float64 on purpose, including a pd.NA: see the dtype note in
        # the module docstring.
        "tackles_won_per90": pd.array(
            [*(round(0.1 * i, 2) for i in range(rows - 1)), None], dtype="Float64"
        ),
    })


_MISC = "misc"


# --- round trip ---------------------------------------------------------------

def test_a_saved_block_comes_back() -> None:
    block = _block()
    key = cache.request_key(["E0"], _MISC)
    cache.save("1819", key, block)
    pd.testing.assert_frame_equal(cache.load("1819", key), block)


def test_dtypes_survive_the_round_trip() -> None:
    """Why this is pickle and not CSV.

    The ingest hands nullable ``Float64`` columns to `rate_players`, which has a
    documented failure that appears only with those dtypes intact -- assigning a
    Float64 block of pd.NA into a float64 column raises. Reading the same data
    back from CSV flattens the dtypes and hides it, so a CSV cache would make
    the cached path behave differently from the fresh one.
    """
    block = _block()
    key = cache.request_key(["E0"], _MISC)
    cache.save("1819", key, block)
    assert cache.load("1819", key)["tackles_won_per90"].dtype == (
        block["tackles_won_per90"].dtype
    )


def test_a_miss_is_none_not_an_error() -> None:
    assert cache.load("1819", cache.request_key(["E0"], _MISC)) is None


def test_an_empty_block_is_not_stored() -> None:
    """Storing nothing would then serve nothing, forever."""
    key = cache.request_key(["E0"], _MISC)
    cache.save("1819", key, _block().iloc[:0])
    assert cache.load("1819", key) is None


# --- what the key has to separate ---------------------------------------------

def test_a_narrower_request_cannot_serve_a_wider_one() -> None:
    """A one-league block served to a five-league run drops four leagues.

    Silently: the ratings file would simply hold fewer players, with nothing on
    the page to say which are missing.
    """
    one = cache.request_key(["E0"], _MISC)
    five = cache.request_key(["D1", "E0", "F1", "I1", "SP1"], _MISC)
    assert one != five

    cache.save("1819", one, _block())
    assert cache.load("1819", five) is None


def test_league_order_does_not_change_the_key() -> None:
    """The same request spelled differently must hit, or nothing ever caches."""
    assert cache.request_key(["E0", "D1"], _MISC) == cache.request_key(
        ["D1", "E0"], _MISC
    )


def test_the_stat_type_separates_the_key() -> None:
    """`misc` and `keeper` are different reads with different columns.

    They are merged afterwards on League/Season/Team/Player, so serving one for
    the other would produce a frame missing half its metrics.
    """
    assert cache.request_key(["E0"], "misc") != cache.request_key(["E0"], "keeper")


def test_a_different_season_is_a_different_block() -> None:
    key = cache.request_key(["E0"], _MISC)
    cache.save("1819", key, _block("1819"))
    assert cache.load("1920", key) is None


# --- staleness ----------------------------------------------------------------

def test_changing_the_code_invalidates_everything(monkeypatch) -> None:
    """A cache that outlives the code that filled it fails invisibly.

    The numbers still load and still look plausible; they merely predate a fix.
    """
    key = cache.request_key(["E0"], _MISC)
    cache.save("1819", key, _block())
    assert cache.load("1819", key) is not None

    before = cache.fingerprint()
    monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
    monkeypatch.setattr(cache, "_fingerprint", None)
    assert cache.fingerprint() != before
    assert cache.load("1819", key) is None


def test_the_fingerprint_covers_the_module_that_fills_it() -> None:
    """Named explicitly, so dropping it from the list is a visible change.

    `fbref.py` and soccerdata are the whole surface here -- nothing else decides
    what a block holds -- which is the narrowness that makes this boundary
    defensible where a derived-season cache was not.
    """
    assert {path.name for path in cache._SOURCE_FILES} == {"fbref.py"}


def test_a_stale_block_is_replaced_not_accumulated() -> None:
    """Otherwise the directory keeps a copy per code change, forever."""
    key = cache.request_key(["E0"], _MISC)
    cache.save("1819", key, _block())
    stale = next(cache.CACHE_DIR.glob("1819-*.pkl"))
    stale.rename(stale.with_name(f"1819-{key}-oldfingerprint.pkl"))

    cache.save("1819", key, _block(rows=4))
    assert len(list(cache.CACHE_DIR.glob("1819-*.pkl"))) == 1


# --- failing open -------------------------------------------------------------

def test_an_unreadable_block_is_a_miss() -> None:
    """An optimisation must not be able to fail the build."""
    key = cache.request_key(["E0"], _MISC)
    cache.save("1819", key, _block())
    path = next(cache.CACHE_DIR.glob("1819-*.pkl"))
    path.write_bytes(b"not a pickle")
    assert cache.load("1819", key) is None


def test_something_that_is_not_a_frame_is_a_miss() -> None:
    key = cache.request_key(["E0"], _MISC)
    cache.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache.CACHE_DIR / f"1819-{key}-{cache.fingerprint()}.pkl"
    with path.open("wb") as handle:
        pickle.dump({"not": "a frame"}, handle)
    assert cache.load("1819", key) is None


def test_clear_removes_every_block() -> None:
    key = cache.request_key(["E0"], _MISC)
    for season in ("1819", "1920", "2021"):
        cache.save(season, key, _block(season))
    assert cache.clear() == 3
    assert cache.load("1819", key) is None


def test_clearing_an_absent_cache_is_not_an_error() -> None:
    assert cache.clear() == 0


# --- the season in progress ---------------------------------------------------

def test_the_latest_season_requested_is_never_cached() -> None:
    """It changes daily, so a cached copy would freeze the live ratings.

    Pinned against the source because the decision lives in the read loop, and
    a cache that quietly started serving the current season would show stale
    player ratings with nothing to indicate it.
    """
    import inspect

    from footballml.players.fbref import fetch_fbref_stats

    source = inspect.getsource(fetch_fbref_stats)
    assert "current_season = max(seasons) if seasons else None" in source
    assert "season != current_season" in source
