"""Cache for the FBref player-stat read, one block per stat type and season.

`build_players` rebuilds every season from 2014/15 on every weekly run. Measured
on 2026-10-13, that is 346s, of which **339s is the ingest** and only 2.1s is
computing the ratings. Split by source:

===============================  ======
understat read_player_season      6.9s
fbref fetch_fbref_stats         150.0s
fifa load_fifa                    6.4s
our own name matching           ~176.0s
===============================  ======

A season that finished years ago cannot read differently, so the FBref figure is
pure waste. This caches it, and nothing else.

Why nothing else
----------------
The obvious bigger win is to cache each finished season's *derived* frame, or
its ratings. Both are wrong, and the reasons are worth stating because they look
season-local and are not:

- `_career_position` resolves a player to one position across their whole
  career by total minutes -- the fix that stopped Szoboszlai, Laimer and
  Zaire-Emery being rated as defenders. A player who converts position is
  relabelled in *every* season at once, which moves their old seasons into a
  different ``(role, Season)`` percentile pool.
- `_match_across_editions` deliberately runs only once every season has been
  matched, because that is what gives it a body of confirmed names to carry
  into the editions that spell a player differently.
- `_settle_reordered_roles` compares a player's seasons against each other, so
  that Kante reading ``"CM, CDM"`` in one edition and ``"CDM, CM"`` in the next
  does not move him between rating pools mid-career.

Deriving seasons one at a time was built and measured before this was written:
it changed 103 EA matches and was reverted. Each of those three corrections
exists because of a specific defect it fixed, and a per-season cache would quietly
undo all three.

The fingerprint
---------------
A cache that outlives the code that filled it fails invisibly -- the numbers
still load and still look plausible. So the key covers the module that produces
the block, the leagues asked for, and a manual :data:`CACHE_VERSION`. Since
nothing but `fbref.py` and soccerdata decides what a block holds, that is the
whole surface; a change anywhere else cannot reach it.

Pickle rather than CSV, because dtypes matter downstream: the ingest hands
nullable ``Float64`` columns to `rate_players`, which has a documented failure
that appears only with those dtypes intact. Reading back through CSV flattens
them and would make the cached path behave differently from the fresh one.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import pandas as pd

from footballml.data import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache" / "fbref-players"

#: Bump to invalidate every cached block. For a change that alters what a read
#: returns without touching `fbref.py` -- a soccerdata upgrade that parses a
#: column differently, say.
CACHE_VERSION = 1

#: The only module whose code decides what a cached block contains.
_SOURCE_FILES = (Path(__file__).with_name("fbref.py"),)

_fingerprint: str | None = None


def fingerprint() -> str:
    """A short digest of everything that decides a cached block's contents."""
    global _fingerprint
    if _fingerprint is not None:
        return _fingerprint

    digest = hashlib.sha256(f"v{CACHE_VERSION}".encode())
    for path in _SOURCE_FILES:
        try:
            digest.update(path.read_bytes())
        except OSError:
            # Unable to say what the code does, so make the key unique and let
            # the block be rebuilt.
            digest.update(f"missing:{path.name}".encode())

    _fingerprint = digest.hexdigest()[:16]
    return _fingerprint


def request_key(leagues: list[str], stat_type: str) -> str:
    """Identifies *what was asked for*, so a narrower read cannot serve a wider one.

    A block read for ``--leagues E0`` holds only English players, and serving it
    to a five-league run would silently drop four leagues from the ratings.
    """
    leagues_key = hashlib.sha256("".join(sorted(leagues)).encode()).hexdigest()[:8]
    return f"{stat_type}-{leagues_key}"


def _path(season: str, request: str) -> Path:
    return CACHE_DIR / f"{season}-{request}-{fingerprint()}.pkl"


def load(season: str, request: str) -> pd.DataFrame | None:
    """A cached block, or ``None`` when there is no usable one.

    Never raises. A cache that cannot be read is a miss, which costs the read it
    was meant to save -- not the run.
    """
    path = _path(season, request)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            frame = pickle.load(handle)  # noqa: S301 - written by this module only
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as exc:
        logger.warning("Ignoring unreadable cache %s (%s)", path.name, exc)
        return None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        logger.warning("Ignoring cache %s: not a populated frame", path.name)
        return None
    return frame


def save(season: str, request: str, frame: pd.DataFrame) -> None:
    """Store a block and drop that season's blocks under older fingerprints.

    Never raises, for the same reason as :func:`load`: an optimisation must not
    be able to fail a build.
    """
    if frame.empty:
        return
    path = _path(season, request)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(frame, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as exc:
        logger.warning("Could not write cache %s (%s)", path.name, exc)
        return

    for stale in CACHE_DIR.glob(f"{season}-{request}-*.pkl"):
        if stale != path:
            try:
                stale.unlink()
            except OSError:  # noqa: PERF203 - one failure must not stop the rest
                logger.debug("Could not remove stale cache %s", stale.name)


def clear() -> int:
    """Remove every cached block. Returns how many files went."""
    if not CACHE_DIR.exists():
        return 0
    gone = 0
    for path in CACHE_DIR.glob("*.pkl"):
        try:
            path.unlink()
            gone += 1
        except OSError:  # noqa: PERF203 - best effort
            logger.debug("Could not remove %s", path.name)
    return gone
