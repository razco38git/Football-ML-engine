"""Tests for the cache that keeps scraping off the request path.

Scoring upcoming fixtures needs the season schedule, and fetching that goes
through soccerdata's FBref reader: a headless browser with a seven-second rate
limit per page. Measured on 2026-10-04, with the result cached for thirty
minutes, that was 71s for the five domestic leagues and 14s for the Champions
League -- and the same again on an immediately repeated call, because
`read_schedule` refuses its own cache for a season still in progress.

So whoever opened Match Predictor first after the cache expired waited a minute
and a half. What these pin is that the rebuild happens once, off the request
path, and that a concurrent caller waits for it rather than starting a second
scrape of a rate-limited source.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

# The module, not the FastAPI instance: `footballml/api/__init__.py` rebinds the
# package attribute `app` to the application object. See `test_reload_guard`.
app_module = importlib.import_module("footballml.api.app")


@pytest.fixture
def stale_cache():
    """An expired cache, restored afterwards so the suite stays order-independent."""
    state = app_module.state
    before = (state._upcoming, state._upcoming_at)
    state._upcoming, state._upcoming_at = None, None
    yield state
    state._upcoming, state._upcoming_at = before


def test_a_fresh_cache_is_not_rebuilt(stale_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "_score_upcoming", lambda: calls.append(1) or [])
    stale_cache._upcoming = []
    stale_cache._upcoming_at = datetime.now(UTC)

    app_module._refresh_upcoming()

    assert calls == []


def test_an_expired_cache_is_rebuilt(stale_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "_score_upcoming", lambda: calls.append(1) or [])
    stale_cache._upcoming = []
    expired = app_module.FIXTURE_CACHE_TTL + timedelta(minutes=1)
    stale_cache._upcoming_at = datetime.now(UTC) - expired

    app_module._refresh_upcoming()

    assert calls == [1]


def test_concurrent_callers_scrape_once(stale_cache, monkeypatch):
    """The reason the lock re-checks staleness inside itself.

    Two threads finding an empty cache must not both hit FBref: the source is
    rate-limited, and the second scrape would be of pages the first has just
    fetched.
    """
    calls = []
    started = threading.Event()

    def slow_score():
        calls.append(1)
        started.set()
        # Long enough that the second thread is certainly waiting on the lock.
        threading.Event().wait(0.2)
        return []

    monkeypatch.setattr(app_module, "_score_upcoming", slow_score)

    threads = [threading.Thread(target=app_module._refresh_upcoming) for _ in range(4)]
    for t in threads:
        t.start()
    started.wait(timeout=5)
    for t in threads:
        t.join(timeout=10)

    assert calls == [1]
    assert stale_cache._upcoming_at is not None


def test_the_warmer_keeps_going_after_a_failure(stale_cache, monkeypatch):
    """A scraper outage must cost freshness, not the warmer.

    If the loop died on the first exception the cache would never refill again
    for the life of the process, and the failure would be invisible until
    someone noticed the fixtures had stopped moving.
    """
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("FBref said no")

    monkeypatch.setattr(app_module, "_refresh_upcoming", flaky)
    monkeypatch.setattr(app_module, "FIXTURE_WARM_INTERVAL", timedelta(seconds=0))

    stop = threading.Event()

    def flaky_then_stop():
        flaky()
        if len(calls) >= 3:
            stop.set()

    monkeypatch.setattr(app_module, "_refresh_upcoming", flaky_then_stop)
    warmer = threading.Thread(target=app_module._warm_upcoming, args=(stop,), daemon=True)
    warmer.start()
    warmer.join(timeout=10)

    assert not warmer.is_alive(), "the warmer ignored its stop flag"
    assert len(calls) >= 3


def test_shutting_down_does_not_wait_for_a_scrape_in_flight(monkeypatch):
    """The API must stop promptly while the warmer is mid-fetch.

    `_refresh_upcoming` drives a headless browser at seven seconds a page --
    85s measured for the six competitions, and unbounded while soccerdata
    retries. The warmer used to be an `asyncio.to_thread` task that shutdown
    cancelled and then awaited; `to_thread` cannot be cancelled, so the await
    only returned when the scrape did.

    Every graceful stop therefore hung on a web scrape, for a cache that does
    not survive the process. It also made `uvicorn --reload` useless and sent
    `scripts/start_site.ps1` to `Stop-Process -Force`.

    This drives the real lifespan, not the warmer in isolation, because the
    await that blocked lived in the lifespan rather than in the loop.
    """
    started, release = threading.Event(), threading.Event()

    def slow_scrape():
        started.set()
        release.wait(timeout=30)

    monkeypatch.setattr(app_module, "_refresh_upcoming", slow_scrape)
    monkeypatch.setattr(app_module, "_load_state", lambda startup=False: None)

    async def start_then_stop():
        async with app_module.lifespan(app_module.app):
            # Yield to the loop rather than blocking it. A `threading.Event`
            # wait here would starve a warmer implemented as a task, and this
            # test would then pass for the wrong reason -- reporting "never
            # started" when what it exists to catch is "stopped too slowly".
            deadline = time.monotonic() + 10
            while not started.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert started.is_set(), "the warmer never started"

    began = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(start_then_stop(), timeout=20))
        elapsed = time.monotonic() - began
    finally:
        release.set()

    assert elapsed < 2.0, f"shutdown waited {elapsed:.1f}s for the scrape"


def test_the_warmer_thread_cannot_hold_the_process_open(monkeypatch):
    """A daemon, so a scrape in flight does not delay interpreter exit.

    `stop.set()` ends the loop at the *end* of the current pass, so without
    this a stop during a fetch would still wait out the fetch at exit.
    """
    seen = {}
    real_thread = threading.Thread

    def capture(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        seen["daemon"] = thread.daemon
        seen["name"] = thread.name
        return thread

    monkeypatch.setattr(app_module, "_load_state", lambda startup=False: None)
    monkeypatch.setattr(app_module, "_refresh_upcoming", lambda: None)
    monkeypatch.setattr(app_module.threading, "Thread", capture)

    async def start_then_stop():
        async with app_module.lifespan(app_module.app):
            pass

    asyncio.run(asyncio.wait_for(start_then_stop(), timeout=20))
    assert seen.get("daemon") is True
    assert seen.get("name") == "fixture-warmer"


def test_the_warmer_ticks_sooner_than_the_cache_expires():
    """The margin is the whole design: a stale cache is refilled before anyone
    can ask for it, rather than at the moment somebody does."""
    assert app_module.FIXTURE_WARM_INTERVAL < app_module.FIXTURE_CACHE_TTL
