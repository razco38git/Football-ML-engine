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

    async def run_briefly():
        task = asyncio.create_task(app_module._warm_upcoming())
        while len(calls) < 3:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(run_briefly(), timeout=10))

    assert len(calls) >= 3


def test_the_warmer_ticks_sooner_than_the_cache_expires():
    """The margin is the whole design: a stale cache is refilled before anyone
    can ask for it, rather than at the moment somebody does."""
    assert app_module.FIXTURE_WARM_INTERVAL < app_module.FIXTURE_CACHE_TTL
