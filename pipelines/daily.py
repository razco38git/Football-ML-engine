"""Daily results refresh: get played matches onto the site the morning after.

The weekly job is the one that rebuilds everything. This is the small one that
runs in between, and it exists because the projected tables and the recent
results were only ever as fresh as the last Monday. A match played on Tuesday
sat unpublished for six days, and on 2026-10-11 the tables were still showing
five matchdays while two more had been played.

**It changes nothing about how a number is produced.** The same
``build_dataset`` brings the same football-data rows with their shots, corners
and xG, and ``project_season`` simulates from the same fully-built feature
table. Nothing here fills a gap with a partial row: if football-data has not
published a match, this job does not invent it, and the tables wait.

What it deliberately does *not* do is the expensive weekly work:

``build_players`` (7m 21s)
    Player ratings and squad strength are whole-season numbers over a minimum
    of 450 minutes, gated at eleven rated players per club. They move slowly
    enough that a day makes no visible difference, and rebuilding them daily
    would cost two thirds of the run for nothing.
``train`` (53s)
    Skipped on purpose, and not to save the minute. Retraining mints a model
    version, so a daily retrain would fill `/models` with near-identical rows
    and scatter the live track record across seven versions a week instead of
    one -- which makes the accuracy page harder to read, for a model that would
    differ by a handful of matches. Between Mondays the site already serves
    predictions from the last weekly model; this keeps that true.

So a projection refreshed here uses the newest *results* with the current
*model*, which is exactly the arrangement the Match Predictor has had all along.

**Nothing new is not an error.** Most days football-data publishes no match
this job has not already seen. That is not a reason to skip settling -- the
schedule pages routinely know a result days earlier -- but it is a reason not
to re-simulate a table built from rows that did not change.

Run::

    python -m pipelines.daily
    python -m pipelines.daily --force     # re-project even with no new matches

**Settling runs every day; projecting runs only when the results table moved.**
Settlement reads two sources -- football-data, and the schedule pages, which
carry a score days before football-data publishes one -- so it has work to do on
a day when the fetch brought nothing. The projection simulates from the results
table alone, so with no new rows it would spend four minutes rewriting the file
it already wrote.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines._lock import holder, refresh_lock  # noqa: E402
from pipelines.weekly import (  # noqa: E402
    API_URL,
    Step,
    _configure_logging,
    _duration,
    reload_api,
    run_step,
)

ROOT = Path(__file__).resolve().parents[1]
MATCHES = ROOT / "data" / "processed" / "matches_all.csv"

logger = logging.getLogger("daily")

FETCH = Step("build_dataset", (), "fetch match results")

#: Settling runs every day, whatever football-data did.
#:
#: It reads two sources -- football-data, and the schedule pages, which carry a
#: score days before football-data publishes one -- so it has work to do on days
#: when the fetch brought nothing.
SETTLE_ONLY = Step(
    "score_upcoming", ("--settle-only",), "settle what was played -- the fast path"
)

SETTLE = Step("score_upcoming", (), "settle what was played, re-forecast the next round")

#: Projecting runs only when the results table actually moved.
#:
#: `project_season` simulates from that table and nothing else, so with no new
#: rows it would spend four minutes reproducing the file it already wrote.
PROJECT = Step("project_season", (), "re-simulate the rest of the season")

#: Both, for callers that want the full chain.
STEPS: tuple[Step, ...] = (SETTLE, PROJECT)


def _fingerprint() -> tuple[int, str]:
    """How much football the results table currently knows about.

    Row count *and* latest date, because either can move without the other: a
    correction to an old match changes the count and not the date, and a
    rescheduled fixture arriving late changes neither in a way a count alone
    would catch.
    """
    if not MATCHES.exists():
        return (0, "")
    frame = pd.read_csv(MATCHES, usecols=["Date"])
    if frame.empty:
        return (0, "")
    return (len(frame), str(pd.to_datetime(frame["Date"]).max().date()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Run the downstream steps even when no new matches arrived.",
    )
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--verbose", action="store_true", help="Log pipeline output too.")
    args = parser.parse_args()

    log_path = _configure_logging(args.verbose, name="daily")
    logger.info("Daily refresh starting; logging to %s", log_path.name)
    started = time.monotonic()

    # The weekly job does everything this one does and more, so if it is
    # running there is nothing to wait for. Both missed triggers firing
    # together is not hypothetical: on 2026-10-12 the machine was off all
    # morning and Windows started the two four seconds apart.
    with refresh_lock("daily") as acquired:
        if not acquired:
            logger.info(
                "Another refresh is running (%s); stopping. It rebuilds "
                "everything this job would have.", holder() or "unknown",
            )
            return 0
        return _run(args, started)


def _run(args: argparse.Namespace, started: float) -> int:
    # Settle first, publish, *then* do the slow work.
    #
    # The order used to be the other way round, and it stopped working the day
    # `build_dataset` grew the FBref top-up: that scrapes a page per team at a
    # seven-second rate limit whenever football-data is behind, and
    # football-data has been behind since 2026-09-20, so it is behind every
    # morning. On 2026-10-13 the job started at 08:55 and was still inside the
    # fetch at 09:28 with sixteen of the previous day's predictions unsettled.
    #
    # Settling needs none of it. It reads the schedule -- five pages, cached for
    # twelve hours -- and the history already on disk. The expensive step exists
    # for the *projection*; Recent results, Prediction Accuracy and Discover
    # only need the cheap one, and now get it about a minute in rather than
    # twenty-five.
    published = False
    try:
        run_step(SETTLE_ONLY)
        published = reload_api(args.api_url)
        if published:
            logger.info(
                "Settled and published in %s. The rest of this run is the "
                "projection, which nobody is waiting on.",
                _duration(time.monotonic() - started),
            )
        else:
            logger.error(
                "Settled, but the API is still serving the old copy. Start it, "
                "or POST /admin/reload, to publish it."
            )
    except subprocess.CalledProcessError as exc:
        # Not fatal. The fetch below may still bring results worth projecting,
        # and a later run settles what this one could not.
        logger.error(
            "Settling failed (exit %d); carrying on to the fetch. The tabs keep "
            "whatever was last settled.", exc.returncode,
        )

    before = _fingerprint()
    try:
        run_step(FETCH)
    except subprocess.CalledProcessError as exc:
        # football-data being unreachable is a normal morning. Anything settled
        # above is already live, so this is the end of a useful run rather than
        # a failure of one.
        logger.error(
            "build_dataset failed (exit %d). Nothing further ran; the settled "
            "results above are published and the tables keep yesterday's "
            "numbers. Tomorrow's run will try again.",
            exc.returncode,
        )
        return 0 if published else 1
    after = _fingerprint()
    advanced = after != before

    if advanced:
        logger.info(
            "New results: %d rows (was %d), latest %s (was %s)",
            after[0], before[0], after[1] or "n/a", before[1] or "n/a",
        )
    else:
        logger.info(
            "football-data published nothing new (%d rows, latest %s); the "
            "tables would be rebuilt identically, so they are left alone",
            after[0], after[1] or "n/a",
        )

    # Re-forecast on the fresh history, and re-simulate only if it moved.
    # `project_season` reads the results table and nothing else, so with no new
    # rows it would spend four minutes rewriting the file it already wrote.
    steps = [SETTLE, PROJECT] if (advanced or args.force) else []
    if not steps:
        logger.info("Nothing left to do. Done in %s", _duration(time.monotonic() - started))
        return 0

    for step in steps:
        try:
            run_step(step)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "%s failed (exit %d). Stopping: a half-refreshed site is worse "
                "than one that is a day behind, because nothing on the page says "
                "which half is which. Rerun with `python -m pipelines.weekly "
                "--only %s` once the cause is fixed.",
                step.name, exc.returncode, step.name,
            )
            return 1

    if not reload_api(args.api_url):
        logger.error(
            "The data is built and on disk but the API is still serving the old "
            "copy. Start it, or POST /admin/reload, to publish it."
        )
        return 1

    logger.info("Daily refresh finished in %s", _duration(time.monotonic() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
