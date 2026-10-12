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

    before = _fingerprint()
    try:
        run_step(FETCH)
    except subprocess.CalledProcessError as exc:
        # The usual cause is football-data being unreachable, which happens.
        # Yesterday's tables are correct and complete; today's absence of new
        # ones is not a reason to touch anything.
        logger.error(
            "build_dataset failed (exit %d). Nothing downstream ran, so the site "
            "keeps serving what it had. Tomorrow's run will try again.",
            exc.returncode,
        )
        return 1
    after = _fingerprint()
    advanced = after != before

    if advanced:
        logger.info(
            "New results: %d rows (was %d), latest %s (was %s)",
            after[0], before[0], after[1] or "n/a", before[1] or "n/a",
        )
    else:
        logger.info(
            "football-data published nothing new (%d rows, latest %s)",
            after[0], after[1] or "n/a",
        )

    # `score_upcoming` runs whatever football-data did, and that is the whole
    # correction here. It settles from *two* sources -- football-data, and the
    # schedule pages that carry a score football-data has not published yet --
    # so gating it on football-data advancing gated it on the wrong thing.
    #
    # On 2026-10-12 that is exactly what happened: 51 predictions for the 11th
    # sat unsettled with all seventeen results already on the schedule pages,
    # because football-data had moved nothing and this job stopped after the
    # fetch. Settling is also cheap and idempotent -- `store.append`
    # de-duplicates per model version, so re-forecasting the same fixtures adds
    # nothing.
    steps = [SETTLE, PROJECT] if (advanced or args.force) else [SETTLE]

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
