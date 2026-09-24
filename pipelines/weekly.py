"""Weekly refresh: results, ratings, model, predictions, and the live site.

The Prediction Accuracy page tells readers its record grows a few matches per
week. Nothing made that true. On 2026-09-22 the match data was eight days old,
eighteen pre-kickoff predictions sat unsettled, and refreshing them surfaced a
crash that had silently broken every prediction path for days. A site nobody
feeds goes stale quietly, which is the worst way for it to happen.

This runs the existing pipelines in the one order that works, and refuses to
half-finish::

    python -m pipelines.weekly
    python -m pipelines.weekly --only score_upcoming
    python -m pipelines.weekly --skip build_players

Order is not arbitrary:

1. ``build_dataset`` fetches results. Everything below reads them.
2. ``build_players`` rebuilds player ratings and ``team_strength.csv``.
3. ``train`` retrains on the new matches -- and reads the strength file, so it
   must come after step 2.
4. ``score_upcoming`` settles predictions whose matches were played, then
   forecasts the coming round using ``registry.load()``, so it must come after
   step 3 to use the model just trained.
5. ``project_season`` simulates every remaining fixture into a projected table,
   also on the freshly trained model.
6. ``reload`` tells a running API to re-read all of it.

Each step is timed and logged to ``logs/weekly-<date>.log`` as well as stdout,
because a scheduled run nobody watched still has to be readable afterwards.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"

#: Where the API listens. Matches the uvicorn invocation in the README.
API_URL = "http://127.0.0.1:8000"

logger = logging.getLogger("weekly")


@dataclass(frozen=True)
class Step:
    """One pipeline invocation."""

    name: str
    args: tuple[str, ...]
    why: str


#: The weekly run, in dependency order. See the module docstring.
STEPS: tuple[Step, ...] = (
    Step("build_dataset", (), "fetch match results"),
    Step(
        "build_players",
        ("--start-season", "1516"),
        "rebuild player ratings and team strength",
    ),
    Step("train", (), "retrain on the new matches"),
    Step("score_upcoming", (), "settle played predictions, forecast the next round"),
    Step("project_season", (), "simulate the rest of the season for each league"),
)

#: Not a pipeline module -- handled in-process, see `reload_api`.
RELOAD = "reload"


def _configure_logging(verbose: bool) -> Path:
    """Log to stdout and to a dated file. Returns the file path."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"weekly-{datetime.now(UTC):%Y-%m-%d}.log"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return path


def run_step(step: Step) -> float:
    """Run one pipeline, streaming its output into our log.

    Raises:
        subprocess.CalledProcessError: If the pipeline exits non-zero. The
            caller stops the run: continuing would leave some files refreshed
            and others not, which reads as a successful update.
    """
    logger.info("--> %s (%s)", step.name, step.why)
    started = time.monotonic()

    # `sys.executable`, not "python": under Task Scheduler the PATH is not the
    # one an interactive shell gets, and a bare "python" would either miss the
    # virtualenv or not exist at all.
    # One merged stream rather than two. A pipeline's own logging goes to
    # stderr, its prints to stdout, and its dependencies choose for themselves;
    # keeping them apart meant guessing which one carried the sentence that
    # said what actually happened.
    completed = subprocess.run(
        [sys.executable, "-m", f"pipelines.{step.name}", *step.args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.monotonic() - started
    output = completed.stdout or ""

    for line in output.splitlines():
        logger.debug("    %s", line)
    if completed.returncode != 0:
        for line in output.splitlines()[-25:]:
            logger.error("    %s", line)
        raise subprocess.CalledProcessError(completed.returncode, step.name, output)

    # Discarding a successful step's output is how a run that did nothing reads
    # exactly like a run that did everything -- the failure this whole module
    # exists to prevent. The tail is enough to say what happened: "stored 48
    # predictions", or "next fixture is 9 October".
    for line in _summary_lines(output):
        logger.info("    %s", line)

    logger.info("    done in %s", _duration(elapsed))
    return elapsed


#: Lines of a successful step's log to carry up into the weekly log.
SUMMARY_LINES = 8


def _summary_lines(output: str | None) -> list[str]:
    """The tail of a step's own log, as a readable summary of what it did."""
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    return lines[-SUMMARY_LINES:]


def reload_api(url: str = API_URL, timeout: float = 120.0) -> bool:
    """Ask a running API to re-read the files this run rewrote.

    A stopped API is not a failure -- the files are on disk and the next start
    picks them up. Only a running-but-broken one is worth shouting about.
    """
    logger.info("--> %s (tell the site to pick up the new data)", RELOAD)
    try:
        with urlopen(  # noqa: S310 - fixed localhost URL
            Request(f"{url}/admin/reload", method="POST"), timeout=timeout
        ) as response:
            logger.info("    API reloaded: %s", response.read().decode("utf-8")[:200])
        return True
    except URLError as exc:
        logger.warning("    API not reachable (%s) -- it will load on next start", exc.reason)
        return False


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    names = [step.name for step in STEPS] + [RELOAD]
    parser.add_argument(
        "--only", choices=names, action="append",
        help="Run just this step. Repeatable. Useful for rerunning one failure.",
    )
    parser.add_argument(
        "--skip", choices=names, action="append", default=[],
        help="Skip this step. Repeatable.",
    )
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--verbose", action="store_true", help="Log pipeline output too.")
    args = parser.parse_args()

    log_path = _configure_logging(args.verbose)
    wanted = set(args.only) if args.only else set(names)
    wanted -= set(args.skip)

    logger.info("Weekly refresh starting; logging to %s", log_path.name)
    started = time.monotonic()
    ran: list[str] = []

    for step in STEPS:
        if step.name not in wanted:
            logger.info("--> %s skipped", step.name)
            continue
        try:
            run_step(step)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "%s failed (exit %d). Stopping: a partial refresh looks like a "
                "complete one, and the site would serve a mix of old and new. "
                "Fix it, then rerun just this step with --only %s",
                step.name, exc.returncode, step.name,
            )
            return 1
        ran.append(step.name)

    if RELOAD in wanted:
        reload_api(args.api_url)
        ran.append(RELOAD)

    logger.info(
        "Weekly refresh finished in %s (%s)", _duration(time.monotonic() - started),
        ", ".join(ran) or "nothing to do",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
