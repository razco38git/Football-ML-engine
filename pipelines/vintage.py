"""Pin and fingerprint an experiment's inputs.

Research-harness only. Nothing in ``src/footballml`` imports this, and
production behaviour is unchanged.

Why it exists
-------------

The squad-size sweep was re-run on what looked like identical inputs and
produced a different table: attack peaked at 16 rather than 15, defence was best
at 16 rather than climbing to 17. A stage-by-stage diff located the divergence
precisely, and it was **not** the model.

``fetch_player_seasons`` caches *finished* seasons only. The season in progress
is re-fetched from Understat on every run, and it grows as matches are played --
measured directly, 2,195 player-seasons to 2,199 inside twenty-five minutes on
one evening. Everything downstream of that is deterministic: with the input held
fixed, two separate processes produce bit-identical goal rates, bit-identical
scoreline matrices and a bit-identical backtest, to the last decimal of RPS.

So the harness was reproducible and its inputs were not.

The in-progress season also **leaks into finished-season results**, which is the
part that makes this worth fixing rather than merely recording. It supplies 3 of
1,173 team-seasons, and ``_season_outcomes`` excludes it entirely, so it should
be inert -- yet rating the frame with it against without it moves the
finished-season correlations by up to **0.0031**:

    with the in-progress season     atk 0.73947  def -0.52705  ovr 0.75556
    without it                      atk 0.73844  def -0.52394  ovr 0.75501

That is the same order as the differences the sweep was trying to resolve
between adjacent squad sizes, which is why its ordering did not replicate.

.. note::
    **The propagation path is not fully established.** Comparing the two rated
    frames row by row, ``rating``, ``performance_rating``, ``fifa_overall`` and
    ``measured_share`` are *identical* for every finished-season player, yet the
    selected squads differ. Something other than those four columns carries it.
    The effect is reproducible and measured; the mechanism is not, and saying so
    is more useful than guessing.
"""

from __future__ import annotations

import hashlib

import pandas as pd

#: Seasons `fetch_player_seasons` serves from cache. Anything after the newest
#: of these is in progress and is re-fetched live on every run.
CACHED_THROUGH = "2526"


def current_season(players: pd.DataFrame) -> str | None:
    """The newest season in the frame, when it is past the cached window.

    Returns ``None`` when every season present is a finished, cached one -- so
    this is a no-op out of season rather than something to remember to disable.
    """
    if "Season" not in players.columns or players.empty:
        return None
    newest = max(players["Season"].astype(str))
    return newest if newest > CACHED_THROUGH else None


def pin_finished_seasons(players: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Drop the in-progress season, so a run does not depend on the clock.

    The season in progress contributes a handful of team-seasons that no outcome
    pair can use -- ``_season_outcomes`` gates them out on matches played -- and
    its presence perturbs the finished seasons it should not touch. Removing it
    costs the experiment nothing it was measuring and makes the run depend only
    on files that are not moving.

    Returns the pinned frame and the season dropped, so the caller can report it.
    """
    season = current_season(players)
    if season is None:
        return players, None
    return players[players["Season"].astype(str) != season].copy(), season


def fingerprint(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """A short, order-insensitive content hash of a frame.

    Sorted before hashing so that a reordering is not mistaken for a change --
    the question being asked is "is this the same data", not "is this the same
    file".
    """
    f = frame[columns] if columns else frame
    f = f.reindex(sorted(f.columns), axis=1)
    f = f.sort_values(list(f.columns), kind="mergesort").reset_index(drop=True)
    return hashlib.md5(
        pd.util.hash_pandas_object(f, index=False).to_numpy().tobytes()
    ).hexdigest()[:12]


def describe_inputs(**frames: pd.DataFrame) -> str:
    """One line per input frame: rows and content hash, for the run log.

    Printing this beside a result is what turns "these two tables disagree and
    nobody knows why" into a one-line diff. The squad-size discrepancy took a
    stage-by-stage bisection to locate; with this in the log it would have been
    visible immediately.
    """
    lines = ["input vintage:"]
    for name, frame in frames.items():
        lines.append(f"  {name:<12} rows={len(frame):>7}  hash={fingerprint(frame)}")
    return "\n".join(lines)
