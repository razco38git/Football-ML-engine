"""Command-line entry points. See each module's docstring for what it does.

This package exists mainly for the side effect below. ``python -m
pipelines.<name>`` imports it before the module it was asked for, which makes it
the one place that can protect every pipeline at once.

**Printing a club's name must not be able to kill a pipeline.** On 2026-10-05
the weekly job's `fetch_european` step died with::

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u011f'

That is a Turkish ``g`` with a breve, in the list of unmatched clubs the step
prints for a human to read. Nothing was wrong with the fetch: 822 matches had
been pulled and written. The step crashed on the *report about* them, exited 1,
and the weekly job carried on without the European data it had just downloaded.

The cause is the console encoding, not the data. Python encodes stdout with
``locale.getpreferredencoding()``, which on this machine is **cp1255** -- a
Hebrew codepage that cannot represent ``g`` with a breve, and cannot represent
``e`` with an acute either. So "Mbappe" and "Kante" are equally fatal, and every
pipeline that prints a player or club name was one accented name away from the
same failure. That the top-20 lists had not hit it yet was luck.

``errors="replace"`` keeps each stream's own encoding and substitutes ``?`` for
anything it cannot represent. A name may read as ``Ba?ak?ehir`` on a Hebrew
console, which is a legible degradation; the alternative was a dead step and a
stack trace pointing at a print statement.

`weekly.run_step` additionally runs its children under ``PYTHONIOENCODING=utf-8``
so the scheduled path keeps the real characters rather than falling back to this.
"""

from __future__ import annotations

import contextlib
import sys


def _tolerate_unencodable_output() -> None:
    """Never let an unrepresentable character raise out of a `print`.

    Deliberately silent and deliberately total. Anything that goes wrong here
    is less important than the pipeline that is about to run, and a stream
    without `reconfigure` -- pytest's capture, a plain `io.StringIO` a caller
    has substituted -- is a normal thing to be handed, not an error.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # Detached, closed, or not a text stream: all normal, none fatal.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(errors="replace")


_tolerate_unencodable_output()
