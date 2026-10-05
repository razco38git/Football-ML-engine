"""A pipeline must not die because the console cannot spell a player's name.

On 2026-10-05 the weekly job's `fetch_european` step exited 1 with

    UnicodeEncodeError: 'charmap' codec can't encode character '\u011f'

-- a Turkish g-breve, in the list of unmatched clubs it prints for review. The
fetch itself had succeeded and written 822 matches; the step died on the report
about them, and the weekly job continued without the European data it had just
downloaded.

The console encoding here is cp1255, which cannot represent an accented `e`
either, so "Mbappe" and "Kante" were equally fatal. Every pipeline that prints a
name was one accented name from the same crash.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Characters cp1255 cannot represent: a Turkish g-breve and an acute e.
AWKWARD = "Ba\u011fak\u015fehir Mbapp\u00e9"


def test_cp1255_really_cannot_spell_these() -> None:
    """Pins the premise. If this ever passes, the rest of the file is moot."""
    with pytest.raises(UnicodeEncodeError):
        AWKWARD.encode("cp1255")


def test_printing_an_unencodable_name_does_not_raise() -> None:
    """The actual contract: `print` degrades, it does not explode.

    Run in a subprocess with the broken encoding forced, because the fix
    reconfigures the real stdout and cannot be exercised in-process.
    """
    done = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, r'{ROOT}'); "
            f"import pipelines; print({AWKWARD!r})",
        ],
        cwd=ROOT,
        capture_output=True,
        env={"PYTHONIOENCODING": "cp1255", "PATH": ""},
        check=False,
    )

    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr
    # Degraded to question marks rather than lost or fatal.
    assert b"Ba?ak?ehir Mbapp?" in done.stdout


def test_the_weekly_runner_asks_its_children_for_utf8() -> None:
    """Replacement is the fallback, not the plan.

    The runner decodes its children as utf-8, so it has to tell them to encode
    that way -- otherwise every accented name in the weekly log is a `?` even
    though nothing went wrong.
    """
    source = (ROOT / "pipelines" / "weekly.py").read_text(encoding="utf-8")
    assert '"PYTHONIOENCODING": "utf-8"' in source
    assert "env=env," in source


def test_a_stream_without_reconfigure_is_not_an_error() -> None:
    """pytest's capture, and anything a caller substitutes, lack it."""
    import pipelines

    original = sys.stdout
    try:
        sys.stdout = io.StringIO()  # no `reconfigure`
        pipelines._tolerate_unencodable_output()
    finally:
        sys.stdout = original
