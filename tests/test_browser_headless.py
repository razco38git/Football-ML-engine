"""Every browser the scrapers open must be silent unless asked otherwise.

soccerdata's FBref reader defaults to `headless=False`, so each of the six
places that construct one used to pop a Chrome window and steal focus, several
times per run, on a machine somebody is working on.

The cost is real and recorded in `footballml.browser`: soccerdata only attempts
its captcha and Cloudflare handling when the browser is visible, so silence
gives that up. The failure mode is staleness, not bad data -- a failed scrape
falls back to cache and a match that cannot be merged is parked for a later
run.

What these pin is that the choice stays in one place. A seventh construction
site that forgets the argument would silently reintroduce the windows, and
nobody would notice until one appeared.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from footballml.browser import DEFAULT, ENV_VAR, headless

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEARCHED = (ROOT / "src" / "footballml", ROOT / "pipelines")


def _reader_constructions() -> list[tuple[str, int, ast.Call]]:
    """Every call that builds an FBref reader, anywhere in the project."""
    found = []
    for base in SEARCHED:
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == "FBref" or name.startswith("_FBrefWith"):
                    found.append((str(path.relative_to(ROOT)), node.lineno, node))
    return found


def test_the_search_actually_finds_the_readers() -> None:
    """A guard that finds nothing passes for free and protects nothing."""
    assert len(_reader_constructions()) >= 6


@pytest.mark.parametrize("where", _reader_constructions(), ids=lambda w: f"{w[0]}:{w[1]}")
def test_every_reader_is_opened_headless(where) -> None:
    path, line, node = where
    passed = {kw.arg for kw in node.keywords}
    assert "headless" in passed, (
        f"{path}:{line} constructs an FBref reader without `headless=`, so it "
        f"will open a visible Chrome window. Pass `headless=headless()`."
    )


def test_silence_is_the_default() -> None:
    assert DEFAULT is True


def test_the_escape_hatch_works(monkeypatch) -> None:
    """One variable, no code change -- the way back if Cloudflare starts
    refusing headless traffic."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert headless() is True

    for falsey in ("0", "false", "No", " OFF "):
        monkeypatch.setenv(ENV_VAR, falsey)
        assert headless() is False, falsey


def test_an_empty_variable_is_not_a_no(monkeypatch) -> None:
    """`set VAR=` on Windows leaves an empty string rather than unsetting it.
    Reading that as "visible" would turn the windows back on by accident."""
    monkeypatch.setenv(ENV_VAR, "")
    assert headless() is True
