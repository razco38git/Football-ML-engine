"""Tests for what a reload does when the model and the code disagree.

On 2026-09-24 the site showed "Cannot reach the API" on Recent results. The
server had been running since the previous day; `POST /admin/reload` re-read the
newly trained artifact -- which expected `home_elo`, `away_elo` and `elo_diff`
-- into a process whose feature builder predated Elo. `_load_state` noticed,
logged an error, and published the model anyway, so every `/matches` call died
with `KeyError: "['home_elo', 'away_elo', 'elo_diff'] not in index"`.

The artifact was fine. The process was the stale half, and the fix was a
restart. What these pin is that a reload in that position **keeps serving what
works** rather than replacing it with something it has already proven it cannot
feed.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

# The module, not the FastAPI instance. `footballml/api/__init__.py` does
# `from footballml.api.app import app`, which rebinds the package attribute
# `app` to the application object -- so both `from footballml.api import app`
# and `import footballml.api.app as api` hand back the FastAPI instance.
api = importlib.import_module("footballml.api.app")


class _Meta:
    """Just enough of registry.ModelMetadata for the compatibility check."""

    def __init__(self, version: str, feature_names: list[str]) -> None:
        self.version = version
        self.feature_names = feature_names


@pytest.fixture
def serving(monkeypatch):
    """A state that is already serving a working model."""
    features = pd.DataFrame({
        "League": ["E0"], "Season": ["2627"], "Date": [pd.Timestamp("2026-09-01")],
        "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"],
        "FTHG": [1.0], "FTAG": [0.0], "FTR": ["H"],
        "home_points_last_5": [2.0], "away_points_last_5": [1.0],
    })
    good = _Meta("good-model", ["home_points_last_5", "away_points_last_5"])

    api.state.model = "working-model"
    api.state.metadata = good
    api.state.features = features
    api.state.columns = ["home_points_last_5", "away_points_last_5"]

    monkeypatch.setattr(api, "load_team_match_history", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(api, "load_team_strength", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(api, "build_match_features", lambda *_a, **_k: features)
    monkeypatch.setattr(api, "feature_columns", lambda f: list(f.columns))
    monkeypatch.setattr(api, "build_index", lambda *_a, **_k: {})
    return features


def test_a_reload_refuses_a_model_it_cannot_feed(serving, monkeypatch):
    monkeypatch.setattr(
        api.registry, "load",
        lambda *_a, **_k: ("new-model", _Meta("needs-elo", ["home_points_last_5", "home_elo"])),
    )

    with pytest.raises(api.IncompatibleModelError) as caught:
        api._load_state()

    assert caught.value.version == "needs-elo"
    assert "home_elo" in caught.value.missing


def test_the_previous_model_is_still_served_afterwards(serving, monkeypatch):
    """The property that actually matters: the site keeps working."""
    monkeypatch.setattr(
        api.registry, "load",
        lambda *_a, **_k: ("new-model", _Meta("needs-elo", ["home_elo"])),
    )

    with pytest.raises(api.IncompatibleModelError):
        api._load_state()

    assert api.state.model == "working-model"
    assert api.state.metadata.version == "good-model"
    assert api.state.columns == ["home_points_last_5", "away_points_last_5"]


def test_a_compatible_model_loads_normally(serving, monkeypatch):
    monkeypatch.setattr(
        api.registry, "load",
        lambda *_a, **_k: ("new-model", _Meta("fine", ["home_points_last_5"])),
    )

    api._load_state()

    assert api.state.model == "new-model"
    assert api.state.metadata.version == "fine"


def test_startup_serves_what_works_rather_than_refusing(serving, monkeypatch):
    """At startup there is no previous model to fall back to.

    Refusing would leave the process with nothing at all; half a site, with the
    error logged loudly, beats none.
    """
    monkeypatch.setattr(
        api.registry, "load",
        lambda *_a, **_k: ("new-model", _Meta("needs-elo", ["home_elo"])),
    )

    api._load_state(startup=True)

    assert api.state.model == "new-model"
    assert api.state.metadata.version == "needs-elo"


# --- the cross-league correction's compatibility verdict ----------------------
#
# The offsets describe one model's *residual* error, so they do not transfer to
# another. The check used to read `fitted_model_version() not in (None, serving)`
# -- which an unstamped file passes, so it would be applied to every later model
# in silence. It is now two explicit cases, and the verdict is surfaced on
# `/health` rather than only logged, because a log line nobody reads is how the
# European results went four weeks stale unnoticed.
#
# These pin that a *clean* verdict requires the check to genuinely pass.


class _FullMeta(_Meta):
    """`health()` reads more of the metadata than the compatibility check does."""

    def __init__(self, version: str, feature_names: list[str]) -> None:
        super().__init__(version, feature_names)
        self.trained_through = "2026-10-18"
        self.n_train = 20_074


def _adjustment(monkeypatch, offsets: dict | None, stamped: str | None) -> None:
    monkeypatch.setattr(api, "load_league_adjustment", lambda *_a, **_k: offsets or {})
    monkeypatch.setattr(api, "fitted_model_version", lambda *_a, **_k: stamped)


def _serve(monkeypatch, version: str = "serving-v1") -> None:
    monkeypatch.setattr(
        api.registry, "load",
        lambda *_a, **_k: ("m", _FullMeta(version, ["home_points_last_5"])),
    )


OFFSETS = {"E0": 0.0, "SP1": -0.18}


def test_an_adjustment_fitted_on_the_serving_model_is_not_stale(serving, monkeypatch):
    _serve(monkeypatch)
    _adjustment(monkeypatch, OFFSETS, "serving-v1")

    api._load_state()

    assert api.state.league_offsets == OFFSETS
    assert api.state.league_offsets_stale is False
    assert api.health().league_adjustment_stale is False


def test_an_adjustment_fitted_on_another_model_is_stale(serving, monkeypatch):
    """Still applied -- a stale correction beats none -- but never silently."""
    _serve(monkeypatch)
    _adjustment(monkeypatch, OFFSETS, "some-older-model")

    api._load_state()

    assert api.state.league_offsets == OFFSETS, "the correction was dropped, not flagged"
    assert api.state.league_offsets_stale is True
    assert api.health().league_adjustment_stale is True


def test_an_unstamped_adjustment_is_stale_rather_than_trusted(serving, monkeypatch):
    """The original bug. `not in (None, serving)` let a file with no stamp
    through as though it had been checked."""
    _serve(monkeypatch)
    _adjustment(monkeypatch, OFFSETS, None)

    api._load_state()

    assert api.state.league_offsets_stale is True
    assert api.health().league_adjustment_stale is True


def test_no_adjustment_at_all_is_not_reported_as_stale(serving, monkeypatch):
    """Nothing is being applied, so there is nothing to be stale about -- and
    claiming staleness here would cry wolf on every fresh checkout."""
    _serve(monkeypatch)
    _adjustment(monkeypatch, {}, None)

    api._load_state()

    assert api.state.league_offsets_stale is False
    assert api.health().league_adjustment_stale is False


def test_a_reload_that_fixes_the_mismatch_clears_the_verdict(serving, monkeypatch):
    """The flag is state, so it has to be cleared before each check rather than
    only set -- otherwise the first bad verdict sticks for the process's life."""
    _serve(monkeypatch)
    _adjustment(monkeypatch, OFFSETS, "some-older-model")
    api._load_state()
    assert api.state.league_offsets_stale is True

    _adjustment(monkeypatch, OFFSETS, "serving-v1")
    api._load_state()

    assert api.state.league_offsets_stale is False
    assert api.health().league_adjustment_stale is False


def test_a_retrain_turns_a_clean_verdict_stale(serving, monkeypatch):
    """The direction that matters in practice: the adjustment does not move, the
    model does, and the verdict has to follow the model."""
    _serve(monkeypatch, "serving-v1")
    _adjustment(monkeypatch, OFFSETS, "serving-v1")
    api._load_state()
    assert api.state.league_offsets_stale is False

    _serve(monkeypatch, "serving-v2")
    api._load_state()

    assert api.state.league_offsets_stale is True
    assert api.health().league_adjustment_stale is True
