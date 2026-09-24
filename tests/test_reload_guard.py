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
