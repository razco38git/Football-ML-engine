"""What the walk-forward run records about itself, and why that shape.

The accuracy page's headline came from a file last written eight days and three
retrains earlier, and nothing said so. The backtest now runs weekly and writes
a sidecar beside its predictions; these pin what goes in it and that the API
reads it back.

The subtlety worth keeping straight: **these metrics have never described the
model the site serves.** Each fold fits its own model on the seasons before the
one it scores -- `registry` is never read -- so they characterise the method on
the data it ran over. Stamping them with the serving model's version would
claim the one relationship that does not hold, which is why the staleness
verdict reads the input digests instead.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd

backtest = importlib.import_module("pipelines.backtest")
api = importlib.import_module("footballml.api.app")


def _preds() -> pd.DataFrame:
    return pd.DataFrame({
        "Season": [1516, 1516, 1617], "FTR": ["H", "D", "A"],
        "prob_H": [0.5, 0.3, 0.2], "prob_D": [0.3, 0.4, 0.3],
        "prob_A": [0.2, 0.3, 0.5],
    })


def test_the_backtest_never_loads_the_deployed_model() -> None:
    """The claim the rest of this rests on, asserted against the source rather
    than assumed: a fold that loaded `registry` would be scoring with a model
    that has seen the season it is being tested on."""
    source = backtest.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    body = text.split("def _provenance")[0] + text.split("return {")[-1]
    assert "registry.load()" not in body, (
        "the walk-forward loop loaded a trained artifact; every fold must fit "
        "its own model on the seasons before the one it scores"
    )


def test_provenance_records_when_and_over_what(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backtest, "PROCESSED_DIR", tmp_path)
    history = tmp_path / "team_match_history_all.csv"
    history.write_bytes(b"history")
    (tmp_path / "team_strength.csv").write_bytes(b"strength")

    meta = backtest._provenance(_preds(), history)

    assert meta["n_matches"] == 3
    assert meta["seasons"] == ["1516", "1617"]
    assert meta["evaluated_at"].startswith("20")
    assert meta["inputs"]["team_match_history_all.csv"] is not None
    assert meta["inputs"]["team_strength.csv"] is not None


def test_a_missing_input_is_recorded_as_absent_not_invented(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backtest, "PROCESSED_DIR", tmp_path)

    meta = backtest._provenance(_preds(), None)

    assert meta["inputs"]["team_match_history_all.csv"] is None
    assert meta["inputs"]["team_strength.csv"] is None


def test_the_model_version_is_provenance_not_a_claim(tmp_path, monkeypatch) -> None:
    """It is recorded so a reader can see which model happened to be current,
    and deliberately not what the API's staleness check reads."""
    monkeypatch.setattr(backtest, "PROCESSED_DIR", tmp_path)

    meta = backtest._provenance(_preds(), None)

    assert "model_version_at_run" in meta
    assert "model_version" not in meta, (
        "a bare `model_version` key reads as 'these metrics describe that model'"
    )


def test_what_the_backtest_writes_is_what_the_api_reads(tmp_path, monkeypatch) -> None:
    """The round trip. Either side alone can look right while the pair does
    not agree on a key name."""
    monkeypatch.setattr(backtest, "PROCESSED_DIR", tmp_path)
    history = tmp_path / "team_match_history_all.csv"
    history.write_bytes(b"history")
    (tmp_path / "team_strength.csv").write_bytes(b"strength")
    meta = backtest._provenance(_preds(), history)
    (tmp_path / "backtest_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(api, "PROCESSED_DIR", tmp_path)
    api.state.backtest = _preds()
    when, stale = api._backtest_freshness()

    assert when == meta["evaluated_at"]
    assert stale is False

    # Now move an input the way `build_players` would.
    (tmp_path / "team_strength.csv").write_bytes(b"rebuilt")
    _, stale = api._backtest_freshness()
    assert stale is True
