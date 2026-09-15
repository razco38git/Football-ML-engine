"""Model persistence.

Training on 29,000 matches takes about half a minute, which is far too slow to
do inside a request. Models are therefore trained by a batch job, written to
disk with the metadata needed to interpret them, and loaded once at API startup.

Every artifact records **what data it saw**. That matters more than it sounds:
a prediction is only meaningful alongside the training cutoff that produced it,
and the accuracy tracker needs to attribute each stored prediction to the exact
model version that made it. Without that, "how accurate are we?" has no
well-defined answer once the model changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from footballml.data import PROJECT_ROOT
from footballml.models.match_model import MatchPredictor

MODEL_DIR = PROJECT_ROOT / "models"
LATEST = "latest"


@dataclass
class ModelMetadata:
    """Everything needed to interpret a stored model."""

    version: str
    trained_at: str
    trained_through: str
    n_train: int
    leagues: list[str]
    feature_names: list[str]
    rho: float
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def make_version(trained_through: str) -> str:
    """Version string combining the data cutoff and the build time.

    The cutoff alone is not enough: retraining the same window after a feature
    change must produce a distinguishable version, or stored predictions become
    ambiguous.
    """
    # Microsecond precision: two retrains within the same second must not
    # collide, or stored predictions cannot be attributed to the right model.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{trained_through.replace('-', '')}-{stamp}"


def save(
    model: MatchPredictor,
    metadata: ModelMetadata,
    directory: Path | None = None,
    mark_latest: bool = True,
) -> Path:
    """Persist a model and its metadata, optionally updating the ``latest`` alias."""
    directory = directory or MODEL_DIR
    target = directory / metadata.version
    target.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, target / "model.joblib")
    (target / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2), encoding="utf-8"
    )

    if mark_latest:
        # A pointer file rather than a symlink: symlinks need elevated
        # permissions on Windows and break in most CI containers.
        (directory / f"{LATEST}.txt").write_text(metadata.version, encoding="utf-8")
    return target


def load(
    version: str = LATEST, directory: Path | None = None
) -> tuple[MatchPredictor, ModelMetadata]:
    """Load a model by version, or the most recently trained one."""
    directory = directory or MODEL_DIR

    if version == LATEST:
        pointer = directory / f"{LATEST}.txt"
        if not pointer.exists():
            raise FileNotFoundError(
                f"No trained model in {directory}. Run `python -m pipelines.train`."
            )
        version = pointer.read_text(encoding="utf-8").strip()

    target = directory / version
    if not target.exists():
        raise FileNotFoundError(f"Model version {version!r} not found in {directory}")

    model: MatchPredictor = joblib.load(target / "model.joblib")
    raw: dict[str, Any] = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    return model, ModelMetadata(**raw)


def list_versions(directory: Path | None = None) -> list[ModelMetadata]:
    """All stored model versions, newest first."""
    directory = directory or MODEL_DIR
    if not directory.exists():
        return []

    found = []
    for path in directory.iterdir():
        meta = path / "metadata.json"
        if meta.exists():
            found.append(ModelMetadata(**json.loads(meta.read_text(encoding="utf-8"))))
    return sorted(found, key=lambda m: m.trained_at, reverse=True)
