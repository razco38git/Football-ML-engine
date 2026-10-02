"""Who plays like this player?

Two questions wearing one name, and they need answering separately.

**EA's attributes** -- pace, shooting, passing, dribbling, defending, physical --
are scored on one scale for every outfield player. A full-back's 82 pace and a
winger's 82 pace mean the same thing, so this axis can compare across positions.
That makes "this full-back has a winger's profile" a finding rather than an
artefact. Goalkeepers are scored on five entirely different attributes (diving,
handling, kicking, positioning, reflexes) and are never comparable to outfielders.

**Our own sub-ratings** are percentiles *within a role*, and the vector itself
differs by role: forwards have no ``defending``, only AMW/MID/FWD have
``volume``, keepers have four of their own. A centre-back's 80 for defending and
a winger's 80 are ranks in different populations -- the same trap that made
Nathan Collins look worse than a winger who tracks back. So percentile
similarity is **same-role only**, and returns nothing across roles rather than a
number that would be quietly meaningless.

Distance is Euclidean over the role's attribute set. Not cosine: absolute level
matters here. A 90-rated winger and a 60-rated winger with identical shape are
not similar players, and cosine would call them identical.

**The scale is calibrated against the pool, not against the theoretical range.**
Dividing by the worst possible distance (99 on all six attributes) was the first
attempt and it made the number useless: measured on 2025/26, two *randomly
chosen* players of the same position scored 89-90% similar, and everything
meaningful was squeezed into 80-96. Ranking was then decided by noise -- Salah's
nearest winger came out as Kenan Yildiz, who is 2.3x further away in attribute
space than Raphinha.

So 50 is pinned to the median distance between two players of that position:
``100 * 0.5 ** (distance / median)``. A score of 50 means "no more alike than
two random players in this position", 100 means identical, and the numbers in
between discriminate. The reference is computed from the candidate pool itself,
so it is symmetric between any two players being compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: EA's six outfield attributes, shared scale across every outfield position.
OUTFIELD_ATTRIBUTES: tuple[str, ...] = (
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physical",
)

#: EA's five goalkeeper attributes. A separate scale; never mixed with the six.
GK_ATTRIBUTES: tuple[str, ...] = (
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes",
)

#: Every sub-rating we produce. Which ones a player has depends on the role --
#: see the module docstring. The set used for a comparison is the intersection
#: of what both players actually carry.
SUB_RATINGS: tuple[str, ...] = (
    "sub_finishing",
    "sub_creation",
    "sub_involvement",
    "sub_volume",
    "sub_defending",
    "sub_shot_stopping",
    "sub_reliability",
    "sub_workload",
    "sub_penalties",
)

GK_ROLE = "GK"

#: How many candidates the reference distance is estimated from. Every pair of a
#: 300-row sample is 45,000 distances, plenty for a median, and taking a
#: deterministic stride keeps the scale stable between calls.
_REFERENCE_SAMPLE = 300

#: Below this, two players share so little measured ground that the number would
#: be noise. Three of six attributes is already a thin comparison.
MIN_SHARED_ATTRIBUTES = 3


def attribute_set(role: str) -> tuple[str, ...]:
    """The EA attributes that mean something for this role."""
    return GK_ATTRIBUTES if role == GK_ROLE else OUTFIELD_ATTRIBUTES


@dataclass(frozen=True)
class Match:
    """One candidate, with the numbers that produced its score."""

    player: str
    team: str
    league: str
    season: str
    role: str
    rating: int | None
    minutes: int
    #: 0-100, or None when either player has no EA entry. A small share have
    #: none, and a silent zero would read as "completely different".
    fifa_similarity: float | None
    #: 0-100, or None across roles -- the percentiles are not comparable there.
    percentile_similarity: float | None
    #: Whichever of the two are available, averaged. Never invented.
    combined: float | None
    attributes: dict[str, float] = field(default_factory=dict)
    sub_ratings: dict[str, float] = field(default_factory=dict)


def _reference_distance(matrix: np.ndarray) -> float:
    """Median distance between two players in this pool -- the anchor for 50.

    Sampled by stride rather than at random so the same pool always yields the
    same scale: a similarity that drifted between calls would be indistinguishable
    from a player actually changing.
    """
    if len(matrix) < 2:
        return 0.0
    step = max(1, len(matrix) // _REFERENCE_SAMPLE)
    sample = matrix[::step][:_REFERENCE_SAMPLE]
    gaps = sample[:, None, :] - sample[None, :, :]
    distances = np.linalg.norm(gaps, axis=-1)
    upper = distances[np.triu_indices(len(sample), k=1)]
    return float(np.median(upper)) if upper.size else 0.0


def _similarity(left: np.ndarray, right: np.ndarray, reference: float) -> float:
    """Euclidean distance expressed as 0-100 against the pool's typical gap.

    Halves every time the distance grows by one reference length, so identical
    players score 100 and a typical pair scores 50. See the module docstring for
    why the theoretical range is the wrong denominator.
    """
    distance = float(np.linalg.norm(left - right))
    if reference <= 0:
        return 100.0 if distance == 0 else 0.0
    return float(round(100.0 * 0.5 ** (distance / reference), 1))


def _vector(row: pd.Series, columns: list[str]) -> np.ndarray | None:
    values = row[columns].astype(float)
    return None if values.isna().any() else values.to_numpy()


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Complete rows only -- a partial vector would distort the reference."""
    if not columns or frame.empty:
        return np.empty((0, 0))
    return frame[columns].dropna().to_numpy(dtype=float)


def _shared(frame: pd.DataFrame, row: pd.Series, columns: list[str]) -> list[str]:
    """Columns present and populated for the subject and somebody else."""
    return [c for c in columns if c in frame.columns and not pd.isna(row.get(c))]


def find_player(rated: pd.DataFrame, name: str, season: str | None = None) -> pd.Series:
    """The one row a similarity search is anchored on.

    Raises:
        LookupError: If the name matches nobody, so callers can turn that into
            a 404 rather than guessing at a near-miss.
    """
    rows = rated[rated["Player"].str.lower() == name.lower()]
    if rows.empty:
        raise LookupError(f"Unknown player {name!r}")
    if season is not None:
        rows = rows[rows["Season"].astype(str) == str(season)]
        if rows.empty:
            raise LookupError(f"No rated season {season} for {name!r}")
    return rows.sort_values("Season").iloc[-1]


def similar_players(
    rated: pd.DataFrame,
    name: str,
    season: str | None = None,
    limit: int = 10,
    same_role: bool = True,
) -> tuple[pd.Series, list[Match]]:
    """Players most like ``name``, and the subject row they were measured against.

    Args:
        rated: Player-season rows, already filtered to ``rated``.
        name: Player to anchor on, matched case-insensitively.
        season: Restrict to this season; defaults to the player's latest.
        limit: How many candidates to return.
        same_role: Compare only within the player's role. Widening it drops the
            percentile axis, because those numbers are ranked within a role.

    Keepers are never compared with outfielders even when ``same_role`` is off:
    the two attribute sets measure different things.
    """
    subject = find_player(rated, name, season)
    subject_season = str(subject["Season"])
    role = str(subject["role"])

    population = rated[rated["Season"].astype(str) == subject_season]
    if same_role or role == GK_ROLE:
        population = population[population["role"] == role]
    else:
        # Keepers stay out of an outfielder's pool regardless: their attributes
        # are a different five, so no comparison exists.
        population = population[population["role"] != GK_ROLE]

    # The scale is calibrated on the whole population, the subject included, so
    # it does not depend on who is being asked about. Calibrating on the pool
    # minus the subject would give `similar(a, b)` and `similar(b, a)` slightly
    # different references, and similarity has to be symmetric.
    pool = population[population["Player"].str.lower() != name.lower()]
    if pool.empty:
        return subject, []

    fifa_columns = _shared(pool, subject, [f"fifa_{a}" for a in attribute_set(role)])
    sub_columns = _shared(pool, subject, list(SUB_RATINGS))

    # FC27 carries only three of the five keeper attributes, and a comparison
    # that thin says less than it appears to. Drop the axis rather than dress it up.
    if len(fifa_columns) < MIN_SHARED_ATTRIBUTES:
        fifa_columns = []
    if len(sub_columns) < MIN_SHARED_ATTRIBUTES:
        sub_columns = []

    subject_fifa = _vector(subject, fifa_columns) if fifa_columns else None
    subject_subs = _vector(subject, sub_columns) if sub_columns else None

    # Widening to other positions drops the percentile axis entirely, not just
    # for the players it cannot cover. Scoring the handful of same-role
    # candidates on two axes and everyone else on one would put two different
    # scales in a single ranked list.
    if not same_role:
        sub_columns = []
        subject_subs = None

    # One reference per axis, from the pool both players sit in, so the scale
    # means the same thing whichever of the two is the subject.
    fifa_reference = _reference_distance(_matrix(population, fifa_columns))
    sub_reference = _reference_distance(_matrix(population, sub_columns))

    matches: list[Match] = []
    for _, row in pool.iterrows():
        fifa_score = None
        if subject_fifa is not None:
            candidate = _vector(row, fifa_columns)
            if candidate is not None:
                fifa_score = _similarity(subject_fifa, candidate, fifa_reference)

        # Percentiles only within a role, and only where both carry the same
        # sub-ratings -- across roles the vectors differ in both scale and shape.
        sub_score = None
        if subject_subs is not None and str(row["role"]) == role:
            candidate = _vector(row, sub_columns)
            if candidate is not None:
                sub_score = _similarity(subject_subs, candidate, sub_reference)

        available = [s for s in (fifa_score, sub_score) if s is not None]
        if not available:
            continue

        matches.append(
            Match(
                player=str(row["Player"]),
                team=str(row["Team"]),
                league=str(row["League"]),
                season=str(row["Season"]),
                role=str(row["role"]),
                rating=None if pd.isna(row.get("rating")) else int(row["rating"]),
                minutes=int(row.get("minutes", 0) or 0),
                fifa_similarity=fifa_score,
                percentile_similarity=sub_score,
                combined=round(sum(available) / len(available), 1),
                attributes=values_for(row, fifa_columns),
                sub_ratings=values_for(row, sub_columns),
            )
        )

    matches.sort(key=lambda m: (m.combined or 0.0), reverse=True)
    return subject, matches[:limit]


def values_for(row: pd.Series, columns: list[str]) -> dict[str, float]:
    """The raw numbers behind a score, so the UI can show *why*."""
    out = {}
    for column in columns:
        value = row.get(column)
        if not pd.isna(value):
            out[column.removeprefix("fifa_")] = round(float(value), 1)
    return out
