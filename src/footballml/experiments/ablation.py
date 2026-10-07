r"""Feature families, for the ablation ladder.

The question is how much each group of features actually contributes. That needs
a *partition* of the 235 model inputs into groups with a defensible boundary, and
the only boundary that is not an opinion is **where the data comes from**:

===========  ======================================================  ==========
family       provenance                                              columns
===========  ======================================================  ==========
``base``     the fixture list alone -- which league, how much rest         6
``elo``      the Elo walk over results                                     3
``form``     football-data.co.uk results and match stats, rolled         120
``xg``       Understat: xG, npxG, deep completions, PPDA                   94
``squad``    FBref + EA player ratings aggregated to a team                12
===========  ======================================================  ==========

235 in total, which is every production feature.

Grouping by source rather than by theme is what keeps this honest. "Attacking
features" would be a judgement call with no fact behind it, and every borderline
case would be arguable. Provenance has an answer: ``xg_overperformance`` mixes
goals with xG, so it *looks* like it could sit in either group, but it cannot be
computed without Understat and therefore belongs to ``xg``. ``shot_accuracy``
looks like a shooting-quality metric and comes from the football-data match
stats, so it is ``form``.

Two allocations worth stating because they are the only ones with any give:

``matches_used_last_{w}`` is window *metadata* -- how many matches the window
actually found -- not a measurement of a team. It is assigned to ``form``
because it is produced by the rolling machinery and is meaningless without it.

``league_code`` is in ``base`` because it is knowable before a ball is kicked
and is the only thing that lets the two goal models learn that Serie A and the
Bundesliga score at different rates. It is the reason ``base`` is not exactly
the base rate.

**This partition is exhaustive and is checked, not assumed.** :func:`classify`
raises on any column it cannot place, so a new feature cannot quietly land
outside the experiment -- it fails instead. That is the same discipline as the
truncation-invariance test: the guarantee is enforced rather than intended.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

#: Non-feature columns the backtest needs carried alongside any rung's features.
#: Identifiers and targets -- `feature_columns` excludes them from the model.
ID_COLUMNS: tuple[str, ...] = (
    "League", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
)

#: Family -> patterns matching the *stem* of a column (prefix and suffix
#: stripped by `_stem`). Order is irrelevant: the patterns are disjoint, which
#: `tests/test_ablation.py` checks by asserting every column matches exactly one.
FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "base": (
        r"^league_code$",
        r"^days_since_last_match$",
        r"^matches_last_\d+_days$",
    ),
    "elo": (
        r"^elo$",
    ),
    "form": (
        # Results: what the scoreline was.
        r"^(points|wins|draws|losses|goal_diff|goals_for|goals_against)_last_\d+(_venue)?$",
        # Match stats from the same source as the results.
        r"^(shots_for|shots_against|shots_on_target_for|shots_on_target_against"
        r"|shot_accuracy)_last_\d+(_venue)?$",
        # How full the window was.
        r"^matches_used_last_\d+(_venue)?$",
    ),
    "xg": (
        r"^(xg_for|xg_against|xg_diff|xg_overperformance|xg_per_shot)_last_\d+(_venue)?$",
        r"^(npxg_for|npxg_against|npxg_diff)_last_\d+(_venue)?$",
        r"^(deep_completions|ppda)_last_\d+(_venue)?$",
    ),
    "squad": (
        r"^strength_(overall|attack|defence|goalkeeper)$",
    ),
}

#: The nested ladder, each rung adding one family to the one before.
#:
#: ``full`` is deliberately *not* defined as the union of the families above. It
#: is the production column list verbatim, so that if the partition ever stops
#: being exhaustive the two disagree and the run fails -- rather than the
#: experiment silently measuring a subset of the real model.
LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("base", ("base",)),
    ("base+elo", ("base", "elo")),
    ("base+elo+form", ("base", "elo", "form")),
    ("base+elo+form+xg", ("base", "elo", "form", "xg")),
    ("base+elo+form+xg+squad", ("base", "elo", "form", "xg", "squad")),
    ("full", ()),  # sentinel: every production column
)

#: Families dropped one at a time from the full model, for the secondary
#: analysis. A nested ladder credits whichever family arrives first with
#: information a later one would also have carried, and `elo` and `form` are
#: both built from results, so they overlap heavily. Leave-one-out measures the
#: other end: what is lost that nothing else replaces.
LEAVE_ONE_OUT: tuple[str, ...] = ("elo", "form", "xg", "squad")

_PREFIX = re.compile(r"^(home|away)_")
_DIFF_SUFFIX = re.compile(r"_diff$")


def _stem(column: str) -> str:
    """Strip the ``home_``/``away_`` prefix and the ``_diff`` suffix.

    Every feature is one of ``home_X``, ``away_X`` or ``X_diff`` for some stem
    ``X``. Note that stems may themselves contain ``diff`` -- ``goal_diff_last_5``
    and ``npxg_diff_last_19`` -- which is why only a *trailing* ``_diff`` goes,
    and only one.
    """
    return _DIFF_SUFFIX.sub("", _PREFIX.sub("", column))


def family_of(column: str) -> str:
    """Which family a model input belongs to.

    Raises:
        KeyError: if no family claims it. Deliberately fatal: an unplaced
            feature means the ladder no longer covers the production model, and
            a silent default would make the experiment quietly wrong.
    """
    stem = _stem(column)
    for family, patterns in FAMILY_PATTERNS.items():
        if any(re.match(p, stem) for p in patterns):
            return family
    raise KeyError(
        f"feature {column!r} (stem {stem!r}) matches no family in FAMILY_PATTERNS. "
        "Add it to the right family in experiments/ablation.py -- the ablation "
        "ladder must cover every production feature or its results are a subset."
    )


def classify(columns: Iterable[str]) -> dict[str, list[str]]:
    """Group model inputs by family, preserving input order within each.

    Every family appears in the result even if empty, so callers can rely on the
    keys.
    """
    grouped: dict[str, list[str]] = {f: [] for f in FAMILY_PATTERNS}
    for column in columns:
        grouped[family_of(column)].append(column)
    return grouped


def rung_columns(
    all_columns: Sequence[str], families: Sequence[str]
) -> list[str]:
    """Production columns belonging to ``families``, in production order.

    An empty ``families`` means the ``full`` rung: every column, unfiltered.
    Order is preserved from ``all_columns`` so two rungs sharing a family hand
    the model identically-ordered inputs.
    """
    if not families:
        return list(all_columns)
    wanted = set(families)
    unknown = wanted - set(FAMILY_PATTERNS)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}")
    return [c for c in all_columns if family_of(c) in wanted]
