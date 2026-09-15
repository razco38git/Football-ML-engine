"""Recent-form detail for display.

The model consumes *aggregated* form -- averages over the last five matches.
The UI needs the underlying matches themselves: five W/D/L badges, five xG bars,
five shot counts. Same window, different shape.

Kept separate from :mod:`footballml.features` deliberately. Those functions feed
a model and must never see the future; these feed a screen and are read by eye.
Mixing them would make it easy to accidentally present something the model was
not allowed to use, or worse, to relax a leakage guarantee for a cosmetic reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TeamForm:
    """One team's recent record going into a fixture."""

    team: str
    results: list[str] = field(default_factory=list)
    xg_for: list[float] = field(default_factory=list)
    xg_against: list[float] = field(default_factory=list)
    shots: list[float] = field(default_factory=list)
    shots_on_target: list[float] = field(default_factory=list)
    goals_for: list[float] = field(default_factory=list)
    goals_against: list[float] = field(default_factory=list)
    opponents: list[str] = field(default_factory=list)
    venues: list[str] = field(default_factory=list)


def build_index(tmh: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre-split the history by team, sorted by date.

    Done once at startup. Filtering a 58,000-row frame per team per fixture is
    the difference between a snappy endpoint and a sluggish one.
    """
    ordered = tmh.sort_values("Date")
    return {str(team): group for team, group in ordered.groupby("Team", sort=False)}


def recent_form(
    index: dict[str, pd.DataFrame],
    team: str,
    before: pd.Timestamp,
    n: int = 5,
    venue: str | None = None,
) -> TeamForm:
    """The team's last ``n`` matches before ``before``, oldest first.

    Args:
        index: Output of :func:`build_index`.
        team: Canonical team name.
        before: Exclusive cutoff -- the fixture's own date.
        n: How many matches to return.
        venue: ``"Home"`` or ``"Away"`` to restrict to one venue.

    Returns:
        A :class:`TeamForm`, empty when the team has no prior matches.
    """
    history = index.get(team)
    if history is None or history.empty:
        return TeamForm(team=team)

    window = history[history["Date"] < before]
    if venue:
        window = window[window["Venue"] == venue]
    window = window.tail(n)
    if window.empty:
        return TeamForm(team=team)

    def numbers(column: str) -> list[float]:
        if column not in window.columns:
            return []
        values = pd.to_numeric(window[column], errors="coerce")
        # NaN is not valid JSON; the UI treats a missing entry as "unknown".
        return [round(float(v), 2) for v in values if pd.notna(v)]

    return TeamForm(
        team=team,
        results=[str(r) for r in window["Result"] if pd.notna(r)],
        xg_for=numbers("xGFor"),
        xg_against=numbers("xGAgainst"),
        shots=numbers("ShotsFor"),
        shots_on_target=numbers("ShotsOnTargetFor"),
        goals_for=numbers("GoalsFor"),
        goals_against=numbers("GoalsAgainst"),
        opponents=[str(o) for o in window["Opponent"]],
        venues=[str(v) for v in window["Venue"]],
    )
