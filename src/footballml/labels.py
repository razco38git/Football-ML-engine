"""Human-readable names for features.

Feature names are built for code (``away_xg_diff_last_5_venue_diff``); this
turns them into something a reader can parse ("away team xG difference (last 5,
edge over opponent, at this venue)").

Lives in the package rather than a pipeline because both the CLI and the API
present explanations, and they must phrase them identically -- a driver labelled
one way in the terminal and another in the UI is the same bug twice.
"""

from __future__ import annotations

#: Metric stems rewritten for display.
METRIC_NAMES = {
    "xg_for": "xG created",
    "xg_against": "xG conceded",
    "xg_diff": "xG difference",
    "xg_overperformance": "finishing vs xG",
    "xg_per_shot": "chance quality",
    "npxg_for": "non-penalty xG created",
    "npxg_against": "non-penalty xG conceded",
    "npxg_diff": "non-penalty xG difference",
    "goals_for": "goals scored",
    "goals_against": "goals conceded",
    "goal_diff": "goal difference",
    "shots_for": "shots",
    "shots_against": "shots faced",
    "shots_on_target_for": "shots on target",
    "shots_on_target_against": "shots on target faced",
    "shot_accuracy": "shooting accuracy",
    "deep_completions": "passes near the box",
    "ppda": "pressing intensity",
    "points": "points",
    "wins": "wins",
    "draws": "draws",
    "losses": "losses",
    "matches_used": "matches of history",
    "days_since_last_match": "days of rest",
    "league_code": "league",
    # Squad strength, from player ratings. Always the previous season's, which
    # is worth knowing when reading an early-season explanation.
    "strength_overall": "squad rating",
    "strength_attack": "attack rating",
    "strength_defence": "defence rating",
}


def humanise(name: str) -> str:
    """Turn a feature name into something a reader can parse at a glance.

    Order matters. The trailing ``_diff`` marking a home-minus-away difference
    must be stripped before metric names are substituted, or a feature like
    ``goal_diff_last_5_diff`` has both its ``_diff`` parts rewritten and comes
    out as "goal edge, last 5 at venue edge".
    """
    qualifiers = []

    if name.endswith("_diff"):
        name = name[: -len("_diff")]
        qualifiers.append("edge over opponent")

    side = ""
    for prefix, label in (("home_", "home team"), ("away_", "away team")):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            side = label
            break

    if name.endswith("_venue"):
        name = name[: -len("_venue")]
        qualifiers.append("at this venue")

    window = ""
    if "_last_" in name:
        stem, _, size = name.rpartition("_last_")
        # Only a numeric suffix marks a rolling window. Without this check,
        # `days_since_last_match` is split into "days_since" + "last match".
        if size.isdigit():
            name, window = stem, f"last {size}"

    metric = METRIC_NAMES.get(name, name.replace("_", " "))
    if window:
        qualifiers.insert(0, window)

    text = f"{side} {metric}".strip()
    return f"{text} ({', '.join(qualifiers)})" if qualifiers else text
