"""Team name reconciliation across sources.

Every football data source spells clubs differently. football-data.co.uk says
``Man United``, Understat says ``Manchester United``, FBref says
``Manchester Utd``. Joining on raw names silently drops rows -- and a silent drop
is far worse than a crash, because the pipeline keeps running and the model
quietly trains on less data than you believe it has.

So resolution here is **strict by default**: an unrecognised name raises rather
than passing through. Adding a source means adding its aliases to
``config/team_aliases.yaml``, which is the single place naming is reconciled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from footballml.data import PROJECT_ROOT

ALIAS_CONFIG = PROJECT_ROOT / "config" / "team_aliases.yaml"


class UnresolvedTeamsError(ValueError):
    """Raised when a source name has no canonical mapping."""


def load_aliases(source: str, path: Path | None = None) -> dict[str, str]:
    """Load the alias map for one source, e.g. ``"understat"``."""
    config = yaml.safe_load((path or ALIAS_CONFIG).read_text(encoding="utf-8")) or {}
    return config.get(source) or {}


def resolve_teams(
    df: pd.DataFrame,
    columns: str | list[str],
    source: str,
    known: set[str] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Map team-name columns onto their canonical form.

    Args:
        df: Frame containing team-name columns.
        columns: Column name or names to resolve.
        source: Key in the alias config, e.g. ``"understat"``.
        known: Canonical names to validate against. When given, any name that is
            neither a known canonical name nor a configured alias raises.
        path: Override the alias config location (used by tests).

    Raises:
        UnresolvedTeamsError: If ``known`` is supplied and names cannot be mapped.
    """
    aliases = load_aliases(source, path)
    cols = [columns] if isinstance(columns, str) else list(columns)

    out = df.copy()
    for col in cols:
        out[col] = out[col].replace(aliases)

    if known is not None:
        seen = set()
        for col in cols:
            seen |= set(out[col].dropna().unique())
        unresolved = seen - known
        if unresolved:
            raise UnresolvedTeamsError(
                f"{len(unresolved)} {source} team name(s) have no canonical mapping: "
                f"{sorted(unresolved)}. Add them to {ALIAS_CONFIG.name}."
            )

    return out
