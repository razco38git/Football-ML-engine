"""Canonical entity resolution across data sources."""

from footballml.entities.teams import (
    UnresolvedTeamsError,
    load_aliases,
    resolve_teams,
)

__all__ = ["UnresolvedTeamsError", "load_aliases", "resolve_teams"]
