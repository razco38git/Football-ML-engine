"""Source adapters. Each returns data already reconciled to canonical names."""

from footballml.ingest.understat import fetch_understat_team_match, merge_xg

__all__ = ["fetch_understat_team_match", "merge_xg"]
