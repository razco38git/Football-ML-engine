"""Match prediction models."""

from footballml.models.dixon_coles import fit_rho, outcome_probs, score_matrix
from footballml.models.match_model import MatchPredictor

__all__ = ["MatchPredictor", "fit_rho", "outcome_probs", "score_matrix"]
