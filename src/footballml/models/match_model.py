"""The match predictor: two Poisson goal-rate models plus a Dixon-Coles matrix.

Rather than classifying home/draw/away directly, we predict *how many goals each
side is expected to score* and derive everything else from the resulting
scoreline distribution. Three reasons this is the better shape:

1. Expected goals are an output the description explicitly asks for, and here
   they are the model's native prediction rather than a bolt-on.
2. 1X2 probabilities, scorelines, over/under and BTTS all come from one object,
   so they can never contradict each other on the site.
3. Draws are notoriously hard to classify directly. Deriving them from a goal
   distribution handles them naturally.

Gradient boosting with a Poisson objective is the right estimator for a count
target: it models the conditional *rate*, keeps predictions positive, and its
loss matches the data-generating process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from footballml.models.dixon_coles import (
    DEFAULT_MAX_GOALS,
    both_teams_score,
    fit_rho,
    most_likely_score,
    outcome_probs,
    over_under,
    score_matrix,
    top_scores,
)

#: Deliberately *not* using early stopping: its validation split is random,
#: which on time-ordered football data means training against future matches.
#: The walk-forward backtest is the honest way to choose these, and it is what
#: chose them.
#:
#: **MEASURED (2026-10-05).** These had never been swept -- the note here used
#: to call them "conservative defaults for a few thousand training rows", and
#: the model trains on 29,143. The expectation was that they were too small.
#: They were too *large*: every capacity axis preferred less, and the model was
#: mildly over-trained.
#:
#: What matters is the product of `learning_rate` and `max_iter` -- one quantity
#: between them, the total learning budget. Across a 24-point grid scored by a
#: four-fold walk-forward:
#:
#:     budget   3.0    4.5    6.0    9.0   12.0   15.0   18.0   27.0   45.0
#:     RPS    .19825 .19801 .19798 .19802 .19817 .19827 .19845 .19889 .19961
#:
#: A flat bottom at 4.5-9 rising monotonically after it, with the old 0.05x300
#: sitting at 15, on the rising side. 0.02x300 is budget 6, the minimum of that
#: curve and interior to the grid on every axis -- 0.03x150 scored a hair
#: better pooled but is at the edge of the iteration axis, which is where a
#: selection artefact would hide.
#:
#: Confirmed on the full 20,013-match walk-forward, paired bootstrap against
#: the previous settings: **RPS 0.2003 -> 0.1991**, -0.00124 [-0.00158,
#: -0.00090], and better on 2025/26 alone as well. Log loss 0.9847 -> 0.9807,
#: Brier 0.5867 -> 0.5838, accuracy 52.69% -> 52.99%. All four moved together,
#: which the three feature experiments that failed the same week did not.
#:
#: `max_leaf_nodes`, `min_samples_leaf` and `l2_regularization` were swept too
#: and are left alone: each is at or within noise of its own optimum.
DEFAULT_GBM_PARAMS: dict[str, Any] = {
    "loss": "poisson",
    "learning_rate": 0.02,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 7,
}

logger = logging.getLogger(__name__)

OUTCOMES = ("H", "D", "A")

#: How many scorelines `predict_frame` reports.
#:
#: Three is enough to show the shape without turning the card into a table: the
#: leader typically clears the runner-up by about a percentage point, which is
#: the fact a single number hides.
TOP_SCORES = 3


@dataclass
class MatchPredictor:
    """Predicts goal rates and the full scoreline distribution for a fixture."""

    max_goals: int = DEFAULT_MAX_GOALS
    gbm_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_GBM_PARAMS))
    use_dixon_coles: bool = True

    home_model_: HistGradientBoostingRegressor | None = field(default=None, init=False)
    away_model_: HistGradientBoostingRegressor | None = field(default=None, init=False)
    rho_: float = field(default=0.0, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def fit(
        self,
        X: pd.DataFrame,
        home_goals: np.ndarray | pd.Series,
        away_goals: np.ndarray | pd.Series,
    ) -> MatchPredictor:
        """Fit both goal-rate models and the Dixon-Coles correlation.

        ``X`` may contain NaNs -- histogram gradient boosting handles missing
        values natively by learning a default split direction. That matters here
        because early-season fixtures genuinely have no prior form, and imputing
        zeros would tell the model something false.
        """
        # Drop features with no values anywhere in training. They carry no
        # information, and sklearn's histogram binner raises "window shape
        # cannot be larger than input array shape" on them rather than ignoring
        # them -- which is how a feature that only exists in later seasons (team
        # strength) crashes a backtest that trains on earlier ones.
        usable = [c for c in X.columns if X[c].notna().any()]
        dropped = [c for c in X.columns if c not in usable]
        if dropped:
            logger.info(
                "Ignoring %d feature(s) with no values in training: %s",
                len(dropped), dropped[:6],
            )

        self.feature_names_ = usable
        X = X[usable]
        hg = np.asarray(home_goals, dtype="float64")
        ag = np.asarray(away_goals, dtype="float64")

        self.home_model_ = HistGradientBoostingRegressor(**self.gbm_params).fit(X, hg)
        self.away_model_ = HistGradientBoostingRegressor(**self.gbm_params).fit(X, ag)

        if self.use_dixon_coles:
            # Fitted in-sample. It is a single scalar estimated from thousands of
            # scorelines, so the overfitting risk is negligible.
            mu_h, mu_a = self.predict_goal_rates(X)
            self.rho_ = fit_rho(mu_h, mu_a, hg, ag, max_goals=self.max_goals)

        return self

    def predict_goal_rates(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals for each side. This is the site's headline number."""
        self._check_fitted()
        X = X[self.feature_names_]
        # Poisson-loss HGB returns the rate directly, but clip anyway: a rate of
        # zero would make the scoreline matrix degenerate.
        mu_h = np.clip(self.home_model_.predict(X), 1e-6, None)
        mu_a = np.clip(self.away_model_.predict(X), 1e-6, None)
        return mu_h, mu_a

    def predict_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Full ``(n, G+1, G+1)`` scoreline probability matrix."""
        mu_h, mu_a = self.predict_goal_rates(X)
        return score_matrix(mu_h, mu_a, rho=self.rho_, max_goals=self.max_goals)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """1X2 probabilities as ``(n, 3)`` columns ``[H, D, A]``."""
        return outcome_probs(self.predict_matrix(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely outcome label per fixture."""
        return np.array(OUTCOMES, dtype=object)[self.predict_proba(X).argmax(axis=1)]

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Everything the site needs for one fixture, in one table.

        Columns: expected goals per side, 1X2 probabilities, the modal scoreline,
        over/under 2.5 and both-teams-to-score.
        """
        return self.frame_from_rates(*self.predict_goal_rates(X), index=X.index)

    def frame_from_rates(
        self, mu_h: np.ndarray, mu_a: np.ndarray, index: pd.Index | None = None
    ) -> pd.DataFrame:
        """The same table, from goal rates that a caller may have adjusted.

        Split out for the one correction the model cannot make for itself. Every
        match it trains on is domestic, so the gap between two *leagues* is
        always zero in training and the booster never learns what a cross-league
        rating difference means -- measured on 822 UEFA ties, its predicted
        home-win probability moved at +0.0034 per rating point where the truth
        moves at +0.0402. :mod:`footballml.league_adjust` corrects the rates
        outside the model, and needs everything downstream of them rebuilt.
        """
        matrix = score_matrix(mu_h, mu_a, rho=self.rho_, max_goals=self.max_goals)
        probs = outcome_probs(matrix)
        modal = most_likely_score(matrix)
        # Three, not one: see `top_scores` for why a single score misrepresents
        # this distribution. `modal` stays as it is -- the stored prediction
        # record and score_upcoming both write it, and rewriting what a past
        # prediction said is not on the table.
        scores, score_probs = top_scores(matrix, n=TOP_SCORES)

        return pd.DataFrame(
            {
                "expected_goals_home": mu_h,
                "expected_goals_away": mu_a,
                "prob_home_win": probs[:, 0],
                "prob_draw": probs[:, 1],
                "prob_away_win": probs[:, 2],
                "predicted_outcome": np.array(OUTCOMES, dtype=object)[probs.argmax(axis=1)],
                "modal_score_home": modal[:, 0],
                "modal_score_away": modal[:, 1],
                # The modal scoreline's own probability. Without it a 10% score
                # is read as the forecast and looks like it contradicts a 50%
                # win probability -- Roma 1-1 beside "Roma Win" -- when the two
                # answer different questions.
                "prob_modal_score": matrix[
                    np.arange(len(modal)), modal[:, 0], modal[:, 1]
                ],
                **{
                    f"score_{k}_{part}": values
                    for k in range(TOP_SCORES)
                    for part, values in (
                        ("home", scores[:, k, 0]),
                        ("away", scores[:, k, 1]),
                        ("prob", score_probs[:, k]),
                    )
                },
                "prob_over_2_5": over_under(matrix, 2.5),
                "prob_btts": both_teams_score(matrix),
            },
            index=pd.RangeIndex(len(mu_h)) if index is None else index,
        )

    def explain(self, X: pd.DataFrame, top_n: int = 5) -> list[dict[str, list[tuple[str, float]]]]:
        """Per-fixture feature contributions to each side's expected goals.

        Uses SHAP, which apportions the gap between a prediction and the dataset
        average across the features that caused it. A contribution of ``+0.18``
        on ``xg_diff_last_5_diff`` means that feature pushed the expected goal
        count up by 0.18 relative to an average fixture.

        This is what lets the site answer "why did you predict that?" with the
        model's actual reasoning rather than a plausible-sounding story.

        Args:
            X: Fixtures to explain.
            top_n: Contributions to keep per side, ranked by absolute magnitude.

        Returns:
            One dict per fixture with ``home`` and ``away`` lists of
            ``(feature, contribution)`` pairs, largest effect first.
        """
        import shap  # imported lazily: heavy, and only needed for explanations

        self._check_fitted()
        X = X[self.feature_names_]

        contributions = {
            side: shap.TreeExplainer(model).shap_values(X)
            for side, model in (("home", self.home_model_), ("away", self.away_model_))
        }

        out: list[dict[str, list[tuple[str, float]]]] = []
        for row in range(len(X)):
            entry: dict[str, list[tuple[str, float]]] = {}
            for side, values in contributions.items():
                pairs = list(zip(self.feature_names_, values[row], strict=True))
                pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
                entry[side] = [(name, float(v)) for name, v in pairs[:top_n]]
            out.append(entry)
        return out

    def _check_fitted(self) -> None:
        if self.home_model_ is None or self.away_model_ is None:
            raise RuntimeError("MatchPredictor is not fitted; call fit() first")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: every numeric column that is not an identifier or target.

    Note what is deliberately absent: bookmaker odds. They are the strongest
    single predictor available and using them would inflate every metric, but a
    model that predicts the market by reading the market has learned nothing.
    Odds are kept strictly as an evaluation benchmark.
    """
    excluded = {
        "League", "Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
    }
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
