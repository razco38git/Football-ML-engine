"""Response models for the API.

These double as the contract the frontend generates its TypeScript types from,
so field names here become field names there. Changing one is a breaking change
for the other.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    model_version: str
    trained_through: date
    n_train: int


class ModelInfo(BaseModel):
    version: str
    trained_at: str
    trained_through: date
    n_train: int
    n_features: int
    leagues: list[str]
    rho: float = Field(description="Fitted Dixon-Coles low-score correlation")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Held-out metrics from a season the model never saw"
    )


class Driver(BaseModel):
    """One feature's contribution to a side's expected goals."""

    feature: str
    label: str = Field(description="Human-readable feature name")
    contribution: float


class Prediction(BaseModel):
    league: str
    date: date
    home_team: str
    away_team: str

    expected_goals_home: float
    expected_goals_away: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    predicted_outcome: str
    modal_score_home: int
    modal_score_away: int
    prob_over_2_5: float
    prob_btts: float

    market_prob_home: float | None = None
    market_prob_draw: float | None = None
    market_prob_away: float | None = None

    actual_home_goals: int | None = None
    actual_away_goals: int | None = None
    actual_result: str | None = None

    drivers_home: list[Driver] = Field(default_factory=list)
    drivers_away: list[Driver] = Field(default_factory=list)


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    league: str | None = Field(
        default=None, description="Inferred from the teams when omitted"
    )
    explain: bool = False


class CalibrationBin(BaseModel):
    bin_lower: float
    bin_upper: float
    n: int
    mean_predicted: float
    observed_rate: float


class Accuracy(BaseModel):
    """Published track record over settled predictions."""

    n: int
    accuracy: float
    rps: float
    log_loss: float
    brier: float
    rps_base_rate: float
    rps_market: float | None = None
    by_league: dict[str, float] = Field(
        default_factory=dict, description="RPS per league"
    )
    calibration: list[CalibrationBin] = Field(default_factory=list)
