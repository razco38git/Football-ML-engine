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


class TeamForm(BaseModel):
    """A team's last five matches going into a fixture, oldest first.

    The same window the model aggregates over, exposed match by match so the UI
    can show what the prediction was actually based on.
    """

    team: str
    results: list[str] = Field(default_factory=list, description="W/D/L, oldest first")
    xg_for: list[float] = Field(default_factory=list)
    xg_against: list[float] = Field(default_factory=list)
    shots: list[float] = Field(default_factory=list)
    shots_on_target: list[float] = Field(default_factory=list)
    goals_for: list[float] = Field(default_factory=list)
    goals_against: list[float] = Field(default_factory=list)
    opponents: list[str] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)


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

    form_home: TeamForm | None = None
    form_away: TeamForm | None = None


class PlayerRating(BaseModel):
    """One player's rating for a season.

    Attributes are the ones we actually measure. There is no pace, dribbling or
    physical score here, because nothing in the source data supports inventing
    them -- the sub-ratings present are derived from real per-90 output.
    """

    player: str
    team: str
    league: str
    season: str
    position: str = Field(description="GK, CB, FB, MID, AMW or FWD")
    minutes: int
    rating: int | None = None
    rated: bool
    unrated_reason: str | None = None

    #: Populated per position group; a forward has no defending sub-rating.
    sub_finishing: int | None = None
    sub_creation: int | None = None
    sub_involvement: int | None = None
    sub_volume: int | None = None
    sub_defending: int | None = None
    sub_shot_stopping: int | None = None
    sub_reliability: int | None = None
    sub_workload: int | None = None
    sub_penalties: int | None = None

    #: Underlying per-90 output, so a rating can be checked against the numbers.
    goals: int | None = None
    assists: int | None = None
    np_xg: float | None = None
    xa: float | None = None
    key_passes_per90: float | None = None
    interceptions_per90: float | None = None
    tackles_won_per90: float | None = None
    save_pct: float | None = None
    goals_against_per90: float | None = None


class PlayerPage(BaseModel):
    """A page of players plus the total available for the same filters."""

    total: int
    players: list[PlayerRating]


class TeamStrength(BaseModel):
    """A team's rating, built from its players.

    ``method`` records how the squad was reduced: ``eleven`` is the predicted
    starting eleven (highest-minute player per formation slot), ``best_n`` the
    top-rated N regardless of minutes, ``squad`` everyone weighted by minutes.
    """

    team: str
    league: str
    season: str
    method: str
    n_players: int
    strength_overall: float
    strength_goalkeeper: float | None = None
    strength_defence: float | None = None
    strength_midfield: float | None = None
    strength_attack: float | None = None


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
