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


class LikelyScore(BaseModel):
    """One candidate scoreline and how likely the model thinks it is."""

    home: int
    away: int
    probability: float


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
    #: Probability of that exact scoreline. Typically ~10%, against a 1X2
    #: probability that sums dozens of scorelines -- the page must not present
    #: the two as if they were the same claim.
    prob_modal_score: float | None = None
    #: The likeliest few scorelines, likeliest first, the first being the modal
    #: one above. One score on its own reads as the forecast and appears to
    #: contradict the outcome -- "1-1" beside "Home win 57%" -- because an
    #: outcome sums a whole triangle of scorelines while a draw's mass sits on
    #: the diagonal. Empty for predictions stored before this existed.
    likely_scores: list[LikelyScore] = Field(default_factory=list)
    prob_over_2_5: float
    prob_btts: float

    market_prob_home: float | None = None
    market_prob_draw: float | None = None
    market_prob_away: float | None = None

    #: Squad strength as the model saw it: the *previous* season's ratings, the
    #: same numbers the Team Strength page shows. Exposed separately from the
    #: SHAP drivers because those list only each side's top five contributions,
    #: where form usually crowds strength out -- it reached just 83 of 480
    #: driver slots across one matchweek. ``None`` for a newly promoted side,
    #: which has no rating in its new league and is left as a genuine gap.
    strength_home_overall: float | None = None
    strength_home_attack: float | None = None
    strength_home_defence: float | None = None
    strength_away_overall: float | None = None
    strength_away_attack: float | None = None
    strength_away_defence: float | None = None

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


class MatchResult(BaseModel):
    """One settled match: what we predicted, and what happened.

    ``source`` separates two different claims. ``live`` means the prediction was
    stored before kickoff and settled afterwards -- a genuine track record.
    ``backtest`` means it came from the walk-forward run: the model was trained
    only on earlier seasons and never saw this match, so it is honestly
    out-of-sample, but it was generated retrospectively rather than timestamped
    in advance. Conflating the two would overstate the record.
    """

    source: str = Field(description="live | backtest")
    league: str
    date: date
    home_team: str
    away_team: str

    predicted_outcome: str
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    expected_goals_home: float | None = None
    expected_goals_away: float | None = None

    actual_home_goals: int
    actual_away_goals: int
    actual_result: str
    correct: bool
    #: Probability assigned to the outcome we picked, 0-100.
    confidence: float


class MatchResultPage(BaseModel):
    """A page of settled matches plus the summary for the same filters."""

    source: str
    total: int
    summary: Accuracy | None = None
    matches: list[MatchResult]


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

    #: Reference points that make `accuracy` readable. A match has three
    #: outcomes, so 52% is not a coin flip falling badly -- it sits against 33%
    #: for a random guess, `accuracy_base_rate` for always backing the home
    #: side, and `accuracy_market` for the bookmakers on the same fixtures.
    accuracy_base_rate: float = Field(
        description="Share won by the most common outcome -- always pick home"
    )
    accuracy_market: float | None = Field(
        default=None,
        description="How often the bookmakers' shortest price won, where odds exist",
    )

    by_league_accuracy: dict[str, float] = Field(
        default_factory=dict, description="Accuracy per league"
    )
    calibration: list[CalibrationBin] = Field(default_factory=list)


class ProjectedTeam(BaseModel):
    """One team's projected finish, from simulating the rest of the season.

    ``projected_points`` is the mean across simulated seasons; ``points_low``
    and ``points_high`` are the 10th and 90th percentiles. The band is the
    honest part -- five matchweeks in it is routinely twenty points wide, and a
    reader shown only the mean would take it for a forecast.
    """

    team: str
    played: int
    points: int
    goal_difference: int

    projected_points: float
    points_low: int
    points_high: int
    projected_position: float

    title_pct: float
    top_four_pct: float
    relegation_pct: float


class SeasonProjection(BaseModel):
    """A league's projected table plus how much of it is still guesswork."""

    league: str
    season: str
    #: Matches still to play. The whole basis for reading the table sceptically.
    remaining: int
    played: int
    teams: list[ProjectedTeam]


class SimilarPlayer(BaseModel):
    """One player judged similar, with the numbers that produced the judgement.

    Two scores, deliberately kept apart. ``fifa_similarity`` compares EA's
    attributes, which share one scale across outfield positions;
    ``percentile_similarity`` compares our own sub-ratings, which are ranks
    *within* a position and so are ``None`` whenever the comparison crosses one.
    Either can be ``None`` -- 8.6% of rated players have no EA entry, and a zero
    there would read as "nothing alike" rather than "not measured".

    50 means "no more alike than two random players in this position"; the scale
    is pinned to the median distance in the pool, not to the theoretical range.
    """

    player: str
    team: str
    league: str
    season: str
    position: str
    rating: int | None = None
    minutes: int

    fifa_similarity: float | None = None
    percentile_similarity: float | None = None
    combined: float | None = None

    #: The attribute values behind the score, so the page can show *why*.
    attributes: dict[str, float] = Field(default_factory=dict)
    sub_ratings: dict[str, float] = Field(default_factory=dict)


class SimilarPlayers(BaseModel):
    """A similarity search: who was asked about, and who came back."""

    player: PlayerRating
    #: EA attribute names used for this player -- the six, or the keeper five.
    attribute_names: list[str]
    #: The subject's own values, so a radar can overlay him against a candidate.
    #: Empty when he has no EA entry, which the page must say rather than draw.
    player_attributes: dict[str, float] = Field(default_factory=dict)
    player_sub_ratings: dict[str, float] = Field(default_factory=dict)
    same_role: bool
    results: list[SimilarPlayer]
