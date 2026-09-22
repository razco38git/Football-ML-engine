"""The prediction API.

Read endpoints serve precomputed or cached results. Nothing here scrapes a
source during a request, and nothing retrains -- the model is loaded once at
startup from a versioned artifact.

``POST /predict`` is the one live-inference path: it builds features for an
arbitrary pairing on demand, which is what makes a "what if these two played?"
control possible in the UI.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from footballml import registry, store
from footballml.api.schemas import (
    Accuracy,
    CalibrationBin,
    Driver,
    Health,
    MatchResult,
    MatchResultPage,
    ModelInfo,
    PlayerPage,
    PlayerRating,
    Prediction,
    PredictRequest,
    TeamForm,
    TeamStrength,
)
from footballml.data import (
    ODDS_COLUMNS,
    PROCESSED_DIR,
    load_team_match_history,
    load_team_strength,
)
from footballml.features.build import build_match_features, build_upcoming_features
from footballml.form import build_index, recent_form
from footballml.ingest.matchhistory import LEAGUES, fetch_fixtures
from footballml.labels import humanise
from footballml.models.evaluate import (
    base_rate_probs,
    calibration_table,
    evaluate,
    odds_implied_probs,
)
from footballml.models.match_model import OUTCOMES, feature_columns

logger = logging.getLogger(__name__)

#: Upcoming fixtures change at most a few times a day, and predicting them costs
#: a feature build over the full history. Cache rather than recompute per request.
FIXTURE_CACHE_TTL = timedelta(minutes=30)

#: Outcome probability columns, in H/D/A order.
PROB_COLUMNS = ["prob_home_win", "prob_draw", "prob_away_win"]


@dataclass
class State:
    """Everything loaded once at startup and shared across requests."""

    model: Any = None
    metadata: registry.ModelMetadata | None = None
    tmh: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    columns: list[str] = field(default_factory=list)
    form_index: dict = field(default_factory=dict)
    players: pd.DataFrame = field(default_factory=pd.DataFrame)
    teams: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Squad strength for the model's features -- distinct from `teams`, which is
    #: the same file presented for display. Every feature build must pass it, or
    #: the artifact's expected columns cannot be produced.
    strength: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Scored predictions, not raw fixtures. Rebuilding features over the full
    # history costs ~6.5s, which is far too slow to repeat per request when the
    # answer only changes when the fixture list does.
    _upcoming: list[Prediction] | None = None
    _upcoming_at: datetime | None = None

    @property
    def upcoming_is_stale(self) -> bool:
        if self._upcoming is None or self._upcoming_at is None:
            return True
        return datetime.now(UTC) - self._upcoming_at > FIXTURE_CACHE_TTL


state = State()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model and history once, at startup."""
    state.model, state.metadata = registry.load()
    logger.info("Loaded model %s", state.metadata.version)

    path = PROCESSED_DIR / "team_match_history_all.csv"
    state.tmh = load_team_match_history(path)
    state.strength = load_team_strength()
    state.features = build_match_features(state.tmh, strength=state.strength)

    # Fail loudly at startup rather than on the first request: an artifact
    # trained with columns this process cannot build would 500 every prediction.
    missing = [c for c in state.metadata.feature_names if c not in state.features.columns]
    if missing:
        logger.error(
            "Model %s expects %d feature(s) the API cannot build, e.g. %s -- "
            "retrain with `python -m pipelines.train`",
            state.metadata.version, len(missing), missing[:3],
        )
    state.columns = feature_columns(state.features)
    state.form_index = build_index(state.tmh)

    players_path = PROCESSED_DIR / "player_ratings.csv"
    if players_path.exists():
        state.players = pd.read_csv(players_path)
        state.players["Season"] = state.players["Season"].astype(str)
        logger.info("Loaded %d player-seasons", len(state.players))
    else:
        logger.warning("No player ratings; run `python -m pipelines.build_players`")

    teams_path = PROCESSED_DIR / "team_strength.csv"
    if teams_path.exists():
        state.teams = pd.read_csv(teams_path)
        state.teams["Season"] = state.teams["Season"].astype(str)
        logger.info("Loaded %d team-seasons", len(state.teams))

    backtest_path = PROCESSED_DIR / "backtest_predictions.csv"
    if backtest_path.exists():
        state.backtest = pd.read_csv(backtest_path)
        state.backtest["Date"] = pd.to_datetime(state.backtest["Date"])
        logger.info("Loaded %d backtest predictions", len(state.backtest))

    logger.info("Loaded %d matches, %d teams", len(state.features), len(state.form_index))
    yield


app = FastAPI(
    title="Football ML Engine",
    description="Match outcome and expected-goals predictions for the top 5 European leagues.",
    version="0.1.0",
    lifespan=lifespan,
)

# The frontend is served from a different origin in development and from a CDN
# in production, so browser requests are cross-origin either way.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_model() -> tuple[Any, registry.ModelMetadata]:
    if state.model is None or state.metadata is None:
        raise HTTPException(503, "Model not loaded")
    return state.model, state.metadata


def _form(team: str, before: pd.Timestamp) -> TeamForm:
    return TeamForm(**vars(recent_form(state.form_index, team, before)))


def _to_predictions(
    frame: pd.DataFrame, drivers: list[dict] | None = None, with_form: bool = True
) -> list[Prediction]:
    """Convert a scored feature frame into API responses."""
    out = []
    for i, (_, r) in enumerate(frame.iterrows()):
        entry = Prediction(
            league=r["League"],
            date=pd.Timestamp(r["Date"]).date(),
            home_team=r["HomeTeam"],
            away_team=r["AwayTeam"],
            expected_goals_home=round(float(r["expected_goals_home"]), 3),
            expected_goals_away=round(float(r["expected_goals_away"]), 3),
            prob_home_win=round(float(r["prob_home_win"]), 4),
            prob_draw=round(float(r["prob_draw"]), 4),
            prob_away_win=round(float(r["prob_away_win"]), 4),
            predicted_outcome=r["predicted_outcome"],
            modal_score_home=int(r["modal_score_home"]),
            modal_score_away=int(r["modal_score_away"]),
            prob_over_2_5=round(float(r["prob_over_2_5"]), 4),
            prob_btts=round(float(r["prob_btts"]), 4),
        )
        # Squad strength exactly as the model saw it, straight off the scored
        # frame rather than re-read from the CSV, so the page cannot drift from
        # the features behind the prediction.
        for side in ("home", "away"):
            for metric in ("overall", "attack", "defence"):
                value = r.get(f"{side}_strength_{metric}")
                if pd.notna(value):
                    setattr(entry, f"strength_{side}_{metric}", round(float(value), 1))

        if pd.notna(r.get("FTR")):
            entry.actual_home_goals = int(r["FTHG"])
            entry.actual_away_goals = int(r["FTAG"])
            entry.actual_result = r["FTR"]
        if pd.notna(r.get("B365H")):
            market = odds_implied_probs(pd.DataFrame([r]), ODDS_COLUMNS)[0]
            entry.market_prob_home = round(float(market[0]), 4)
            entry.market_prob_draw = round(float(market[1]), 4)
            entry.market_prob_away = round(float(market[2]), 4)
        if with_form:
            when = pd.Timestamp(r["Date"])
            entry.form_home = _form(r["HomeTeam"], when)
            entry.form_away = _form(r["AwayTeam"], when)
        if drivers:
            for side, target in (("home", "drivers_home"), ("away", "drivers_away")):
                setattr(
                    entry,
                    target,
                    [
                        Driver(feature=n, label=humanise(n), contribution=round(v, 4))
                        for n, v in drivers[i][side]
                    ],
                )
        out.append(entry)
    return out


@app.get("/health", response_model=Health)
def health() -> Health:
    _, meta = _require_model()
    return Health(
        status="ok",
        model_version=meta.version,
        trained_through=pd.Timestamp(meta.trained_through).date(),
        n_train=meta.n_train,
    )


@app.get("/leagues")
def leagues() -> dict[str, str]:
    """Division codes to full names."""
    return LEAGUES


@app.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    """Every trained version, newest first, with its held-out metrics."""
    return [
        ModelInfo(
            version=m.version,
            trained_at=m.trained_at,
            trained_through=pd.Timestamp(m.trained_through).date(),
            n_train=m.n_train,
            n_features=m.n_features,
            leagues=m.leagues,
            rho=m.rho,
            metrics=m.metrics,
        )
        for m in registry.list_versions()
    ]


@app.get("/fixtures/upcoming", response_model=list[Prediction])
def upcoming(
    league: str | None = Query(None, description="Division code, e.g. E0"),
    explain: bool = Query(False, description="Include SHAP drivers"),
) -> list[Prediction]:
    """Predictions for fixtures that have not been played.

    Always scored with explanations and cached whole, then filtered per request.
    Explanations add little to the cost once features are built, and computing
    them eagerly means toggling "show analysis" in the UI is instant.
    """
    if state.upcoming_is_stale:
        state._upcoming = _score_upcoming()
        state._upcoming_at = datetime.now(UTC)

    results = state._upcoming or []
    if league:
        results = [p for p in results if p.league == league]
    if not explain:
        # Strip rather than recompute: the caller asked for a lighter payload.
        results = [p.model_copy(update={"drivers_home": [], "drivers_away": []}) for p in results]
    return results


def _score_upcoming() -> list[Prediction]:
    """Fetch and score every published fixture. Expensive; call via the cache."""
    model, _ = _require_model()

    fixtures = fetch_fixtures()
    if fixtures.empty:
        return []

    scored = build_upcoming_features(state.tmh, fixtures, strength=state.strength)
    if scored.empty:
        logger.warning("No published fixtures matched a known team")
        return []

    odds_cols = [c for c in ODDS_COLUMNS if c in fixtures.columns]
    if odds_cols:
        key = ["League", "Date", "HomeTeam", "AwayTeam"]
        scored = scored.merge(
            fixtures[[*key, *odds_cols]].assign(Date=pd.to_datetime(fixtures["Date"])),
            on=key,
            how="left",
        )

    preds = model.predict_frame(scored[state.columns])
    frame = pd.concat([scored.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    drivers = model.explain(scored[state.columns], top_n=5)
    return _to_predictions(frame, drivers)


@app.get("/matches", response_model=list[Prediction])
def matches(
    league: str | None = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
) -> list[Prediction]:
    """Recent played matches, predicted and compared against what happened.

    Scored with the current model, so these are retrospective rather than a
    track record. The honest track record lives at ``/accuracy``, which reads
    only predictions stored before kickoff.
    """
    model, _ = _require_model()

    played = state.features[state.features["FTR"].notna()]
    if league:
        played = played[played["League"] == league]
    window = played.sort_values("Date", ascending=False).iloc[offset : offset + limit]
    if window.empty:
        return []

    preds = model.predict_frame(window[state.columns])
    frame = pd.concat([window.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    return _to_predictions(frame)


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest) -> Prediction:
    """Predict any pairing, using each side's form as of today.

    The only endpoint that runs the model live rather than serving stored output.
    """
    model, _ = _require_model()

    known = set(state.tmh["Team"].unique())
    for team in (request.home_team, request.away_team):
        if team not in known:
            raise HTTPException(404, f"Unknown team {team!r}")
    if request.home_team == request.away_team:
        raise HTTPException(400, "A team cannot play itself")

    league = request.league
    if league is None:
        recent = state.tmh[state.tmh["Team"] == request.home_team].nlargest(1, "Date")
        league = str(recent["League"].iloc[0])

    fixture = pd.DataFrame(
        [
            {
                "League": league,
                "Season": str(state.tmh["Season"].max()),
                # Dated a day ahead so it sorts after every played match and
                # therefore picks up each side's complete history.
                "Date": pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
                "HomeTeam": request.home_team,
                "AwayTeam": request.away_team,
            }
        ]
    )

    scored = build_upcoming_features(state.tmh, fixture, strength=state.strength)
    if scored.empty:
        raise HTTPException(422, "Could not build features for that pairing")

    preds = model.predict_frame(scored[state.columns])
    frame = pd.concat([scored.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    drivers = model.explain(scored[state.columns], top_n=5) if request.explain else None
    return _to_predictions(frame, drivers)[0]


@app.get("/accuracy", response_model=Accuracy)
def accuracy(league: str | None = Query(None)) -> Accuracy:
    """Track record over predictions stored *before* kickoff and later settled."""
    rows = store.settled()
    if rows.empty:
        raise HTTPException(
            404,
            "No settled predictions yet. Predictions accumulate once "
            "`python -m pipelines.score_upcoming` runs regularly.",
        )
    if league:
        rows = rows[rows["League"] == league]
        if rows.empty:
            raise HTTPException(404, f"No settled predictions for {league}")

    probs = rows[PROB_COLUMNS].to_numpy()
    actual = rows["actual_result"]
    metrics = evaluate(actual, probs)

    result = Accuracy(
        n=metrics["n"],
        accuracy=round(metrics["accuracy"], 4),
        rps=round(metrics["rps"], 4),
        log_loss=round(metrics["log_loss"], 4),
        brier=round(metrics["brier"], 4),
        rps_base_rate=round(evaluate(actual, base_rate_probs(actual, len(rows)))["rps"], 4),
        accuracy_base_rate=round(float(actual.value_counts(normalize=True).max()), 4),
        by_league_accuracy={
            str(lg): round(
                evaluate(g["actual_result"], g[PROB_COLUMNS].to_numpy())["accuracy"], 4
            )
            for lg, g in rows.groupby("League")
        },
        calibration=[
            CalibrationBin(**row)
            for row in calibration_table(actual, probs)
            .drop(columns=["gap"])
            .to_dict("records")
        ],
    )

    if all(c in rows.columns for c in ODDS_COLUMNS):
        with_odds = rows[rows[list(ODDS_COLUMNS)].notna().all(axis=1)]
        if not with_odds.empty:
            market = odds_implied_probs(with_odds, ODDS_COLUMNS)
            result.rps_market = round(evaluate(with_odds["actual_result"], market)["rps"], 4)

    return result


#: Sub-rating and stat columns surfaced by the player endpoints.
_PLAYER_SUBS = (
    "finishing", "creation", "involvement", "volume", "defending",
    "shot_stopping", "reliability", "workload", "penalties",
)
_PLAYER_STATS = (
    "goals", "assists", "np_xg", "xa", "key_passes_per90",
    "interceptions_per90", "tackles_won_per90", "save_pct", "goals_against_per90",
)


def _to_player(row: pd.Series) -> PlayerRating:
    """Convert one rating row into an API response."""

    def num(column: str, cast: type) -> int | float | None:
        value = row.get(column)
        return None if pd.isna(value) else cast(value)

    return PlayerRating(
        player=str(row["Player"]),
        team=str(row["Team"]),
        league=str(row["League"]),
        season=str(row["Season"]),
        # `role` is the detailed position (CB/FB/MID/AMW/FWD/GK) derived from
        # EA data; `position_group` is the coarse Understat fallback.
        position=str(row.get("role") or row.get("position_group") or "?"),
        minutes=int(row["minutes"]),
        rating=num("rating", int),
        rated=bool(row.get("rated", False)),
        unrated_reason=(
            None if pd.isna(row.get("unrated_reason")) else str(row["unrated_reason"])
        ),
        **{f"sub_{name}": num(f"sub_{name}", int) for name in _PLAYER_SUBS},
        **{
            stat: num(stat, int if stat in {"goals", "assists"} else float)
            for stat in _PLAYER_STATS
        },
    )


@app.get("/players", response_model=PlayerPage)
def players(
    league: str | None = Query(None, description="Division code, e.g. E0"),
    position: str | None = Query(None, description="GK, D, M or F"),
    season: str | None = Query(None, description="Defaults to the latest rated season"),
    search: str | None = Query(None, description="Case-insensitive name match"),
    min_rating: int = Query(0, ge=0, le=99),
    sort: str = Query("rating"),
    descending: bool = Query(True),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
) -> PlayerPage:
    """The rated player database, filtered and sorted."""
    if state.players.empty:
        raise HTTPException(
            404, "No player ratings loaded. Run `python -m pipelines.build_players`."
        )

    rows = state.players[state.players["rated"]]
    # Default to the most recent season with ratings rather than mixing seasons,
    # which would otherwise let an old peak outrank current form.
    rows = rows[rows["Season"] == (season or rows["Season"].max())]

    if league:
        rows = rows[rows["League"] == league]
    if position:
        rows = rows[rows["role"] == position]
    if search:
        rows = rows[rows["Player"].str.contains(search, case=False, na=False)]
    if min_rating:
        rows = rows[rows["rating"] >= min_rating]

    if sort in rows.columns:
        rows = rows.sort_values(sort, ascending=not descending, na_position="last")

    return PlayerPage(
        total=len(rows),
        players=[_to_player(r) for _, r in rows.iloc[offset : offset + limit].iterrows()],
    )


@app.get("/players/{name}", response_model=list[PlayerRating])
def player_history(name: str) -> list[PlayerRating]:
    """Every rated season for one player, newest first."""
    if state.players.empty:
        raise HTTPException(404, "No player ratings loaded")

    rows = state.players[state.players["Player"].str.lower() == name.lower()]
    if rows.empty:
        raise HTTPException(404, f"Unknown player {name!r}")

    rows = rows.sort_values("Season", ascending=False)
    return [_to_player(r) for _, r in rows.iterrows()]


@app.get("/teams", response_model=list[TeamStrength])
def teams(
    league: str | None = Query(None, description="Division code, e.g. E0"),
    season: str | None = Query(None, description="Defaults to the latest available"),
    limit: int = Query(100, le=200),
) -> list[TeamStrength]:
    """Team ratings built from player ratings, strongest first."""
    if state.teams.empty:
        raise HTTPException(
            404, "No team ratings loaded. Run `python -m pipelines.build_players`."
        )

    rows = state.teams
    rows = rows[rows["Season"] == (season or rows["Season"].max())]
    if league:
        rows = rows[rows["League"] == league]

    rows = rows.sort_values("strength_overall", ascending=False).head(limit)

    def num(row: pd.Series, column: str) -> float | None:
        value = row.get(column)
        return None if pd.isna(value) else round(float(value), 1)

    return [
        TeamStrength(
            team=str(r["Team"]),
            league=str(r["League"]),
            season=str(r["Season"]),
            method=str(r.get("method", "eleven")),
            n_players=int(r.get("n_players", 0)),
            strength_overall=round(float(r["strength_overall"]), 1),
            strength_goalkeeper=num(r, "strength_goalkeeper"),
            strength_defence=num(r, "strength_defence"),
            strength_midfield=num(r, "strength_midfield"),
            strength_attack=num(r, "strength_attack"),
        )
        for _, r in rows.iterrows()
    ]


@app.get("/teams/{name}/squad", response_model=list[PlayerRating])
def team_squad(
    name: str,
    season: str | None = Query(None),
) -> list[PlayerRating]:
    """The players behind a team's rating, highest minutes first.

    Shows who the eleven was built from, so the team number can be checked
    against the players that produced it.
    """
    if state.players.empty:
        raise HTTPException(404, "No player ratings loaded")

    rows = state.players[
        state.players["rated"] & (state.players["Team"].str.lower() == name.lower())
    ]
    if rows.empty:
        raise HTTPException(404, f"Unknown team {name!r}")

    rows = rows[rows["Season"] == (season or rows["Season"].max())]
    return [_to_player(r) for _, r in rows.sort_values("minutes", ascending=False).iterrows()]


#: Column names differ between the two sources; this maps each onto MatchResult.
_LIVE_COLUMNS = {
    "prob_home_win": "prob_home_win",
    "prob_draw": "prob_draw",
    "prob_away_win": "prob_away_win",
    "predicted_outcome": "predicted_outcome",
    "expected_goals_home": "expected_goals_home",
    "expected_goals_away": "expected_goals_away",
    "actual_home_goals": "actual_home_goals",
    "actual_away_goals": "actual_away_goals",
    "actual_result": "actual_result",
}
_BACKTEST_COLUMNS = {
    "prob_home_win": "prob_H",
    "prob_draw": "prob_D",
    "prob_away_win": "prob_A",
    "expected_goals_home": "mu_home",
    "expected_goals_away": "mu_away",
    "actual_home_goals": "FTHG",
    "actual_away_goals": "FTAG",
    "actual_result": "FTR",
}


def _to_match_result(row: pd.Series, source: str, columns: dict[str, str]) -> MatchResult:
    """Build one settled-match row from whichever source it came from."""
    probs = {
        key: float(row[columns[key]])
        for key in ("prob_home_win", "prob_draw", "prob_away_win")
    }
    actual = str(row[columns["actual_result"]])

    predicted = row.get(columns.get("predicted_outcome", ""))
    if pd.isna(predicted):
        # The backtest stores probabilities but not the pick; it is the argmax.
        by_outcome = {
            "H": probs["prob_home_win"],
            "D": probs["prob_draw"],
            "A": probs["prob_away_win"],
        }
        predicted = max(by_outcome, key=lambda k: by_outcome[k])

    chosen = {"H": "prob_home_win", "D": "prob_draw", "A": "prob_away_win"}[str(predicted)]

    def maybe(key: str) -> float | None:
        column = columns.get(key)
        if column is None or column not in row or pd.isna(row[column]):
            return None
        return round(float(row[column]), 3)

    return MatchResult(
        source=source,
        league=str(row["League"]),
        date=pd.Timestamp(row["Date"]).date(),
        home_team=str(row["HomeTeam"]),
        away_team=str(row["AwayTeam"]),
        predicted_outcome=str(predicted),
        prob_home_win=round(probs["prob_home_win"], 4),
        prob_draw=round(probs["prob_draw"], 4),
        prob_away_win=round(probs["prob_away_win"], 4),
        expected_goals_home=maybe("expected_goals_home"),
        expected_goals_away=maybe("expected_goals_away"),
        actual_home_goals=int(row[columns["actual_home_goals"]]),
        actual_away_goals=int(row[columns["actual_away_goals"]]),
        actual_result=actual,
        correct=str(predicted) == actual,
        confidence=round(probs[chosen] * 100, 1),
    )


def _summarise(rows: pd.DataFrame, prob_cols: list[str], actual_col: str) -> Accuracy:
    """Metric block for a set of settled matches."""
    probs = rows[prob_cols].to_numpy()
    actual = rows[actual_col]
    metrics = evaluate(actual, probs)

    # How often the bookmakers' shortest price won, over whichever rows carry
    # odds. This is the benchmark worth publishing beside our own accuracy: the
    # market has team news and money behind it, so it is the realistic ceiling
    # rather than a straw man.
    accuracy_market = None
    if all(c in rows.columns for c in ODDS_COLUMNS):
        priced = rows[list(ODDS_COLUMNS)].notna().all(axis=1)
        if priced.any():
            sub = rows[priced]
            favourite = np.array(OUTCOMES)[
                odds_implied_probs(sub, ODDS_COLUMNS).argmax(axis=1)
            ]
            accuracy_market = round(float((favourite == sub[actual_col]).mean()), 4)

    return Accuracy(
        n=metrics["n"],
        accuracy=round(metrics["accuracy"], 4),
        rps=round(metrics["rps"], 4),
        log_loss=round(metrics["log_loss"], 4),
        brier=round(metrics["brier"], 4),
        rps_base_rate=round(evaluate(actual, base_rate_probs(actual, len(rows)))["rps"], 4),
        # Always predicting whichever outcome is most common -- in practice a
        # home win. The floor any real model has to clear.
        accuracy_base_rate=round(float(actual.value_counts(normalize=True).max()), 4),
        accuracy_market=accuracy_market,
        by_league_accuracy={
            str(lg): round(evaluate(g[actual_col], g[prob_cols].to_numpy())["accuracy"], 4)
            for lg, g in rows.groupby("League")
        },
        calibration=[
            CalibrationBin(**row)
            for row in calibration_table(actual, probs).drop(columns=["gap"]).to_dict("records")
        ],
    )


@app.get("/accuracy/history", response_model=MatchResultPage)
def accuracy_history(
    source: str = Query("backtest", description="live | backtest"),
    league: str | None = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
) -> MatchResultPage:
    """Settled matches with the prediction beside the real score, newest first.

    ``live`` reads only predictions stored before kickoff -- the honest track
    record, currently very short. ``backtest`` reads the walk-forward run, where
    every match was predicted by a model trained solely on earlier seasons.
    Both are out-of-sample; only the first was committed to in advance, and the
    two are kept apart so the stronger claim is never made for the weaker data.
    """
    if source not in {"live", "backtest"}:
        raise HTTPException(400, "source must be 'live' or 'backtest'")

    if source == "live":
        rows = store.settled()
        columns, prob_cols, actual_col = _LIVE_COLUMNS, PROB_COLUMNS, "actual_result"
    else:
        rows = state.backtest
        columns = _BACKTEST_COLUMNS
        prob_cols, actual_col = ["prob_H", "prob_D", "prob_A"], "FTR"

    if rows.empty:
        hint = (
            "Run `python -m pipelines.score_upcoming` regularly to build one."
            if source == "live"
            else "Run `python -m pipelines.backtest --all-leagues --save-predictions`."
        )
        raise HTTPException(404, f"No {source} results yet. {hint}")

    rows = rows[rows[actual_col].notna()]
    if league:
        rows = rows[rows["League"] == league]
    if rows.empty:
        raise HTTPException(404, f"No {source} results for {league}")

    summary = _summarise(rows, prob_cols, actual_col)
    window = rows.sort_values("Date", ascending=False).iloc[offset : offset + limit]

    return MatchResultPage(
        source=source,
        total=len(rows),
        summary=summary,
        matches=[_to_match_result(r, source, columns) for _, r in window.iterrows()],
    )
