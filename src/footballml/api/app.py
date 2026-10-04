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
    CompetitionAccuracy,
    Driver,
    Health,
    LeagueStrength,
    LikelyScore,
    MatchResult,
    MatchResultPage,
    ModelInfo,
    PlayerPage,
    PlayerRating,
    Prediction,
    PredictRequest,
    ProjectedTeam,
    SeasonProjection,
    SimilarPlayer,
    SimilarPlayers,
    TeamForm,
    TeamStrength,
    TeamStrengthPage,
)
from footballml.data import (
    ODDS_COLUMNS,
    PROCESSED_DIR,
    load_team_match_history,
    load_team_strength,
)
from footballml.entities import load_aliases
from footballml.features.build import (
    LEAGUE_CODES,
    build_match_features,
    build_upcoming_features,
)
from footballml.form import build_index, recent_form
from footballml.ingest.european import SHOWN_COMPETITIONS
from footballml.ingest.matchhistory import LEAGUES, fetch_fixtures
from footballml.labels import humanise
from footballml.league_adjust import adjust as adjust_for_league
from footballml.league_adjust import fitted_model_version
from footballml.league_adjust import load as load_league_adjustment
from footballml.models.dixon_coles import score_matrix
from footballml.models.evaluate import (
    base_rate_probs,
    calibration_table,
    evaluate,
    odds_implied_probs,
)
from footballml.models.match_model import OUTCOMES, TOP_SCORES, feature_columns
from footballml.players.similarity import (
    SUB_RATINGS,
    attribute_set,
    similar_players,
    values_for,
)
from footballml.players.team_strength import MIN_RATED_PLAYERS

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
    projection: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Scored European ties, from `pipelines.validate_european`. These cannot be
    #: produced on demand like domestic ones: `state.features` is built from the
    #: domestic match history, which deliberately excludes European results so
    #: they never reach a rolling window or the Elo walk as history.
    european: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per-league goal-rate offsets for cross-league pairings. Empty until
    #: `pipelines.fit_league_adjustment` runs, and empty means every gap is
    #: zero, i.e. the correction is the identity.
    league_offsets: dict[str, float] = field(default_factory=dict)
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


class IncompatibleModelError(RuntimeError):
    """A newly trained model needs features this process cannot build.

    Raised only on reload, where there is a working model to keep. See
    `_load_state` for why that distinction matters.
    """

    def __init__(self, version: str, missing: list[str]) -> None:
        self.version = version
        self.missing = missing
        super().__init__(
            f"model {version} expects {len(missing)} feature(s) this process "
            f"cannot build, e.g. {missing[:3]}"
        )


def _load_state(startup: bool = False) -> dict[str, int]:
    """Read every CSV and the model artifact into `state`.

    Shared by startup and `POST /admin/reload`. The weekly job rewrites these
    files underneath a running server, and without a way to re-read them the
    site would serve last week's ratings until someone restarted it by hand --
    which is exactly the staleness the job exists to prevent.

    Args:
        startup: True when there is no previously-loaded state to fall back to.

    Returns:
        What was loaded, so a reload can report it rather than claiming success
        silently.

    Raises:
        IncompatibleModelError: On reload, when the artifact needs features this
            process cannot build. The running state is left untouched.
    """
    # Into locals first. `reload` swaps in a model trained by another process,
    # and that process may be running newer feature code than this one: on
    # 2026-09-24 a server started the previous day loaded a model built with
    # Elo, could not produce `home_elo`, and every /matches call died on a
    # KeyError. Publishing only after the pair is proven consistent means a bad
    # artifact costs a refused reload instead of a broken site.
    model, metadata = registry.load()
    logger.info("Loaded model %s", metadata.version)

    path = PROCESSED_DIR / "team_match_history_all.csv"
    tmh = load_team_match_history(path)
    strength = load_team_strength()
    features = build_match_features(tmh, strength=strength)

    missing = [c for c in metadata.feature_names if c not in features.columns]
    if missing and not startup:
        # Keep serving what works. The fix is a restart, not a retrain -- the
        # artifact is fine, this process is the stale half.
        logger.error(
            "Refusing to load model %s: it expects %d feature(s) this process "
            "cannot build, e.g. %s. Still serving %s. This usually means the "
            "feature code changed since the server started -- restart it.",
            metadata.version, len(missing), missing[:3],
            state.metadata.version if state.metadata else "nothing",
        )
        raise IncompatibleModelError(metadata.version, missing)
    if missing:
        # At startup there is nothing to fall back to, so serve what works and
        # say loudly what does not. Half a site beats none.
        logger.error(
            "Model %s expects %d feature(s) the API cannot build, e.g. %s -- "
            "predictions will fail; retrain with `python -m pipelines.train`",
            metadata.version, len(missing), missing[:3],
        )

    state.model, state.metadata = model, metadata
    state.tmh, state.strength, state.features = tmh, strength, features
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

    european_path = PROCESSED_DIR / "european_predictions.csv"
    if european_path.exists():
        state.european = pd.read_csv(european_path)
        state.european["Date"] = pd.to_datetime(state.european["Date"])
        logger.info("Loaded %d scored European ties", len(state.european))

    offsets = load_league_adjustment()
    if offsets:
        state.league_offsets = offsets
        logger.info(
            "Cross-league correction loaded for %d leagues: %s",
            len(offsets), ", ".join(f"{k} {v:+.3f}" for k, v in sorted(offsets.items())),
        )
        fitted_against = fitted_model_version()
        if metadata is not None and fitted_against not in (None, metadata.version):
            # Not fatal: a stale correction is still closer to right than none,
            # and refusing to serve would be worse. But it is measuring a model
            # that no longer exists, so it should not pass unremarked.
            logger.warning(
                "Cross-league correction was fitted against model %s, serving %s. "
                "Re-run `pipelines.validate_european` then "
                "`pipelines.fit_league_adjustment`.",
                fitted_against, metadata.version,
            )
    else:
        logger.info(
            "No cross-league correction; run `python -m pipelines.fit_league_adjustment`"
        )

    projection_path = PROCESSED_DIR / "season_projection.csv"
    if projection_path.exists():
        state.projection = pd.read_csv(projection_path)
        state.projection["Season"] = state.projection["Season"].astype(str)
        logger.info("Loaded projections for %d teams", len(state.projection))

    backtest_path = PROCESSED_DIR / "backtest_predictions.csv"
    if backtest_path.exists():
        state.backtest = pd.read_csv(backtest_path)
        state.backtest["Date"] = pd.to_datetime(state.backtest["Date"])
        logger.info("Loaded %d backtest predictions", len(state.backtest))

    # Scored fixtures are cached for a few minutes; after a reload that cache
    # describes the previous model, so drop it.
    state._upcoming = None
    state._upcoming_at = None

    logger.info("Loaded %d matches, %d teams", len(state.features), len(state.form_index))
    return {
        "matches": len(state.features),
        "teams": len(state.form_index),
        "players": len(state.players),
        "team_seasons": len(state.teams),
        "backtest_predictions": len(state.backtest),
    }


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model and history once, at startup."""
    _load_state(startup=True)
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


def _likely_scores(row: pd.Series) -> list[LikelyScore]:
    """The top scorelines off a scored row, if it carries them.

    Stored predictions written before these columns existed simply have none,
    and an empty list is the honest answer -- back-filling them from today's
    model would put numbers the prediction never made into the record.
    """
    out: list[LikelyScore] = []
    for k in range(TOP_SCORES):
        home, away, prob = (row.get(f"score_{k}_{part}") for part in ("home", "away", "prob"))
        if pd.isna(home) or pd.isna(away) or pd.isna(prob):
            continue
        out.append(
            LikelyScore(home=int(home), away=int(away), probability=round(float(prob), 4))
        )
    return out


#: How many goals each way the published scoreline grid covers.
#:
#: 0-5 is 36 cells and holds ~99.5% of the mass at typical goal rates. The
#: model's own matrix runs to 10 so the tail is not discarded from any
#: probability; it is only the *picture* that stops at 5, because a 11x11 grid
#: of mostly-zero cells is harder to read and says less.
SCORE_GRID_MAX = 5


def _score_grid(mu_home: float, mu_away: float) -> list[list[float]]:
    """The scoreline distribution behind one prediction, as a small grid.

    Rebuilt from the two goal rates rather than carried through the frame: the
    rates are what every caller already has, including a prediction stored
    months ago, and 36 numbers in every row of every response would be a lot of
    payload for something only the expanded card draws.
    """
    model = state.model
    if model is None or not np.isfinite([mu_home, mu_away]).all():
        return []
    matrix = score_matrix(
        np.array([mu_home]), np.array([mu_away]),
        rho=getattr(model, "rho_", 0.0),
        max_goals=getattr(model, "max_goals", 10),
    )[0]
    size = SCORE_GRID_MAX + 1
    return [[round(float(matrix[h][a]), 5) for a in range(size)] for h in range(size)]


def _to_predictions(
    frame: pd.DataFrame, drivers: list[dict] | None = None, with_form: bool = True
) -> list[Prediction]:
    """Convert a scored feature frame into API responses."""
    out = []
    for i, (_, r) in enumerate(frame.iterrows()):
        entry = Prediction(
            league=r["League"],
            date=pd.Timestamp(r["Date"]).date(),
            league_adjusted=bool(r.get("league_adjusted", False)),
            competition=(r.get("competition") or None),
            home_division=r.get("home_division") or None,
            away_division=r.get("away_division") or None,
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
            prob_modal_score=(
                round(float(r["prob_modal_score"]), 4)
                if pd.notna(r.get("prob_modal_score"))
                else None
            ),
            likely_scores=_likely_scores(r),
            score_grid=_score_grid(
                float(r["expected_goals_home"]), float(r["expected_goals_away"])
            ),
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


@app.get("/projections", response_model=SeasonProjection)
def projections(league: str = Query(..., description="Division code, e.g. E0")) -> SeasonProjection:
    """Projected final table, from simulating every remaining fixture.

    Precomputed by `pipelines.project_season` in the weekly job rather than on
    request: the simulation runs thousands of seasons across ~1,500 fixtures,
    and repeating that per visitor would be wasteful for a number that only
    changes when results do.
    """
    if state.projection.empty:
        raise HTTPException(
            404,
            "No projections yet. Run `python -m pipelines.project_season`.",
        )
    rows = state.projection[state.projection["League"] == league]
    if rows.empty:
        raise HTTPException(404, f"No projection for {league}")

    rows = rows.sort_values("projected_points", ascending=False)
    return SeasonProjection(
        league=league,
        season=str(rows["Season"].iloc[0]),
        remaining=int(rows["remaining"].iloc[0]),
        played=int(rows["played"].sum() // 2),
        teams=[
            ProjectedTeam(
                team=r["Team"],
                played=int(r["played"]),
                points=int(r["points"]),
                goal_difference=int(r["goal_difference"]),
                projected_points=round(float(r["projected_points"]), 1),
                points_low=int(r["points_low"]),
                points_high=int(r["points_high"]),
                projected_position=round(float(r["projected_position"]), 1),
                title_pct=round(float(r["title_pct"]), 4),
                top_four_pct=round(float(r["top_four_pct"]), 4),
                relegation_pct=round(float(r["relegation_pct"]), 4),
            )
            for _, r in rows.iterrows()
        ],
    )


@app.post("/admin/reload")
def reload_state() -> dict[str, object]:
    """Re-read the data files and model artifact without restarting.

    Called by `pipelines.weekly` after it refreshes results, ratings and the
    model. Without it the weekly job would update the files while the running
    server kept serving the previous week's, which is the staleness the job
    exists to remove.

    Not authenticated, deliberately: uvicorn binds 127.0.0.1, so this is
    reachable only from this machine. It reads local files and mutates nothing
    on disk. If the API is ever exposed beyond localhost, this needs a guard --
    the permissive CORS policy above does *not* make it remotely reachable, but
    a future bind to 0.0.0.0 would.
    """
    try:
        loaded = _load_state()
    except IncompatibleModelError as exc:
        # 409, not 500: nothing is broken, the request simply cannot be honoured
        # by a process running older feature code than the artifact it was asked
        # to load. The previous model is still being served.
        raise HTTPException(
            409,
            {
                "reloaded": False,
                "reason": "model needs features this process cannot build",
                "model_version": exc.version,
                "serving": state.metadata.version if state.metadata else None,
                "missing_features": exc.missing[:10],
                "fix": "restart the API so it runs the current feature code",
            },
        ) from exc

    logger.info("Reloaded on request: %s", loaded)
    return {"reloaded": True, "model_version": state.metadata.version, **loaded}


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
        # A European tie carries `league` = the home side's division, because the
        # model has no code for a competition. So filtering by "UCL" has to look
        # at `competition`, or it would match nothing at all.
        results = [p for p in results if league in (p.league, p.competition)]
    if not explain:
        # Strip rather than recompute: the caller asked for a lighter payload.
        results = [p.model_copy(update={"drivers_home": [], "drivers_away": []}) for p in results]
    return results


def _score_upcoming() -> list[Prediction]:
    """Fetch and score every published fixture. Expensive; call via the cache.

    Fixtures come from the season schedule rather than football-data's rolling
    file. That file is a snapshot, not a schedule -- on a Monday it routinely
    holds the round that has just been *played* -- and it publishes no UEFA
    divisions at all, so a Champions League tie could never appear through it.
    """
    model, _ = _require_model()

    fixtures = _upcoming_fixture_list()
    if fixtures.empty:
        return []

    scored = build_upcoming_features(state.tmh, fixtures, strength=state.strength)
    if scored.empty:
        logger.warning("No published fixtures matched a known team")
        return []

    carry = ["League", "Date", "HomeTeam", "AwayTeam"]
    extras = [c for c in (*ODDS_COLUMNS, "competition") if c in fixtures.columns]
    if extras:
        scored = scored.merge(
            fixtures[[*carry, *extras]].assign(Date=pd.to_datetime(fixtures["Date"])),
            on=carry,
            how="left",
        )
    scored["competition"] = scored.get("competition", store.DOMESTIC)
    scored["competition"] = scored["competition"].fillna(store.DOMESTIC)

    # A European tie is cross-league by construction, so it needs the same
    # correction `/predict` applies -- the model has never trained on a match
    # between two divisions and cannot judge the gap itself.
    divisions = {
        side: [_division_of(t, lg) for t, lg in zip(scored[col], scored["League"], strict=True)]
        for side, col in (("home", "HomeTeam"), ("away", "AwayTeam"))
    }
    mu_home, mu_away = model.predict_goal_rates(scored[state.columns])
    mu_home, mu_away = adjust_for_league(
        mu_home, mu_away, divisions["home"], divisions["away"], state.league_offsets
    )
    preds = model.frame_from_rates(mu_home, mu_away, index=scored.index)

    frame = pd.concat([scored.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    frame["home_division"] = divisions["home"]
    frame["away_division"] = divisions["away"]
    frame["league_adjusted"] = [
        h != a and bool(state.league_offsets)
        for h, a in zip(divisions["home"], divisions["away"], strict=True)
    ]
    drivers = model.explain(scored[state.columns], top_n=5)
    return _to_predictions(frame, drivers)


def _upcoming_fixture_list() -> pd.DataFrame:
    """Scheduled fixtures across the five leagues and the Champions League.

    Odds come from football-data, which is the only source for them and covers
    the domestic leagues only -- a European tie simply has none, and the
    accuracy page's market benchmark is absent for those rows rather than wrong.
    """
    from footballml.ingest.schedule import upcoming_fixtures

    try:
        domestic = upcoming_fixtures()
    except Exception as exc:  # noqa: BLE001 - a scraper outage must not 500
        logger.warning("No schedule available (%s); falling back to the fixture file", exc)
        domestic = fetch_fixtures()
    domestic = domestic.assign(competition=store.DOMESTIC) if not domestic.empty else domestic

    european = _upcoming_european()
    fixtures = pd.concat([f for f in (domestic, european) if not f.empty], ignore_index=True)
    if fixtures.empty:
        return fixtures

    odds = fetch_fixtures()
    odds_cols = [c for c in ODDS_COLUMNS if c in odds.columns]
    if odds_cols and not odds.empty:
        key = ["League", "Date", "HomeTeam", "AwayTeam"]
        odds = odds[[*key, *odds_cols]].assign(Date=pd.to_datetime(odds["Date"]))
        fixtures = fixtures.merge(odds, on=key, how="left")
    return fixtures


def _upcoming_european() -> pd.DataFrame:
    """Upcoming UEFA ties where both clubs are rateable.

    Only Champions League, and only ties between big-five clubs: a match
    against Benfica or Ajax has no form, no xG and no squad rating to predict
    from. That is 41% of a UCL round, which is why the page says so.
    """
    from footballml.ingest.european import big_five_only
    from footballml.ingest.schedule import COMPETITION_HORIZON_DAYS, upcoming_fixtures

    try:
        schedule = upcoming_fixtures(
            list(SHOWN_COMPETITIONS), horizon_days=COMPETITION_HORIZON_DAYS
        )
    except Exception as exc:  # noqa: BLE001 - one missing source is survivable
        logger.warning("No European schedule (%s)", exc)
        return pd.DataFrame()
    if schedule.empty:
        return pd.DataFrame()

    kept, _ = big_five_only(
        schedule.assign(FTHG=pd.NA, FTAG=pd.NA, FTR=pd.NA), state.tmh
    )
    if kept.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            # The home side's division, not the competition: `LEAGUE_CODES` has
            # no UEFA entry and a NaN `league_code` is a value no training row
            # ever carried.
            "League": kept["home_league"],
            "competition": kept["League"],
            "Season": kept["Season"],
            "Date": pd.to_datetime(kept["Date"]),
            "HomeTeam": kept["HomeTeam"],
            "AwayTeam": kept["AwayTeam"],
        }
    )


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
    if league in SHOWN_COMPETITIONS:
        return _european_matches(league, limit, offset)

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


def _european_matches(competition: str, limit: int, offset: int) -> list[Prediction]:
    """Played European ties, already scored.

    Served from `european_predictions.csv` rather than recomputed, because
    `state.features` holds only domestic matches -- European results are kept
    out of the match history on purpose. Without this, selecting a competition
    under "Recent results" is a permanent dead end: the filter would match
    nothing, for ever, with no way for the reader to tell that from "no matches
    have been played yet".
    """
    if state.european.empty:
        return []
    rows = state.european[state.european["competition"] == competition]
    rows = rows[rows["FTR"].notna()]
    window = rows.sort_values("Date", ascending=False).iloc[offset : offset + limit]
    if window.empty:
        return []

    frame = window.rename(columns={"home_league": "home_division", "away_league": "away_division"})
    # `_to_predictions` reads the result straight off FTR/FTHG/FTAG, so those
    # stay as they are. `League` becomes the home division for display, matching
    # how these were stored.
    frame = frame.assign(
        League=frame["home_division"],
        league_adjusted=frame["home_division"] != frame["away_division"],
    )
    # Form is attached, unlike before. It is drawn from `state.form_index`,
    # which is built from the *domestic* match history -- and that is exactly
    # the right window for a European tie: the last five league matches each
    # club played before it. Every one of the 72 clubs in this file is in that
    # index, since the ties are filtered to big-five clubs on both sides, so
    # nothing here is reaching for a team it cannot find.
    return _to_predictions(frame)


def _canonical_team(name: str, known: set[str]) -> str:
    """Accept a team by any name the project knows it under.

    `/predict` is the one endpoint callers hand a team name to, and the obvious
    place to get it from is `/teams` -- which serves the ratings table, spelled
    the way Understat does ("Manchester City"). The match history is spelled the
    way football-data does ("Man City"), so the two do not meet and every such
    request 404'd. `config/team_aliases.yaml` is where naming is reconciled, so
    resolve through it rather than asking callers to know which spelling wins.
    """
    if name in known:
        return name
    for source in ("understat", "fbref"):
        try:
            resolved = load_aliases(source).get(name)
        except FileNotFoundError:  # pragma: no cover - config ships with the repo
            continue
        if resolved in known:
            return resolved
    raise HTTPException(404, f"Unknown team {name!r}")


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest) -> Prediction:
    """Predict any pairing, using each side's form as of today.

    The only endpoint that runs the model live rather than serving stored output.
    """
    model, _ = _require_model()

    known = set(state.tmh["Team"].unique())
    home, away = (_canonical_team(t, known) for t in (request.home_team, request.away_team))
    if home == away:
        raise HTTPException(400, "A team cannot play itself")

    league = request.league
    if league is None:
        recent = state.tmh[state.tmh["Team"] == home].nlargest(1, "Date")
        league = str(recent["League"].iloc[0])

    fixture = pd.DataFrame(
        [
            {
                "League": league,
                "Season": str(state.tmh["Season"].max()),
                # Dated a day ahead so it sorts after every played match and
                # therefore picks up each side's complete history.
                "Date": pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
                "HomeTeam": home,
                "AwayTeam": away,
            }
        ]
    )

    scored = build_upcoming_features(state.tmh, fixture, strength=state.strength)
    if scored.empty:
        raise HTTPException(422, "Could not build features for that pairing")

    # The model cannot make this correction for itself: every match it trained
    # on was domestic, so the gap between two leagues is always zero in its
    # training data. See `footballml.league_adjust`. A same-league pairing has a
    # zero gap and passes through untouched.
    home_division = _division_of(home, league)
    away_division = _division_of(away, league)
    mu_home, mu_away = model.predict_goal_rates(scored[state.columns])
    corrected_home, corrected_away = adjust_for_league(
        mu_home, mu_away, home_division, away_division, state.league_offsets
    )
    preds = model.frame_from_rates(corrected_home, corrected_away, index=scored.index)

    frame = pd.concat([scored.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    frame["league_adjusted"] = home_division != away_division and bool(state.league_offsets)
    frame["home_division"] = home_division
    frame["away_division"] = away_division
    drivers = model.explain(scored[state.columns], top_n=5) if request.explain else None
    return _to_predictions(frame, drivers)[0]


def _division_of(team: str, fallback: str) -> str:
    """The domestic league a club plays in, for the cross-league correction.

    Taken from the match history rather than the fixture's own label, which for
    a hypothetical pairing is the *home* side's league and so would report both
    clubs as playing in the same one.

    A club we have never seen falls back to the label. If that label is a
    competition rather than a division -- "UCL" -- the correction would look up
    an offset that does not exist, get 0.0, and silently become the identity for
    exactly the match that needs it most. These fixtures are filtered to
    big-five clubs so it should not arise; warn loudly if it ever does.
    """
    rows = state.tmh[state.tmh["Team"] == team]
    if not rows.empty:
        return str(rows.nlargest(1, "Date")["League"].iloc[0])
    if fallback not in LEAGUE_CODES:
        logger.warning(
            "No division known for %r and the fixture is labelled %r, which is not "
            "a division -- the cross-league correction will not apply to it",
            team, fallback,
        )
    return fallback


def _by_competition(rows: pd.DataFrame) -> dict[str, CompetitionAccuracy]:
    """Each competition's record on its own, never pooled.

    Skill over a base rate is 12.8% on domestic matches and 6.2% on
    cross-league ones, measured over 822 UEFA ties. Publishing one blended
    figure would overstate the European predictions and flatter the domestic
    ones, which is the kind of claim this project takes trouble to avoid.
    """
    if rows.empty or "competition" not in rows.columns:
        return {}
    out = {}
    for name, block in rows.groupby(rows["competition"].fillna(store.DOMESTIC)):
        actual = block["actual_result"]
        metrics = evaluate(actual, block[PROB_COLUMNS].to_numpy())
        out[str(name)] = CompetitionAccuracy(
            n=metrics["n"],
            accuracy=round(metrics["accuracy"], 4),
            rps=round(metrics["rps"], 4),
            rps_base_rate=round(
                evaluate(actual, base_rate_probs(actual, len(block)))["rps"], 4
            ),
        )
    return out


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
        by_competition=_by_competition(rows),
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
        fifa_overall=num("fifa_overall", int),
        performance_rating=num("performance_rating", int),
        fifa_on_our_scale=num("fifa_on_our_scale", int),
        **{f"sub_{name}": num(f"sub_{name}", int) for name in _PLAYER_SUBS},
        **{
            stat: num(stat, int if stat in {"goals", "assists"} else float)
            for stat in _PLAYER_STATS
        },
    )


#: A season needs this share of a typical season's rated players before it is
#: shown by default.
#:
#: Ratings need a minutes threshold, so a season in progress fills up slowly.
#: Five matchweeks into 2026/27 only 181 players qualified against ~1,980 in a
#: full season -- and because the default was simply the latest season with any
#: ratings, the page silently switched to those 181 and searching for anyone
#: else returned nothing. The same reasoning as `MIN_RATED_PLAYERS` for teams.
#: Columns the player table may be sorted by.
#:
#: An allowlist rather than "any column that exists": the sort silently did
#: nothing for an unrecognised name, so a typo or a renamed column returned
#: whatever order the file happened to be in, looking every bit as deliberate
#: as a real sort.
#:
#: The sub-ratings are here even though they are position-dependent -- a
#: forward has no `sub_defending` -- because sorting by one is meaningful once
#: the caller has filtered to a position, which is exactly what the page does.
SORTABLE: frozenset[str] = frozenset({
    "rating", "minutes", "fifa_overall", "performance_rating",
    "position_group", "role", "Player", "Team", "League", "Season",
    "goals", "assists", "np_xg", "xa",
    *(f"sub_{name}" for name in _PLAYER_SUBS),
})

MIN_SEASON_SHARE = 0.5


def _default_season(rows: pd.DataFrame) -> str:
    """Newest season with enough rated players to represent a league."""
    counts = rows.groupby("Season").size()
    if counts.empty:
        return ""
    full = counts.max() * MIN_SEASON_SHARE
    eligible = counts[counts >= full]
    return str((eligible if not eligible.empty else counts).index.max())


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
    # Resolve it rather than just applying it: when the caller passes nothing,
    # the page still has to say which season it ended up showing.
    shown = str(season or _default_season(rows))
    rows = rows[rows["Season"].astype(str) == shown]

    if league:
        rows = rows[rows["League"] == league]
    if position:
        rows = rows[rows["role"] == position]
    if search:
        rows = rows[rows["Player"].str.contains(search, case=False, na=False)]
    if min_rating:
        rows = rows[rows["rating"] >= min_rating]

    if sort not in SORTABLE:
        raise HTTPException(
            400,
            f"Cannot sort by {sort!r}. A silently ignored sort returns rows in "
            f"an arbitrary order that looks deliberate. Sortable: "
            f"{', '.join(sorted(SORTABLE))}",
        )
    if sort in rows.columns:
        # Explicit tie-breakers, and a stable sort under them.
        #
        # Hundreds of players share a rating, and `sort_values` defaults to
        # quicksort, which is not stable -- so the order *within* a rating was
        # whatever the partitioning happened to produce, and changing the
        # minutes filter reshuffled players the filter had not touched. It
        # looked like the filter was reordering the table, which it was, just
        # not deliberately.
        #
        # Minutes then name is a real ordering rather than a tidier accident:
        # among equally rated players the one who played more is the more
        # established answer, and the name settles the rest. Both are present
        # for every row, so the result does not depend on the file's order.
        tie_breaks = [c for c in ("minutes", "Player") if c in rows.columns and c != sort]
        rows = rows.sort_values(
            [sort, *tie_breaks],
            ascending=[not descending, *(c != "Player" for c in tie_breaks)],
            na_position="last",
            kind="stable",
        )

    return PlayerPage(
        total=len(rows),
        players=[_to_player(r) for _, r in rows.iloc[offset : offset + limit].iterrows()],
        seasons=sorted(
            {str(s) for s in state.players["Season"].dropna().unique()}, reverse=True
        ),
        season=shown,
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


@app.get("/players/{name}/similar", response_model=SimilarPlayers)
def player_similarity(
    name: str,
    season: str | None = Query(None, description="Defaults to the player's latest"),
    limit: int = Query(10, ge=1, le=50),
    same_role: bool = Query(True, description="False widens the search, EA axis only"),
) -> SimilarPlayers:
    """Players who most resemble this one, on EA's attributes and on ours.

    The two scores answer different questions and are returned separately. EA's
    attributes share a scale across outfield positions, so they survive a
    cross-position search; our sub-ratings are percentiles *within* a position,
    so widening the search drops that axis rather than comparing ranks drawn
    from different populations.
    """
    if state.players.empty:
        raise HTTPException(404, "No player ratings loaded")

    rated = state.players[state.players["rated"]]

    # Default to the same season the player database shows, not simply the
    # player's newest. A season five matchweeks old has a handful of qualifying
    # players, so anchoring there would compare a 450-minute Alisson against a
    # near-empty pool while a team-mate absent from it got a full season.
    if season is None:
        default = _default_season(rated)
        if (rated["Player"].str.lower() == name.lower()).any():
            has_default = rated[
                (rated["Player"].str.lower() == name.lower())
                & (rated["Season"].astype(str) == default)
            ]
            season = default if not has_default.empty else None

    try:
        subject, matches = similar_players(rated, name, season, limit, same_role)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc

    return SimilarPlayers(
        player=_to_player(subject),
        attribute_names=list(attribute_set(str(subject["role"]))),
        player_attributes=values_for(
            subject, [f"fifa_{a}" for a in attribute_set(str(subject["role"]))]
        ),
        player_sub_ratings=values_for(subject, list(SUB_RATINGS)),
        same_role=same_role,
        results=[
            SimilarPlayer(
                player=m.player,
                team=m.team,
                league=m.league,
                season=m.season,
                position=m.role,
                rating=m.rating,
                minutes=m.minutes,
                fifa_similarity=m.fifa_similarity,
                percentile_similarity=m.percentile_similarity,
                combined=m.combined,
                attributes=m.attributes,
                sub_ratings=m.sub_ratings,
            )
            for m in matches
        ],
    )


@app.get("/teams", response_model=TeamStrengthPage)
def teams(
    league: str | None = Query(None, description="Division code, e.g. E0"),
    season: str | None = Query(None, description="Defaults to the latest available"),
    limit: int = Query(100, le=200),
) -> TeamStrengthPage:
    """Team ratings built from player ratings, strongest first."""
    if state.teams.empty:
        raise HTTPException(
            404, "No team ratings loaded. Run `python -m pipelines.build_players`."
        )

    seasons = sorted(
        {str(s) for s in state.teams["Season"].dropna().unique()}, reverse=True
    )
    shown = season or (seasons[0] if seasons else None)
    rows = state.teams
    rows = rows[rows["Season"].astype(str) == str(shown)]
    if league:
        rows = rows[rows["League"] == league]

    rows = rows.sort_values("strength_overall", ascending=False).head(limit)

    def num(row: pd.Series, column: str) -> float | None:
        value = row.get(column)
        return None if pd.isna(value) else round(float(value), 1)

    return TeamStrengthPage(
        teams=[
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
        ],
        seasons=seasons,
        season=str(shown) if shown is not None else None,
    )


@app.get("/leagues/strength", response_model=list[LeagueStrength])
def league_strength(
    season: str | None = Query(None, description="Defaults to the latest available"),
) -> list[LeagueStrength]:
    """How the five leagues compare, averaged over each one's squad ratings.

    Context for a cross-league question: `/predict` will happily score Real
    Madrid against Man City, but the model has never seen the two divisions
    meet, so the honest framing is "here is how far apart these leagues are on
    our own ratings" rather than a fixture forecast.

    The comparison is meaningful because a rating is not league-relative -- see
    `LeagueStrength` -- with one caveat worth stating: the performance half is
    built from per-90 output, which is easier to accumulate against weaker
    opponents. That flatters the weaker leagues, so if anything the real gaps
    are wider than these.
    """
    if state.teams.empty:
        raise HTTPException(
            404, "No team ratings loaded. Run `python -m pipelines.build_players`."
        )

    rows = state.teams
    rows = rows[rows["Season"] == (season or rows["Season"].max())]
    rows = rows[rows["strength_overall"].notna()]
    if rows.empty:
        raise HTTPException(404, f"No team ratings for season {season!r}")

    out: list[LeagueStrength] = []
    for league, block in rows.groupby("League"):
        strongest = block.loc[block["strength_overall"].idxmax()]
        weakest = block.loc[block["strength_overall"].idxmin()]
        out.append(
            LeagueStrength(
                league=str(league),
                season=str(block["Season"].iloc[0]),
                n_teams=len(block),
                mean_strength=round(float(block["strength_overall"].mean()), 1),
                median_strength=round(float(block["strength_overall"].median()), 1),
                # Population sd, not sample: these are all the clubs in the
                # league, not a sample drawn from a larger set.
                spread=round(float(block["strength_overall"].std(ddof=0)), 1),
                strongest_team=str(strongest["Team"]),
                strongest_strength=round(float(strongest["strength_overall"]), 1),
                weakest_team=str(weakest["Team"]),
                weakest_strength=round(float(weakest["strength_overall"]), 1),
            )
        )
    return sorted(out, key=lambda x: x.mean_strength, reverse=True)


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

    # Not simply the newest season this player file has. Five matchweeks into a
    # campaign barely anyone has cleared the minutes floor -- 181 rated players
    # across all five leagues in 2026/27 -- so `max()` picked a season in which
    # Liverpool had two rated players, a goalkeeper and a centre back. Team
    # *ratings* already refuse a season that thin (`MIN_RATED_PLAYERS`), so the
    # panel header read 2025/26 while the squad under it was 2026/27 and two
    # names long.
    if season is None:
        counts = rows.groupby("Season").size()
        enough = counts[counts >= MIN_RATED_PLAYERS]
        season = (enough.index.max() if not enough.empty else counts.index.max())
    rows = rows[rows["Season"].astype(str) == str(season)]
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
