"""Does a defender's rating travel with him when he changes club?

A centre back's rating correlates +0.67 with his own club's defensive record,
and `config/player_rating.yaml` gives CB ``team_defence: 0.30`` against FB's
``0.0`` -- so 30% of a centre back's composite *is* his club's expected goals
against, goals against and clean sheet rate. The config states the consequence
("Saliba and Gabriel both take Arsenal's 98 and both land on 85"); what is
unknown is whether that makes the number wrong about the *player*.

A transfer separates the two. If the rating is about him it should predict his
next EA assessment equally well whether or not he moves; if it is about the side
he just left, it should attenuate.

The full protocol, its acceptance criteria and the reasoning behind every
constant below live in ``docs/PROTOCOLS.md``. This module holds the constants so
they are readable in the source before any number exists, which is the standard
the extended-Elo and positional-quota experiments set.

.. warning::
    **Writes nothing.** Every frame lives in memory and the production rating
    and strength files are never opened for writing. ``tests/test_rating_travel_ab.py``
    asserts that by scanning this source, because a ``finally`` does not survive
    the process being killed and one experiment already left ``team_strength.csv``
    holding its own numbers with the API serving them.

.. note::
    **Nothing here changes production weights under any outcome.** A candidate
    reweighting is measured and reported against the guards below; adopting one
    is a separate decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- the roles under test -----------------------------------------------------

#: Analysed separately, never pooled into a single "defender" estimate. They
#: differ in the variable under test -- CB carries `team_defence` at 0.30, FB at
#: 0.0 -- which is what makes the contrast informative at all.
ROLES = ("CB", "FB")

# --- the predictor ------------------------------------------------------------

#: `performance_rating`, never `rating`. `rating` is a 50/50 blend with
#: `fifa_overall`, so scoring it against EA's *next* edition would be largely EA
#: predicting itself. This is the number `rating.py` computes before
#: `_blend_with_fifa`.
PREDICTOR = "performance_rating"

#: Projected out of the predictor for the decomposition. Exactly the three club
#: aggregates that constitute the `team_defence` sub-rating in the config --
#: adding more would project out legitimate player signal that merely correlates
#: with club quality, and the question is about *these* inputs.
PROJECTED_COLUMNS = (
    "team_xga_per_match",
    "team_ga_per_match",
    "team_clean_sheet_pct",
)

# --- the outcome --------------------------------------------------------------

#: EA's scale drifts: mean overall runs 73.41 in the 1415 edition to 75.98 in
#: 1920, with the sd between 5.14 and 5.60. A raw difference of overalls would
#: pool that drift with the player's actual move, so each edition is
#: standardised over its own full worldwide export first.
OUTCOME_SCALES = ("zscore", "rank")

#: Excluded as an *outcome* edition: it carries 504 big-five players against
#: ~1,900 in a complete edition and is still being populated.
INCOMPLETE_EDITIONS = ("2627",)

#: EA ages are integers and a player is one year older per edition; ±1.5 admits
#: the rounding without admitting a different footballer. Measured on the 6,209
#: departed player-seasons this recovers 77.9% uniquely with 3 ambiguous.
AGE_TOLERANCE = 1.5

#: Above this, a defender can only move down, so the thin ceiling (per-edition
#: max 91-94, under 1% at 90+) distorts the move. Sensitivity only; the headline
#: controls for the level instead.
TOP_BAND = 88

#: Where the unrecovered share is 3.7-6.8% rather than 25.5%.
AGE_CAP = 30

# --- the bar, fixed before the run --------------------------------------------

#: Equivalence bounds on the travel ratio: a **declared practical threshold**,
#: not a derivation. Nothing implies a 30% composite weight yields a 30% slope
#: attenuation. What the weight establishes is the order of magnitude -- if club
#: contamination runs through `team_defence` at all, the attenuation is
#: plausibly tens of percent rather than single digits -- so 15% marks roughly
#: half that scale, below which an effect is too small to attribute to the
#: mechanism or to justify a weight change. 1.18 is 1/0.85, so the interval is
#: symmetric in log-ratio.
EQUIVALENCE = (0.85, 1.18)

#: The same decision re-run at a looser and a stricter threshold. If the verdict
#: flips across these, it depends on the bound rather than the data, and that is
#: reported as inconclusive rather than resolved.
ALTERNATIVE_BOUNDS = ((0.80, 1.25), (0.90, 1.11))

#: Above this share of bootstrap replicates flipping the sign of the stayer
#: slope, the ratio has a denominator indistinguishable from zero and is
#: reported as undefined rather than summarised.
SIGN_FLIP_MAX = 0.05

#: Resamples draw *players*, not rows: one player contributes up to 11
#: consecutive-season pairs, and a row-level bootstrap would understate every
#: interval. `player_rating.yaml` makes the same point for the FB sweep -- "the
#: paired bootstrap's effective sample is nearer 12 than 20,060".
BOOTSTRAP_RESAMPLES = 10_000
SEED = 7

# --- the guards a candidate reweighting must clear ----------------------------
#
# Identical to the positional-quota experiment's, so the two are directly
# comparable. Production weights are not changed here under any outcome.

#: `def_vs_ga` may not worsen by more than this -- the whole 13-to-17 range of
#: `def>GA` from the squad-size sweep.
DEF_GAIN = 0.010

#: `ovr_vs_pts` and `atk_vs_gf` may not worsen by more than this.
OTHER_TOLERANCE = 0.005

#: Walk-forward RPS upper CI bound. The measured floor of this harness.
MARGIN = 0.0005


def standardise_within(
    frame: pd.DataFrame, column: str, by: list[str], scale: str = "zscore"
) -> pd.Series:
    """Standardise ``column`` within each ``by`` group.

    Both the raw and the residualised predictor go through this before entering
    any model. Residualising necessarily shrinks variance, and a
    smaller-variance predictor yields a smaller coefficient *for that reason
    alone* -- so comparing an unscaled raw slope against an unscaled residual
    slope would manufacture the finding the experiment is looking for.

    Args:
        frame: Source frame.
        column: Column to standardise.
        by: Grouping keys, normally ``["role", "Season"]``.
        scale: ``"zscore"`` for mean 0 / sd 1, or ``"rank"`` for a within-group
            percentile. The rank form respects the bounded, ordinal nature of an
            EA rating and is immune to compression at the top.

    Returns:
        The standardised column, aligned to ``frame``'s index.
    """
    if scale not in OUTCOME_SCALES:
        raise ValueError(f"scale must be one of {OUTCOME_SCALES}, got {scale!r}")
    grouped = frame.groupby(by, observed=True)[column]
    if scale == "rank":
        return grouped.rank(pct=True)
    mean = grouped.transform("mean")
    sd = grouped.transform("std")
    # A stratum with one member has no spread; it contributes no information and
    # must not become an infinity.
    return (frame[column] - mean) / sd.replace(0.0, np.nan)


def residualise(
    frame: pd.DataFrame,
    column: str = PREDICTOR,
    projected: tuple[str, ...] = PROJECTED_COLUMNS,
    by: tuple[str, ...] = ("role", "Season"),
) -> tuple[pd.Series, pd.DataFrame]:
    """Project the club defensive aggregates out of the predictor.

    Fitted **within each (role, season) stratum on the full rated population**,
    not on the analysis sample. That superset is deliberate: it uses only
    season-`t` information and does not condition on a player having an EA
    outcome at `t+1`, so which rows survive into the regression cannot change
    what the residual means.

    No outcome of any kind enters this, so there is no path by which it leaks.

    Returns:
        The residual, and a per-stratum frame carrying ``r_squared`` -- which is
        the quantity saying how much of the rating these three club numbers
        account for at all. Where it is near zero the decomposition has nothing
        to remove, and a residual slope equal to the raw one is uninformative
        rather than reassuring.
    """
    missing = [c for c in (column, *projected) if c not in frame.columns]
    if missing:
        raise KeyError(f"residualise needs {missing}, which the frame does not carry")

    residual = pd.Series(np.nan, index=frame.index, dtype="float64")
    diagnostics: list[dict[str, object]] = []

    for key, group in frame.groupby(list(by), observed=True):
        usable = group.dropna(subset=[column, *projected])
        if len(usable) <= len(projected) + 1:
            continue
        x = np.column_stack(
            [np.ones(len(usable)), *(usable[c].to_numpy() for c in projected)]
        )
        y = usable[column].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residual.loc[usable.index] = y - fitted
        total = float(((y - y.mean()) ** 2).sum())
        diagnostics.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                "n": len(usable),
                "r_squared": float(1.0 - ((y - fitted) ** 2).sum() / total)
                if total > 0
                else np.nan,
            }
        )

    return residual, pd.DataFrame(diagnostics)


def travel_ratio(stayer: float, mover_delta: float) -> float:
    """``(beta + delta) / beta`` -- reported, never the decision variable.

    `METHODOLOGY.md` records the defender EA-move correlation at 0.04-0.06, so
    the denominator is expected to be small and the ratio correspondingly
    unstable. The decision rests on ``delta``, which is a difference and is
    stable however small ``beta`` is.
    """
    if stayer == 0.0:
        return float("nan")
    return (stayer + mover_delta) / stayer


def ratio_is_interpretable(stayer_draws: np.ndarray) -> bool:
    """False when the bootstrap flips the denominator's sign too often.

    A ratio whose denominator straddles zero has no stable interpretation and no
    meaningful interval; reporting one anyway is how a near-zero slope becomes a
    dramatic-looking attenuation.
    """
    draws = np.asarray(stayer_draws, dtype="float64")
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return False
    minority = min((draws > 0).mean(), (draws < 0).mean())
    return bool(minority <= SIGN_FLIP_MAX)


def verdict(
    delta_ci: tuple[float, float],
    ratio: float,
    ratio_ci: tuple[float, float],
    stayer_ci: tuple[float, float],
    bounds: tuple[float, float] = EQUIVALENCE,
) -> str:
    """The three-way decision, computed from the declared constants.

    A confidence interval containing 1.0 is **not** evidence of no effect, which
    is why ``"equivalent"`` requires the interval to sit inside the bounds rather
    than merely to cover them.

    ``"uninformative"`` fires first: the attenuation of a slope indistinguishable
    from zero is undefined, and given the documented 0.04-0.06 baseline that is a
    realistic outcome. Pre-declaring it is what stops it being rationalised after
    the fact.
    """
    low, high = stayer_ci
    if low <= 0.0 <= high:
        return "uninformative"

    d_low, d_high = delta_ci
    if not (d_low <= 0.0 <= d_high) and ratio < bounds[0]:
        return "contamination"

    r_low, r_high = ratio_ci
    if bounds[0] <= r_low and r_high <= bounds[1]:
        return "equivalent"

    return "inconclusive"


# --- building the sample ------------------------------------------------------

RATINGS = "data/processed/player_ratings.csv"
EDITIONS = "data/raw/fifa/fifa_{season}.csv"


def _next_season(label: str) -> str:
    return f"{int(label[:2]) + 1:02d}{int(label[2:]) + 1:02d}"


def edition_stats(season: str) -> tuple[float, float, pd.Series]:
    """Mean, sd and the sorted overalls of one EA edition's **full worldwide**
    export.

    Standardising against the whole export rather than the big-five subset is
    deliberate: the drift being removed is EA's, and EA sets its scale across
    every league it ships.
    """
    frame = pd.read_csv(EDITIONS.format(season=season), low_memory=False)
    overall = frame["overall"].dropna().astype("float64")
    return float(overall.mean()), float(overall.std()), overall.sort_values()


def recover_outcomes(
    rated: pd.DataFrame, lookup: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Attach each player's EA rating and club in edition ``t+1``.

    Two tiers, and the second is the one that matters. Requiring a *rated*
    season at ``t+1`` would condition on the player clearing the 450-minute gate
    at his new club -- post-transfer information, selecting for successful
    transfers, which is the thing under test. EA's exports are the whole
    worldwide database, so the outcome is recoverable without that condition.

    Tier 2 matches on normalised name with ``age`` within ``AGE_TOLERANCE`` of
    ``age_t + 1``, and **requires the match to be unique**. Age consistency is a
    strong disambiguator EA supplies for free: measured over the departed
    player-seasons it leaves 3 ambiguous cases in 6,209.
    """
    from footballml.players.fifa import normalise_name

    rated = rated.copy()
    rated["next_label"] = rated["Season"].map(_next_season)

    # Tier 1: the production matcher already placed him in t+1.
    #
    # Built from `lookup` -- every rated player-season -- and *not* from the
    # eligible rows. A season can be ineligible as a predictor while remaining
    # perfectly good as an outcome, and conflating the two destroyed a whole
    # fold: 2526 is dropped as a predictor because its own t+1 edition is still
    # filling, and when the lookup was built from the filtered frame the
    # 2425 -> 2526 pairs lost tier 1 entirely. That fold recovered 36.2%
    # against 94-96% everywhere else.
    source = rated if lookup is None else lookup
    keyed = (
        source.dropna(subset=["fifa_overall"])
        .drop_duplicates(subset=["Player", "Season"])
        .set_index(["Player", "Season"])
    )
    pairs = pd.MultiIndex.from_arrays([rated["Player"], rated["next_label"]])
    rated["next_overall"] = keyed["fifa_overall"].reindex(pairs).to_numpy()
    rated["next_club"] = keyed["fifa_club"].reindex(pairs).to_numpy()
    rated["tier"] = np.where(rated["next_overall"].notna(), 1, 0)

    # Tier 2: the raw edition, for everyone tier 1 could not place.
    for season, group in rated[rated["tier"] == 0].groupby("next_label"):
        try:
            edition = pd.read_csv(EDITIONS.format(season=season), low_memory=False)
        except FileNotFoundError:
            continue
        edition["_n"] = edition["name"].astype(str).map(normalise_name)
        edition = edition.dropna(subset=["_n"])
        groups = edition.groupby("_n").groups
        for idx, name, age in zip(
            group.index, group["fifa_name"], group["fifa_age"], strict=True
        ):
            key = normalise_name(str(name))
            if key not in groups or pd.isna(age):
                continue
            candidates = edition.loc[groups[key]]
            candidates = candidates[(candidates["age"] - age).abs() <= AGE_TOLERANCE]
            if len(candidates) != 1:
                continue
            rated.loc[idx, "next_overall"] = float(candidates["overall"].iloc[0])
            rated.loc[idx, "next_club"] = str(candidates["club"].iloc[0])
            rated.loc[idx, "tier"] = 2

    rated["recovered"] = rated["next_overall"].notna()
    # Mover status is defined within EA's own club naming, so no alias map is
    # needed and a departure outside the big five classifies correctly.
    rated["mover"] = np.where(
        rated["recovered"],
        rated["fifa_club"].astype(str) != rated["next_club"].astype(str),
        np.nan,
    )
    return rated


def build_sample() -> pd.DataFrame:
    """Every rated player-season eligible for the EA-move analysis.

    Excludes mid-season transfers (two club rows in one player-season), which
    are a different treatment, and any season whose ``t+1`` edition is still
    being populated.
    """
    players = pd.read_csv(RATINGS, low_memory=False)
    players["Season"] = players["Season"].astype(str)
    rated = players[players["rated"]].copy()

    clubs_per_season = rated.groupby(["Player", "Season"])["Team"].transform("size")
    rated = rated[clubs_per_season == 1]

    rated = rated[
        rated["fifa_overall"].notna()
        & rated["fifa_age"].notna()
        & rated["fifa_club"].notna()
    ]
    # The lookup keeps every rated season; only the *eligible* rows are filtered.
    lookup = rated
    eligible = rated[~rated["Season"].map(_next_season).isin(INCOMPLETE_EDITIONS)]
    return recover_outcomes(eligible, lookup)


# --- estimation ---------------------------------------------------------------

CONTROLS = ("fifa_age", "age_squared", "minutes", "measured_share", "z_prior")


def prepare(
    sample: pd.DataFrame, scale: str = "zscore", include_unrecovered: bool = False
) -> pd.DataFrame:
    """Attach the outcome, the standardised predictors and the controls.

    Everything on the right-hand side is season-`t` information. The only `t+1`
    quantities that enter are the outcome itself and the mover indicator, which
    is the treatment definition rather than a conditioning variable.
    """
    # Standardisation runs over the full eligible set, so that the unrecovered
    # rows the missingness grid needs carry comparable predictors. Which rows
    # are *analysed* is decided at the end, not here.
    frame = sample.copy()
    frame["mover"] = frame["mover"].astype(float)

    stats = {}
    for season in set(frame["Season"]) | set(frame["Season"].map(_next_season)):
        try:
            stats[season] = edition_stats(season)
        except FileNotFoundError:
            continue

    def standardise(values: pd.Series, seasons: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=values.index, dtype="float64")
        for season, idx in seasons.groupby(seasons).groups.items():
            if season not in stats:
                continue
            mean, sd, sorted_overall = stats[season]
            if scale == "rank":
                out.loc[idx] = np.searchsorted(
                    sorted_overall.to_numpy(), values.loc[idx].to_numpy()
                ) / len(sorted_overall)
            else:
                out.loc[idx] = (values.loc[idx] - mean) / sd
        return out

    frame["z_prior"] = standardise(frame["fifa_overall"], frame["Season"])
    frame["z_next"] = standardise(frame["next_overall"], frame["Season"].map(_next_season))
    frame["outcome"] = frame["z_next"] - frame["z_prior"]

    frame["z_perf"] = standardise_within(frame, PREDICTOR, ["role", "Season"])
    residual, diagnostics = residualise(frame)
    frame["residual"] = residual
    frame["z_perf_res"] = standardise_within(frame, "residual", ["role", "Season"])
    frame.attrs["projection"] = diagnostics

    frame["age_squared"] = frame["fifa_age"] ** 2
    frame = frame.dropna(subset=["z_perf", *CONTROLS])
    if include_unrecovered:
        return frame
    return frame.dropna(subset=["outcome", "mover"])


def _design(frame: pd.DataFrame, predictor: str, extra: list[str] | None = None):
    """Design matrix: predictor, mover, their interaction, controls and FE."""
    pieces = [
        np.ones((len(frame), 1)),
        frame[[predictor]].to_numpy(),
        frame[["mover"]].to_numpy(),
        (frame[predictor] * frame["mover"]).to_numpy()[:, None],
        frame[list(CONTROLS)].to_numpy(),
    ]
    names = ["const", "perf", "mover", "perf_x_mover", *CONTROLS]
    for column in extra or []:
        pieces.append(frame[[column]].to_numpy())
        names.append(column)
    for key in ("Season", "League"):
        dummies = pd.get_dummies(frame[key], prefix=key, drop_first=True, dtype=float)
        pieces.append(dummies.to_numpy())
        names += list(dummies.columns)
    return np.hstack(pieces), names


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def fit(frame: pd.DataFrame, predictor: str = "z_perf") -> dict[str, float]:
    """Stayer slope, mover slope and their difference, in one model."""
    x, names = _design(frame, predictor)
    beta = _ols(x, frame["outcome"].to_numpy())
    stayer = float(beta[names.index("perf")])
    delta = float(beta[names.index("perf_x_mover")])
    return {"stayer": stayer, "delta": delta, "mover": stayer + delta}


def cluster_bootstrap(
    frame: pd.DataFrame,
    predictor: str = "z_perf",
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Resample **players**, not rows.

    A player contributes up to 11 consecutive-season pairs. A row-level
    bootstrap would treat those as independent and understate every interval --
    the same effective-sample point `player_rating.yaml` makes for the FB sweep.
    """
    rng = np.random.default_rng(seed)
    blocks = {p: g.index.to_numpy() for p, g in frame.groupby("Player", observed=True)}
    players = np.array(list(blocks), dtype=object)
    draws = []
    for _ in range(resamples):
        picked = rng.integers(0, len(players), size=len(players))
        rows = np.concatenate([blocks[players[i]] for i in picked])
        try:
            draws.append(fit(frame.loc[rows], predictor))
        except np.linalg.LinAlgError:  # pragma: no cover - singular resample
            continue
    return pd.DataFrame(draws)


def fieller(draws: pd.DataFrame, alpha: float = 0.05) -> tuple[float, float]:
    """Fieller interval for ``mover / stayer``, the correct construction for a
    ratio of normals.

    A percentile interval on bootstrapped ratios is not valid when the
    denominator approaches zero: the ratio's distribution is heavy-tailed and
    its quantiles are not the quantiles of a ratio of means. Fieller returns an
    unbounded interval in that case rather than a tidy wrong one.
    """
    from scipy import stats as sps

    a, b = draws["mover"].to_numpy(), draws["stayer"].to_numpy()
    va, vb, cov = a.var(ddof=1), b.var(ddof=1), np.cov(a, b)[0, 1]
    a_hat, b_hat = a.mean(), b.mean()
    t = sps.norm.ppf(1 - alpha / 2)

    g = t**2 * vb / b_hat**2
    if g >= 1.0:
        return (-np.inf, np.inf)
    ratio = a_hat / b_hat
    centre = ratio - g * cov / vb
    under = va - 2 * ratio * cov + ratio**2 * vb - g * (va - cov**2 / vb)
    if under < 0:
        return (-np.inf, np.inf)
    half = (t / abs(b_hat)) * np.sqrt(under)
    return (float((centre - half) / (1 - g)), float((centre + half) / (1 - g)))


def bootstrap_fast(
    frame: pd.DataFrame,
    predictor: str = "z_perf",
    extra: list[str] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """`cluster_bootstrap`, with the design matrix built once.

    Identical in what it estimates; `get_dummies` inside the resample loop made
    the honest version take hours. The slow path is kept because it is the
    readable definition and the tests check the two agree.
    """
    x, names = _design(frame, predictor, extra)
    y = frame["outcome"].to_numpy()
    w = np.ones(len(frame)) if weights is None else np.asarray(weights, dtype="float64")
    i_perf, i_int = names.index("perf"), names.index("perf_x_mover")
    three_way = "is_CB_x_perf_x_mover"
    i_cb = names.index(three_way) if three_way in names else None

    positions = {
        player: np.flatnonzero(frame.index.isin(rows))
        for player, rows in frame.groupby("Player", observed=True).groups.items()
    }
    players = np.array(list(positions), dtype=object)
    rng = np.random.default_rng(seed)

    draws = []
    for _ in range(resamples):
        picked = rng.integers(0, len(players), size=len(players))
        rows = np.concatenate([positions[players[i]] for i in picked])
        xb, yb, wb = x[rows], y[rows], w[rows]
        root = np.sqrt(wb)[:, None]
        try:
            beta = _ols(xb * root, yb * np.sqrt(wb))
        except np.linalg.LinAlgError:  # pragma: no cover - singular resample
            continue
        row = {
            "stayer": beta[i_perf],
            "delta": beta[i_int],
            "mover": beta[i_perf] + beta[i_int],
        }
        if i_cb is not None:
            row["delta_difference"] = beta[i_cb]
        draws.append(row)
    return pd.DataFrame(draws)


def point_fit(
    frame: pd.DataFrame,
    predictor: str = "z_perf",
    extra: list[str] | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    x, names = _design(frame, predictor, extra)
    y = frame["outcome"].to_numpy()
    w = np.ones(len(frame)) if weights is None else np.asarray(weights, dtype="float64")
    root = np.sqrt(w)
    beta = _ols(x * root[:, None], y * root)
    out = {
        "stayer": float(beta[names.index("perf")]),
        "delta": float(beta[names.index("perf_x_mover")]),
    }
    out["mover"] = out["stayer"] + out["delta"]
    if extra and "is_CB_x_perf_x_mover" in names:
        out["delta_difference"] = float(beta[names.index("is_CB_x_perf_x_mover")])
    return out


def ci(draws: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(draws.dropna(), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
