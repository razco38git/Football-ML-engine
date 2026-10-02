"""Score upcoming fixtures and settle finished ones.

The job that builds the track record. Intended to run on a schedule:

1. Predict every fixture scheduled within the next ``--horizon-days`` and store
   it, stamped with the model version and the moment it was made.
2. Attach results to any stored prediction whose match has since been played.

Step 1 must happen *before* kickoff for the record to mean anything, which is
why this runs on a timer rather than on demand.

Fixtures come from :mod:`footballml.ingest.schedule` -- the whole season's
schedule -- rather than football-data's rolling file of "upcoming" matches.
That file is a snapshot, and on a Monday it usually still holds the round just
played, so this job would fetch it, correctly skip every already-played
fixture, store nothing, and exit zero. The result was a live record of
eighteen predictions that were all Spanish: coverage depended on whether the
feed happened to be ahead of the job for a given league. football-data is still
read, but only for the bookmaker odds the accuracy page benchmarks against.

Run with::

    python -m pipelines.score_upcoming
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml import registry, store  # noqa: E402
from footballml.data import (  # noqa: E402
    ODDS_COLUMNS,
    PROCESSED_DIR,
    load_team_match_history,
    load_team_strength,
)
from footballml.features.build import build_match_features, build_upcoming_features  # noqa: E402
from footballml.ingest.european import SHOWN_COMPETITIONS
from footballml.ingest.matchhistory import LEAGUES, fetch_fixtures  # noqa: E402
from footballml.ingest.schedule import (  # noqa: E402
    COMPETITION_HORIZON_DAYS,
    DEFAULT_HORIZON_DAYS,
    fetch_schedule,
    window_fixtures,
)

OUTPUT_COLUMNS = [
    "League", "competition", "Date", "HomeTeam", "AwayTeam",
    "expected_goals_home", "expected_goals_away",
    "prob_home_win", "prob_draw", "prob_away_win", "predicted_outcome",
    "modal_score_home", "modal_score_away", "prob_over_2_5", "prob_btts",
]


def _european_results() -> pd.DataFrame:
    """Played UEFA ties, shaped so `store.settle` can match them.

    `League` is the home side's division and `competition` the competition, the
    same convention the predictions were stored under.
    """
    path = PROCESSED_DIR / "european_matches.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "League": frame["home_league"],
            "competition": frame["League"],
            "Date": pd.to_datetime(frame["Date"]),
            "HomeTeam": frame["HomeTeam"],
            "AwayTeam": frame["AwayTeam"],
            "FTHG": frame["FTHG"],
            "FTAG": frame["FTAG"],
            "FTR": frame["FTR"],
        }
    )


def _european_fixtures(leagues: list[str] | None, log: logging.Logger) -> pd.DataFrame:
    """Upcoming UEFA ties we can actually rate.

    Only matches where **both** clubs played in one of the five leagues this
    season: everyone else -- Benfica, Ajax, Celtic, Galatasaray -- has no form
    history, no xG and no squad rating here, so there is nothing to predict
    from. That keeps about 41% of a Champions League round.

    `League` is set to the **home side's domestic division**, not the
    competition. `LEAGUE_CODES` has no UEFA entry, so a row labelled "UCL" would
    give the model a NaN `league_code` -- a value no training row ever carried,
    which it would accept in silence. The competition is carried separately.
    """
    wanted = [c for c in (leagues or SHOWN_COMPETITIONS) if c in SHOWN_COMPETITIONS]
    if not wanted:
        return pd.DataFrame()

    from footballml.ingest.european import big_five_only

    try:
        schedule = fetch_schedule(wanted)
        schedule = window_fixtures(schedule, horizon_days=COMPETITION_HORIZON_DAYS)
    except Exception as exc:  # noqa: BLE001 - one missing source is survivable
        log.warning("No European schedule (%s)", exc)
        return pd.DataFrame()
    if schedule.empty:
        return pd.DataFrame()

    tmh = load_team_match_history(PROCESSED_DIR / "team_match_history_all.csv")
    # `big_five_only` wants the wide match shape and fills the two division
    # columns; the schedule has no scores, which is exactly what we want here.
    kept, dropped = big_five_only(schedule.assign(FTHG=pd.NA, FTAG=pd.NA, FTR=pd.NA), tmh)
    log.info(
        "European: %d of %d fixtures have both clubs in the big five",
        len(kept), len(schedule),
    )
    if kept.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "League": kept["home_league"],
            "competition": kept["League"],
            "Season": kept["Season"],
            "Date": pd.to_datetime(kept["Date"]),
            "HomeTeam": kept["HomeTeam"],
            "AwayTeam": kept["AwayTeam"],
            "played": kept.get("played", False),
        }
    )


def _collect_fixtures(
    leagues: list[str] | None, horizon_days: int, log: logging.Logger
) -> pd.DataFrame:
    """Fixtures to predict, with bookmaker odds attached where published.

    The schedule decides *which* fixtures exist -- it carries the whole season,
    so coverage no longer depends on what football-data's rolling file happens
    to hold on the morning the job runs. football-data is still consulted, but
    only for the odds, which are the one thing FBref does not provide and which
    the accuracy page uses as its benchmark.

    Falls back to football-data alone if the schedule is unavailable, so an
    outage at one source degrades coverage instead of stopping the record.
    """
    schedule = fetch_schedule(leagues)
    schedule = pd.concat([schedule, _european_fixtures(leagues, log)], ignore_index=True)
    if schedule.empty:
        # The schedule source itself gave us nothing, which is a problem rather
        # than an answer. football-data's rolling file is the fallback.
        log.warning("No schedule available; falling back to the published fixture file")
        return fetch_fixtures(leagues)

    fixtures = window_fixtures(schedule, horizon_days=horizon_days)
    if fixtures.empty:
        # Nothing is scheduled. Deliberately *not* falling back: the published
        # file routinely holds the round that has just been played, and treating
        # that as "upcoming" is what produced a track record of eighteen
        # predictions from a single league.
        pending = schedule[~schedule["played"]]
        if pending.empty:
            log.info("Nothing left to predict; every fixture has been played")
        else:
            nxt = pending["Date"].min()
            log.info(
                "Nothing to predict: next fixture is %s, %d day(s) past the "
                "%d-day horizon (an international break, most likely)",
                nxt.date(),
                (nxt - pd.Timestamp.today().normalize()).days - horizon_days,
                horizon_days,
            )
        return fixtures

    try:
        published = fetch_fixtures(leagues)
    except Exception as exc:  # noqa: BLE001 - odds are optional, fixtures are not
        log.warning("Could not fetch odds (%s); predicting without them", exc)
        return fixtures

    odds_cols = [c for c in ODDS_COLUMNS if c in published.columns]
    if published.empty or not odds_cols:
        log.info("No odds published for these fixtures")
        return fixtures

    key = ["League", "Date", "HomeTeam", "AwayTeam"]
    merged = fixtures.merge(
        published[[*key, *odds_cols]].assign(Date=pd.to_datetime(published["Date"])),
        on=key,
        how="left",
    )
    with_odds = int(merged[odds_cols[0]].notna().sum())
    log.info("%d of %d fixture(s) have odds", with_odds, len(merged))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leagues", nargs="+", choices=sorted([*LEAGUES, *SHOWN_COMPETITIONS])
    )
    parser.add_argument("--settle-only", action="store_true")
    parser.add_argument(
        "--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS,
        help="How far ahead to predict. See ingest.schedule for why this exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("score_upcoming")

    tmh = load_team_match_history(PROCESSED_DIR / "team_match_history_all.csv")
    strength = load_team_strength()

    # Settle first: a fixture predicted last week may have been played since.
    played = build_match_features(tmh, strength=strength)
    results = played[played["FTR"].notna()].assign(competition=store.DOMESTIC)
    # European results live in their own file -- they are deliberately kept out
    # of the match history so they never reach a rolling window or the Elo walk
    # as *history*. Without them a stored UCL prediction never settles, and sits
    # in the record forever with null results.
    results = pd.concat([results, _european_results()], ignore_index=True)
    settled = store.settle(results)
    log.info("Settled %d previously stored predictions", settled)

    if args.settle_only:
        return

    model, metadata = registry.load()
    log.info("Using model %s", metadata.version)

    # `_collect_fixtures` has already said why this is empty -- it is the only
    # place that can tell "nothing scheduled" from "no schedule".
    fixtures = _collect_fixtures(args.leagues, args.horizon_days, log)
    if fixtures.empty:
        return

    # Without strength here the artifact's feature list cannot be satisfied, and
    # this is the path that writes the live pre-kickoff track record.
    scored = build_upcoming_features(tmh, fixtures, strength=strength)
    if scored.empty:
        log.warning("No fixtures could be matched to known teams")
        return

    preds = model.predict_frame(scored[metadata.feature_names])
    frame = pd.concat([scored.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    odds_cols = [c for c in ODDS_COLUMNS if c in fixtures.columns]
    columns = [*OUTPUT_COLUMNS]
    if odds_cols:
        key = ["League", "Date", "HomeTeam", "AwayTeam"]
        frame = frame.merge(
            fixtures[[*key, *odds_cols]].assign(Date=pd.to_datetime(fixtures["Date"])),
            on=key,
            how="left",
        )
        columns += odds_cols

    added = store.append(frame[columns], metadata.version)
    log.info("Stored %d new predictions", added)

    total = len(store.load())
    done = len(store.settled())
    print(f"\nPrediction store: {total} total, {done} settled, {total - done} awaiting results")


if __name__ == "__main__":
    main()
