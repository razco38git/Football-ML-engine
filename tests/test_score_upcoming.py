"""What the weekly job must still have by the time it writes to the store.

On 2026-10-04 `score_upcoming` died with ``KeyError: "['competition'] not in
index"`` on the first run that had Champions League ties in it. The column was
attached by `_collect_fixtures`, stripped by `build_upcoming_features` -- which
rebuilds every column from the match history, and the history has no notion of
a competition -- and then demanded by `OUTPUT_COLUMNS` when the store was
written.

It had been unreachable rather than correct: until UEFA registration was fixed
earlier the same day, `_european_fixtures` returned nothing, so there was never
a `competition` to lose. The failure arrived with the fixtures.
"""

from __future__ import annotations

import pandas as pd

from footballml import store
from pipelines.score_upcoming import OUTPUT_COLUMNS, _restore_fixture_columns

KEY = ["League", "Date", "HomeTeam", "AwayTeam"]


def _scored(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """A frame shaped like `build_upcoming_features` output: no competition."""
    frame = pd.DataFrame(
        [{"League": lg, "Date": pd.Timestamp(d), "HomeTeam": h, "AwayTeam": a}
         for lg, d, h, a in rows]
    )
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns and column != "competition":
            frame[column] = 0.0
    return frame


def _fixtures(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"League": lg, "Date": pd.Timestamp(d), "HomeTeam": h, "AwayTeam": a,
          "competition": comp}
         for lg, d, h, a, comp in rows]
    )


def test_every_output_column_survives_to_the_store():
    """The contract the KeyError broke: `frame[columns]` must not raise."""
    scored = _scored([("E0", "2026-10-10", "Arsenal", "Chelsea")])
    fixtures = _fixtures([("E0", "2026-10-10", "Arsenal", "Chelsea", store.DOMESTIC)])

    frame, columns = _restore_fixture_columns(scored, fixtures)

    assert set(OUTPUT_COLUMNS) <= set(frame.columns)
    frame[columns]  # the exact expression that raised


def test_a_european_tie_keeps_its_competition():
    """`League` is the home side's division and `competition` is the cup.

    Losing it would file a Champions League tie as a Premier League match, and
    the accuracy page splits its record on exactly this column.
    """
    scored = _scored([("E0", "2026-10-22", "Arsenal", "Real Madrid")])
    fixtures = _fixtures([("E0", "2026-10-22", "Arsenal", "Real Madrid", "UCL")])

    frame, _ = _restore_fixture_columns(scored, fixtures)

    assert frame["competition"].tolist() == ["UCL"]


def test_a_domestic_fixture_is_labelled_not_left_blank():
    """The store's convention is an explicit label. A NaN would read as unknown
    rather than as a league match, and it is a league match."""
    scored = _scored([("SP1", "2026-10-11", "Barcelona", "Sevilla")])
    fixtures = _fixtures([("SP1", "2026-10-11", "Barcelona", "Sevilla", None)])

    frame, _ = _restore_fixture_columns(scored, fixtures)

    assert frame["competition"].tolist() == [store.DOMESTIC]


def test_fixtures_with_no_competition_column_still_work():
    """The football-data fallback path carries no competition at all. It is a
    degraded source, not a broken one, and the job has to survive it."""
    scored = _scored([("I1", "2026-10-11", "Inter", "Milan")])
    fixtures = _fixtures([("I1", "2026-10-11", "Inter", "Milan", None)]).drop(
        columns=["competition"]
    )

    frame, columns = _restore_fixture_columns(scored, fixtures)

    assert frame["competition"].tolist() == [store.DOMESTIC]
    frame[columns]


def test_odds_are_carried_when_published_and_not_demanded_when_not():
    """Odds exist for domestic fixtures and never for European ones, so they
    are appended to the column list only when they are actually there."""
    scored = _scored([("E0", "2026-10-10", "Arsenal", "Chelsea")])
    fixtures = _fixtures([("E0", "2026-10-10", "Arsenal", "Chelsea", store.DOMESTIC)])
    fixtures["B365H"] = 1.8

    frame, columns = _restore_fixture_columns(scored, fixtures)

    assert "B365H" in columns
    assert frame["B365H"].tolist() == [1.8]

    bare, bare_columns = _restore_fixture_columns(
        _scored([("E0", "2026-10-10", "Arsenal", "Chelsea")]),
        _fixtures([("E0", "2026-10-10", "Arsenal", "Chelsea", "UCL")]),
    )
    assert "B365H" not in bare_columns
    bare[bare_columns]
