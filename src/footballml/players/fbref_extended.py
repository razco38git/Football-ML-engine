"""FBref's full stat tables: defending, possession and passing.

.. warning::
    **This currently returns almost nothing, and that is FBref's doing, not a
    bug here.** The pages are fetched and parsed correctly, but FBref serves
    these tables with the values stripped: the column headers arrive, and the
    cells come back as ``data-stat="tackles"></td>``. Verified empty across
    2024/25, 2025/26 and 2026/27, so it is not a data-lag issue.

    Of roughly forty metrics, only ``interceptions``, ``tackles_won`` and
    ``assists`` carry values -- and the ``misc`` table already supplies those.
    Tackles, blocks, clearances, errors, every carry, every touch and the whole
    passing table are empty.

    The module is kept because it is correct and cheap to re-run: if FBref
    starts serving the values, the categories below light up with no further
    work. :func:`fetch_extended_stats` drops the empty columns loudly rather
    than passing a wall of NaN downstream, where it would silently dilute every
    sub-rating it touched.

``soccerdata`` exposes only five of FBref's player tables (standard, shooting,
playing_time, keeper, misc). That restriction is a hardcoded list, not a
technical limit -- the library's URL builder derives straight from the stat-type
string -- so this module reuses soccerdata's browser-backed session to reach the
rest. Direct requests to FBref return 403, so that session is the only way in.

Parsing uses ``pandas.read_html`` rather than soccerdata's ``_parse_table``,
which returned all-null columns for several stats while populating others beside
them -- worse than an error, because the pipeline runs to completion on missing
data.

What these tables *would* unlock, against a professional scouting report's
categories:

===================  ====================================================
Category             Source
===================  ====================================================
Defending            tackles by third, tackle success, blocks, clearances
Progression          progressive carry and pass distance, final-third entries
Receiving            passes received, touches by zone
Dribbling            take-ons attempted and succeeded
Passing accuracy     completion by pass distance
Creation             key passes, passes into the penalty area, xAG
===================  ====================================================

Two things such reports contain cannot be reproduced at all: **VAEP** and
**xThreat**. Both are model outputs over full event streams -- every touch with
pitch coordinates -- which no free source publishes.
"""

from __future__ import annotations

import logging

import pandas as pd

from footballml.ingest.understat import UNDERSTAT_LEAGUES

logger = logging.getLogger(__name__)

#: Tables to fetch beyond the five soccerdata lists, and the columns worth
#: keeping from each. Keys are FBref's own column labels, which repeat across
#: groups -- ``_flatten_columns`` disambiguates them before this mapping applies.
EXTENDED_TABLES: dict[str, dict[str, str]] = {
    "defense": {
        "Tkl": "tackles",
        "TklW": "tackles_won",
        "Def 3rd": "tackles_def_third",
        "Mid 3rd": "tackles_mid_third",
        "Att 3rd": "tackles_att_third",
        "Tkl%": "dribbler_tackle_pct",
        "Blocks": "blocks",
        "Sh": "shots_blocked",
        "Pass": "passes_blocked",
        "Int": "interceptions",
        "Clr": "clearances",
        "Err": "errors",
    },
    # FBref's current schema has no PrgC/PrgP/PrgR columns -- progression is
    # published as *distance* instead, which is the better measure anyway: ten
    # metres carried forward counts for more than a two-metre pass that happens
    # to clear the threshold.
    "possession": {
        "Touches": "touches",
        "Def 3rd": "touches_def_third",
        "Att 3rd": "touches_att_third",
        "Att Pen": "touches_att_pen",
        "Att": "take_ons",
        "Succ": "take_ons_won",
        "Succ%": "take_on_success_pct",
        "Tkld%": "take_on_tackled_pct",
        "Carries": "carries",
        "PrgDist": "carry_progressive_distance",
        "1/3": "carries_into_final_third",
        "CPA": "carries_into_pen_area",
        "Mis": "miscontrols",
        "Dis": "dispossessed",
        "Rec": "passes_received",
    },
    "passing": {
        "Cmp%": "pass_completion_pct",
        "TotDist": "pass_distance",
        "PrgDist": "pass_progressive_distance",
        "Long_Att": "long_passes",
        "Long_Cmp%": "long_pass_completion_pct",
        "Ast": "fb_assists",
        # xAG is not published directly, only assists minus xAG. The difference
        # recovers it, and is derived after the fetch.
        "A-xAG": "assists_minus_xag",
        "KP": "fb_key_passes",
        "1/3": "passes_into_final_third",
        "PPA": "passes_into_pen_area",
        "CrsPA": "crosses_into_pen_area",
    },
}


def fetch_extended_stats(
    leagues: list[str] | None = None, seasons: list[str] | None = None
) -> pd.DataFrame:
    """Fetch the defending, possession and passing tables.

    Slow: each table is a separate page load through headless Chrome, roughly
    fifteen seconds per league-season. Cached by soccerdata afterwards, so the
    cost is paid once.

    Returns:
        ``League``, ``Season``, ``Team``, ``Player`` plus per-90 rates for every
        mapped metric.
    """
    import soccerdata as sd
    from lxml import etree, html
    from soccerdata.fbref import FBREF_API, _parse_table

    from footballml.players.fbref import _flatten_columns

    codes = leagues or sorted(UNDERSTAT_LEAGUES)
    inverse = {v: k for k, v in UNDERSTAT_LEAGUES.items()}
    fb = sd.FBref(leagues=[UNDERSTAT_LEAGUES[c] for c in codes], seasons=seasons)
    season_index = fb.read_seasons()

    frames: list[pd.DataFrame] = []
    for stat_type, wanted in EXTENDED_TABLES.items():
        tables: list[pd.DataFrame] = []
        for (lkey, skey), season in season_index.iterrows():
            try:
                raw = _fetch_table(
                    fb, FBREF_API, season, lkey, skey, stat_type, html, etree, _parse_table
                )
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
                logger.warning("FBref %s %s %s failed: %s", stat_type, lkey, skey, exc)
                continue

            raw.columns = _flatten_columns(raw.columns)
            out = pd.DataFrame(
                {
                    "League": inverse.get(lkey, lkey),
                    "Season": str(skey),
                    "Team": raw.get("Squad"),
                    "Player": raw.get("Player"),
                }
            )
            nineties = pd.to_numeric(raw.get("90s"), errors="coerce")
            for source, name in wanted.items():
                if source not in raw.columns:
                    continue
                values = pd.to_numeric(raw[source], errors="coerce")
                if source.endswith("%"):
                    # Already a rate; dividing by minutes would be meaningless.
                    out[name] = values
                else:
                    out[f"{name}_per90"] = values.div(nineties).where(nineties > 0)
            tables.append(out)

        if tables:
            frames.append(pd.concat(tables, ignore_index=True))
            logger.info("FBref %s: %d rows", stat_type, len(frames[-1]))

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame, on=["League", "Season", "Team", "Player"], how="outer", suffixes=("", "_dup")
        )
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")])

    # FBref returns these tables with most values stripped: the column headers
    # are served but the cells come back as `data-stat="tackles"></td>`. Only
    # interceptions, tackles won and assists survive, and the misc table already
    # supplies those. Drop the empty columns loudly rather than passing a wall of
    # NaN downstream, where it silently dilutes every sub-rating it touches.
    keys = {"League", "Season", "Team", "Player"}
    empty = [c for c in merged.columns if c not in keys and merged[c].isna().all()]
    if empty:
        logger.warning(
            "FBref served %d of %d extended columns with no values; dropping them. "
            "Populated: %s",
            len(empty),
            len(merged.columns) - len(keys),
            sorted(c for c in merged.columns if c not in keys and c not in empty),
        )
        merged = merged.drop(columns=empty)

    return _derive(merged)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Recover metrics FBref publishes only as differences or totals."""
    out = df.copy()

    # xAG is published as assists minus xAG, so subtract back out.
    if {"fb_assists_per90", "assists_minus_xag_per90"} <= set(out.columns):
        out["expected_assisted_goals_per90"] = (
            out["fb_assists_per90"] - out["assists_minus_xag_per90"]
        )

    # Share of touches taken in the final third: a positional signal that
    # separates an overlapping full-back from one who stays home, without
    # needing a position label to say so.
    if {"touches_att_third_per90", "touches_per90"} <= set(out.columns):
        out["att_third_touch_share"] = (
            out["touches_att_third_per90"].div(out["touches_per90"]).where(out["touches_per90"] > 0)
        )

    # Ball security: losses per touch. Raw miscontrols punish players who see
    # more of the ball, which is backwards.
    losses = [c for c in ("miscontrols_per90", "dispossessed_per90") if c in out.columns]
    if losses and "touches_per90" in out.columns:
        out["losses_per_touch"] = (
            out[losses].sum(axis=1).div(out["touches_per90"]).where(out["touches_per90"] > 0)
        )

    return out


def _fetch_table(fb, api, season, lkey, skey, stat_type, html, etree, parse_table):
    """Load one FBref stat page through soccerdata's cached browser session.

    The page is fetched with soccerdata's session -- direct requests to FBref
    return 403 -- but parsed with ``pandas.read_html`` rather than soccerdata's
    own ``_parse_table``. That parser silently returns an all-null column for
    several stats on these tables: ``Tkl``, ``Blocks`` and ``Clr`` came back
    empty while ``Int`` beside them was fully populated, which is worse than an
    error because the pipeline runs to completion on missing data.
    """
    import io

    import pandas as pd

    big_five = lkey == "Big 5 European Leagues Combined"
    url = (
        api
        + "/".join(season.url.split("/")[:-1])
        + f"/{stat_type}"
        + ("/players/" if big_five else "/")
        + season.url.split("/")[-1]
    )
    path = fb.data_dir / f"players_{lkey}_{skey}_{stat_type}.html"
    tree = html.parse(fb.get(url, path))

    if big_five:
        (element,) = tree.xpath(f"//table[@id='stats_{stat_type}']")
        markup = etree.tostring(element, encoding="unicode")
    else:
        # FBref hides secondary tables inside HTML comments to defer rendering.
        (comment,) = tree.xpath(f"//comment()[contains(.,'div_stats_{stat_type}')]")
        markup = comment.text

    table = pd.read_html(io.StringIO(markup))[0]
    # FBref repeats the header row every 25 rows for readability.
    if "Player" in [c[1] if isinstance(c, tuple) else c for c in table.columns]:
        player_col = next(
            c for c in table.columns
            if (c[1] if isinstance(c, tuple) else c) == "Player"
        )
        table = table[table[player_col] != "Player"]
    return table
