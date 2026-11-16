# Should an attacking midfielder count as midfield?

**Status: proposal, measured. Nothing here is built.** Every number below comes
from `pipelines/cam_line_ab.py`, run against the 2026-11-02 `player_ratings.csv`
vintage — the last successful `build_players` before the Understat outage. The
measurement was made before the recommendation was written, and the
recommendation changed as a result: see
[What the measurement changed](#what-the-measurement-changed).

## The question

It arrived as an observation about one club: Real Madrid's published midfield
rating is **73.8**, for a midfield that contains Jude Bellingham.

That is not a bug, and the arithmetic is exactly right. It is a consequence of
a taxonomy decision made in `fifa.py`, and the question is whether that
decision is the one we want.

## Why the number is 73.8

`ROLE_BY_POSITION` in `src/footballml/players/fifa.py` collapses five EA
positions into one role:

```
CAM, LM, RM, LW, RW  ->  AMW   ("Attacking midfielder / winger")
```

and `LINE_BY_ROLE` in `team_strength.py` sends `AMW` to the **attack** line. So
every number 10 scores as a forward.

For Real Madrid 2026/27 that leaves the entire midfield line as three players:

| player | role | EA positions | minutes | rating |
|---|---|---|---|---|
| Federico Valverde | MID | CM, RB RM CDM | 819 | 76 |
| Aurélien Tchouaméni | DM | CDM, CM CB | 539 | 73 |
| Bernardo Silva | MID | CM, CDM RW RM | 458 | 71 |

Minutes-weighted, that is **73.85** — the published figure. Bellingham
(`CAM, CM LM ST`, 880 minutes, rated 87) is in the attack line instead,
alongside Mbappé (92) and Vinícius (83), which is why attack reads 88.0.

A second factor compounds it: only 13 Real Madrid players clear the
450-minute gate this season and exactly three are midfielders, so the line
quota of three is not a *best* three — it is the only three. Rüdiger (304
minutes) and Brahim Díaz (136) sit below the gate unrated.

**The bundling is a choice, not a limitation of the data.** EA distinguishes
`CAM` from `LW`/`RW`/`LM`/`RM`; we discard that distinction.

## How widespread it is

Over all 13 seasons in the file:

| primary EA position within `AMW` | player-seasons |
|---|---|
| RM | 1,765 |
| **CAM** | **1,715** |
| LM | 1,715 |
| LW | 735 |
| RW | 734 |

`CAM`-primary is **24.0% of the AMW bucket** — as common as any single winger
position, not a rare edge case. It is 4.9% of all player-seasons in the file
and **4.7% of rated ones** (1,161 of 24,669), which is the population that
actually reaches a team rating.

### It is a redistribution, not the correction of a bias

Moving `CAM` to the midfield line, ratings held fixed, over the 1,206
team-seasons production rates:

| line | team-seasons moved | mean change | mean absolute | range |
|---|---|---|---|---|
| midfield | 647 / 1,203 (53.8%) | **+0.09** | 0.57 | +8.98 / −8.81 |
| attack | 649 / 1,206 (53.8%) | −0.03 | 0.47 | +5.66 / −5.50 |
| overall | 651 / 1,206 (54.0%) | +0.02 | 0.13 | +2.28 / −2.08 |
| defence | 0 / 1,206 | 0.00 | 0.00 | — |

The means are indistinguishable from zero while the spread is nine points
wide. Real Madrid (+4.29 on midfield) is a large mover, not a typical one, and
it is not even the largest: Marseille's midfield **falls 8.81**, because adding
a lower-rated number 10 to a strong holding pair drags the line mean down.

So this cannot be argued as "the midfield rating is systematically too low".
It is not. It is too low *for some clubs* and too high for others, and the
question is which arrangement describes a side better.

## What the measurement changed

The first version of this proposal measured one arm — relabel the line, hold
the ratings fixed — and proposed a new `CAM` role on the strength of it. Both
halves of that turned out to be wrong in an interesting way.

### `role` is a rating pool, not a label

`rate_players` computes every percentile within `(role, Season)`. So a new
`CAM` role does not merely move those players to another line: it re-ranks
them against each other instead of against wingers, **and re-ranks the wingers
they leave behind**. Relabelling the line and holding the ratings fixed answers
a narrower question than the change poses, and the difference is most of the
effect.

Hence five arms, in two pairs that are never crossed:

| arm | what varies |
|---|---|
| `production` | nothing — reproduces `team_strength.csv` exactly |
| `line_only` | the midfield line gains `CAM`; the rating pool is untouched |
| `rerated` | nothing, but re-rated, so it carries the same jitter as the two below |
| `cam_to_mid` | `CAM` joins the existing `MID` role: MID's pool, MID's yardstick |
| `cam_new_role` | `CAM` gets a role of its own on `AMW`'s yardstick |

`cam_new_role` follows the precedent `DM` set when it was split out of `MID` —
*"the same metrics and the same weights as MID, deliberately … Separate the
players, not the yardstick."* Inventing weights for a new pool would be a
separate change with its own sweep.

### Against the following season's outcome

Production filtering throughout, so these are comparable with the figures
recorded elsewhere in the repo:

| arm | mid>pts | atk>GF | def>GA | ovr>pts |
|---|---|---|---|---|
| production | 0.6996 | 0.7382 | −0.5287 | 0.7556 |
| **line_only** | **0.7097** | **0.7417** | −0.5287 | **0.7568** |
| rerated | 0.6996 | 0.7382 | −0.5287 | 0.7556 |
| cam_to_mid | 0.7033 | 0.7397 | −0.5259 | 0.7549 |
| cam_new_role | 0.7073 | 0.7380 | −0.5265 | 0.7556 |

**The gain is in the line arrangement. Re-pooling is what costs.** `line_only`
takes midfield up **+0.0101**, past the 0.003 floor, and nothing else moves
against it: attack *improves* 0.0036, overall 0.0012, defence not at all.
Re-pooling gives a third to two-thirds of that back — `cam_new_role` keeps
+0.0077 and `cam_to_mid` only +0.0037, barely clearing the floor, while both
make `def>GA` slightly worse by re-selecting which fifteen players a club
contributes.

Two things the first version of this document got wrong, both found by
measuring under production filtering rather than a throwaway harness's own:
the midfield gain is **+0.0101, not +0.0156**, and the feared attack cost does
not merely straddle zero — it **reverses to +0.0036**.

### Walk-forward RPS

20,263 fixtures, paired bootstrap, each arm against its own baseline:

| arm | RPS | delta | 95% CI |
|---|---|---|---|
| production | 0.19906 | — | — |
| line_only | 0.19901 | −0.00005 | [−0.00018, +0.00008] |
| rerated | 0.19904 | — | — |
| cam_to_mid | 0.19905 | +0.00002 | [−0.00012, +0.00016] |
| cam_new_role | 0.19902 | −0.00002 | [−0.00016, +0.00012] |

Every interval straddles zero and every delta is inside ±0.00005, a tenth of
the 0.0005 floor. That is the expected and the wanted answer: the squad family
is 12 of 235 features and removing it *entirely* is worth +0.00015
[−0.00001, +0.00032], so nothing inside it can move this metric. A movement
well past the floor would have been evidence of a mistake.

### One caveat, stated rather than buried

`mid > points` is not one of the project's three declared benchmarks. It was
chosen because it is the line under discussion and because points is the only
outcome a midfield plausibly predicts — there is no "midfield goals" column.
It should be read as the metric this proposal nominated for itself, which is a
weaker thing than one declared in advance. The three declared benchmarks are
reported above in every arm and none of them degrades under `line_only`.

## What to build

**Send `CAM`-primary players to the midfield line, and leave the rating pool
alone.** One rule, keyed on EA's primary position, applied where
`team_strength` derives `line` from `role`.

This is a change to `LINE_BY_ROLE`'s *use*, not to `ROLE_BY_POSITION`. It is
the arm that measured best, and it is also the smallest:

- **No new role, so no new weights spec.** `cam_new_role` needed one and only
  a precedent — copy `AMW`'s — kept it honest.
- **No change to `_OUTFIELD` or `COMPATIBLE_ROLES`**, so the FIFA match rate
  cannot move. A new role would have had to be registered there, and the set's
  own comment records what forgetting costs: *"splitting `DM` out of `MID` and
  forgetting to list it here dropped the FIFA match rate from 92% to 87% in one
  run, because every CDM-primary entry in the database suddenly contradicted
  every player."*
- **The existing pool argument survives intact.** `ROLE_BY_POSITION` groups
  wingers with number 10s because *"their output profiles are near-identical"*
  — an argument about how to *score* a player. Which line he occupies is a
  different question, and the measurement says keep the first answer and change
  the second.

It also leaves alone the boundary the adjacent comment warns about:
*"Forward-to-midfield is not here and must not be added: false nines and
withdrawn forwards cross it constantly."* No role moves, so nothing about
person-matching changes.

The cost is cosmetic: the Player Database still labels these players
"Attacking midfielder / winger", because the role is what carries
`ROLE_LABELS`. A distinct label would need the EA position string surfaced
separately, which is a display change with no effect on any rating.

### Recorded as measured and not chosen

`cam_new_role` (+0.0077) and `cam_to_mid` (+0.0037) both clear the bar and
neither is worth its extra surface. `cam_to_mid` additionally flattens a
number 10 into a central midfielder, which is the thing the first version of
this proposal rejected on sight — and the measurement agrees with that
instinct for once, by a margin of 0.004.

## The hard gate

`team_strength.csv` is a **model input** — `features/build.py:156`,
`add_team_strength` — so unlike a purely published number, this one reaches
the feature table. Three conditions were declared; two are now met.

1. **Outstanding.** The full suite must pass with **no test modified to
   accommodate it**, the leakage guards included. This cannot be checked until
   the change is written.
2. **Met.** The feature table was rebuilt and walk-forward RPS re-measured for
   every arm. All deltas are inside ±0.00005 against a 0.0005 floor and all
   intervals straddle zero.
3. **Met.** The three recorded correlations were re-measured under production
   filtering. Under `line_only` all three improve or hold: `atk > GF` +0.0036,
   `ovr > pts` +0.0012, `def > GA` unchanged.

A fourth condition, not in the original list and worth adding: `build_players`
cannot run at all while Understat serves a holding page, so the published
ratings cannot be rebuilt today even once the change is written. The
measurement does not need it — the harness reads the built file — but shipping
does.

## Explicitly not in scope

- **No change to the minutes gate.** Real Madrid's thin line is partly a
  450-minute coverage question, and conflating the two would make neither
  measurable.
- **No change to `OVERALL_LINES` or the 4-3-3 quota.** If midfield gains a
  player class the temptation is to re-weight the lines; that is a separate
  change with its own sweep, and the defending-weight sweep already records
  how much work one of those is.
- **No reclassification of wingers.** `LM`/`RM`/`LW`/`RW` are genuine wide
  attackers and the measurement gives no reason to touch them. Güler
  (`RM`-primary) staying in attack is correct.
- **No change to the rating pools.** That is now a measured decision rather
  than an omission: re-pooling was the worse half of every arm that tried it.

## Open question for whoever builds it

Bellingham's EA positions are `CAM, CM LM ST` — his *secondary* positions
include a central midfield role. A narrower version of this change would move
only players whose secondary positions also include `CM` or `CDM`, leaving a
pure number 10 with no central cover in the attack line. That would cut the
1,715 affected player-seasons down substantially, and might sharpen the
`mid > points` gain rather than merely preserving it. It was not measured,
because the simple version is the one that should be accepted or rejected
first — and the simple version now has nothing measured against it.

## Reproducing the numbers

`pipelines/cam_line_ab.py`, which writes nothing to `data/processed`:

```
python -m pipelines.cam_line_ab                # ~30 minutes, five backtests
python -m pipelines.cam_line_ab --no-backtest  # ratings and correlations only
```

It reads the built `player_ratings.csv` rather than re-ingesting, because
Understat cannot be reached and because that file is the vintage these numbers
were measured on. Two fidelity checks are asserted at runtime rather than
claimed: the role chain re-derived from `fifa_positions` must reproduce the
shipped `role` column for all 34,856 rows, and the `production` arm must
reproduce `team_strength.csv` for all 1,206 team-seasons. Both hold.

Re-rating is faithful but not bit-exact — 10 of 24,669 ratings come back one
point different, from percentile ties resolving the other way after a CSV
round-trip. That is why `rerated` exists as a separate baseline: the three
re-rated arms share the jitter, so their deltas do not contain it. Its
correlations match `production`'s to four decimals, which says the jitter is
immaterial, but the pairing does not rely on that.
