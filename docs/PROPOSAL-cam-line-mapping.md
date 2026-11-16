# Should an attacking midfielder count as midfield?

**Status: proposal. Nothing here is built.** Every number below was measured on
the 2026-11-02 `player_ratings.csv` vintage — the last successful
`build_players` before the Understat outage — and measured before the proposal
was written, so the design rests on what the data supports rather than on what
would be nice.

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

Over all 12 seasons of rated player-seasons:

| primary EA position within `AMW` | player-seasons |
|---|---|
| RM | 1,765 |
| **CAM** | **1,715** |
| LM | 1,715 |
| LW | 735 |
| RW | 734 |

`CAM`-primary is **24.0% of the AMW bucket** and 4.9% of all rated
player-seasons — as common as any single winger position, not a rare edge
case. Moving it would change **51.8%** of team-seasons' midfield rating and
51.6% of their attack rating.

### It is a redistribution, not the correction of a bias

| line | team-seasons moved | mean change | range |
|---|---|---|---|
| midfield | 623 / 1,203 (51.8%) | **+0.09** | +8.98 / −8.81 |
| attack | 622 / 1,206 (51.6%) | −0.03 | +5.66 / −5.50 |
| overall | 525 / 1,206 (43.5%) | +0.01 | +2.28 / −2.08 |

The means are indistinguishable from zero while the spread is nine points
wide. Real Madrid (+5.8 on midfield) is a large mover, not a typical one.
Midfield moves *down* for many clubs, because adding a lower-rated number 10
to a strong holding pair drags the line mean down.

So this cannot be argued as "the midfield rating is systematically too low".
It is not. It is too low *for some clubs* and too high for others, and the
question is which arrangement describes a side better.

## The finding that makes this worth proposing

Judged on the project's own three correlations, plus the midfield line against
points. Paired bootstrap, 2,000 resamples over 841 completed team-seasons:

| correlation | as published | CAM to midfield | delta | 95% CI |
|---|---|---|---|---|
| **mid > points** | 0.7597 | 0.7755 | **+0.0156** | [+0.0073, +0.0248] |
| atk > goals for | 0.8437 | 0.8387 | −0.0050 | [−0.0107, +0.0003] |
| ovr > points | 0.8464 | 0.8482 | +0.0017 | [+0.0001, +0.0034] |
| def > goals against | −0.6594 | −0.6594 | 0.0000 | unchanged |

**One result clears both zero and the project's 0.003 correlation floor:
`mid > points`, at +0.0156.** The midfield line genuinely describes a side
better when attacking midfielders are counted in it.

The feared cost does not appear. `atk > GF` drops 0.0050, but its interval
touches zero, so it is suggestive rather than established — moving number 10s
out of attack does not demonstrably damage the attack line. `ovr > points`
clears zero statistically but at +0.0017 sits *below* the measurement floor,
so the headline team rating is unaffected either way.

### Two caveats, stated rather than buried

The absolute correlations here do not match the 0.7562 / 0.7408 / −0.5232
recorded elsewhere in the repo, because this used its own filter — completed
seasons only, at least 30 matches played. Only the **deltas** are meaningful,
and both arms use identical filtering over identical team-seasons (the
pairing is asserted in the harness).

`mid > points` is not one of the project's three declared benchmarks. It was
chosen because it is the line under discussion and because points is the only
outcome a midfield plausibly predicts — there is no "midfield goals" column.
It should be read as the metric this proposal nominated for itself, which is
a weaker thing than one declared in advance.

## What to build

One change, in `src/footballml/players/fifa.py`:

```
CAM            -> its own role, mapped to the midfield line
LM, RM, LW, RW -> AMW, unchanged, still attack
```

A new role rather than mapping `CAM` onto `MID`, so the Player Database can
still say "Attacking midfielder" rather than flattening him into a central
midfielder, and so `ROLE_LABELS` stays honest.

**The new role must be added to `_OUTFIELD` in the same module.** That set
feeds `COMPATIBLE_ROLES`, which is what `contradicts_position()` reads to
decide whether an EA entry describes the same *person* as the Understat row.
`COMPATIBLE_ROLES` derives its `D`, `M` and `F` groups from `_OUTFIELD`
directly (`F` only subtracts `CB`), so one edit covers all three — but
omitting it would reject every `CAM`-primary match wholesale.

That exact mistake has been made here before, and the set's own comment
records it: *"splitting `DM` out of `MID` and forgetting to list it here
dropped the FIFA match rate from 92% to 87% in one run, because every
CDM-primary entry in the database suddenly contradicted every player."*

Note also the adjacent warning, which this change must not disturb:
*"Forward-to-midfield is not here and must not be added: false nines and
withdrawn forwards cross it constantly."* A new attacking-midfield role sits
squarely on that boundary, so it belongs in `_OUTFIELD` — reachable from the
`F` group as well as `M` — rather than being treated as midfield-only.

## The hard gate

`team_strength.csv` is a **model input** — `features/build.py:156`,
`add_team_strength` — so unlike a purely published number, this one reaches
the feature table. The change therefore ships only if:

1. The full suite passes with **no test modified to accommodate it**, the
   leakage guards included.
2. The feature table is rebuilt and the walk-forward RPS re-measured. The
   squad family is 12 of 235 features and removing it entirely is worth
   +0.00015 RPS [−0.00001, +0.00032], so the expected movement here is
   *smaller than that* and should not clear the 0.0005 RPS floor. **A
   movement larger than the floor is the bug signal**, not a result, and the
   change is reverted rather than argued about.
3. The three recorded correlations are re-measured under production
   filtering, not this proposal's, and `atk > GF` is confirmed not to have
   moved beyond its floor.

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

## Open question for whoever builds it

Bellingham's EA positions are `CAM, CM LM ST` — his *secondary* positions
include a central midfield role. A narrower version of this change would move
only players whose secondary positions also include `CM` or `CDM`, leaving a
pure number 10 with no central cover in the attack line. That would cut the
1,715 affected player-seasons down substantially, and might keep the
`atk > GF` delta nearer zero while retaining most of the `mid > points` gain.
It was not measured, because the simple version is the one that should be
accepted or rejected first.

## Reproducing the numbers

The three harnesses that produced every figure above are not committed — they
are throwaway measurement scripts, and the numbers they produced are recorded
here so they do not have to be re-derived. Each reads
`data/processed/player_ratings.csv` and `team_match_history_all.csv`, relabels
`role` for `CAM`-primary rows, and calls the production
`team_strength(method="best_n")` on both arms so that nothing about the
aggregation differs between them.

One detail worth repeating if they are rebuilt: two team-seasons have a `NaN`
midfield rating, and dropping them pairwise rather than letting them through
is the difference between a usable `mid > points` interval and a `NaN`.
