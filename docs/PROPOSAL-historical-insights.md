# Historical insights: who outperformed expectations, and can we prove it

**Status: proposal. Nothing here is built.** Every number below was measured
on the 2026-11-02 vintage before the proposal was written, so the design
rests on what the data supports rather than on what would be nice.

## The question

*"Can we create historical insights about players or teams that outperformed
expectations — their performance rating was way above their EA rating? Are
there other options to show historical insights about players or teams?"*

The site already publishes both halves of a player rating side by side:
`performance_rating` (what we measured from the season) and
`fifa_on_our_scale` (EA's overall, ranked within role and season and mapped
onto our 0–99 scale, so the two are comparable by construction). The gap
between them is sitting there unused.

## What the gap is, measured

Over **23,683** rated player-seasons carrying both halves:

| | gap = performance − EA, on our scale |
|---|---|
| mean | +0.06 |
| sd | 9.37 |
| 5th / 95th percentile | −15.0 / +15.9 |
| 1st / 99th percentile | −21.3 / +23.2 |
| min / max | −33.5 / +44.3 |

It is centred on zero by construction and symmetric, and it is **stable across
seasons** (season means span +0.01 to +0.14) and **across roles** (role means
span −0.12 for keepers to +0.19 for forwards). So "outperformed EA" is a
well-defined quantity that needs no per-season or per-position normalisation.

### The board writes itself

Biggest positive gaps, all seasons:

| season | player | club | EA | ours | gap |
|---|---|---|---|---|---|
| 2015/16 | Mahmoud Dahoud | M'gladbach | 60 | 94 | **+44.3** |
| 2023/24 | **Cole Palmer** | Chelsea | 66 | 94 | **+42.0** |
| 2025/26 | Jacobo Ramón | Como | 64 | 88 | +36.7 |
| 2025/26 | Yan Diomande | Leipzig | 66 | 89 | +36.6 |
| 2019/20 | Dejan Kulusevski | Parma | 64 | 87 | +35.1 |
| 2020/21 | Maxence Lacroix | Wolfsburg | 66 | 89 | +33.6 |
| 2023/24 | Jarell Quansah | Liverpool | 62 | 83 | +34.4 |

These are recognisable breakout seasons, not artefacts. Cole Palmer's
2023/24 — EA 66, our 94 — is the case a reader will check first, and it is
the right answer.

## The finding that makes this publishable

A board of large gaps is only a curiosity. The question worth answering is
whether the gap is *right*: **does a player we rate far above EA get raised by
EA in the next edition?**

Over 12,776 player-seasons from 2016/17 on, joining each season's gap to the
same player's EA overall one edition later:

| gap decile | mean gap | EA's next move | share EA raised |
|---|---|---|---|
| 1 (we rate them lowest) | −16.0 | −0.93 | 17.4% |
| 5 | −1.2 | +0.48 | 42.9% |
| 10 (we rate them highest) | +17.7 | **+4.95** | **91.3%** |

Raw correlation **+0.58**.

### …and it survives the obvious objection

EA reverts to its own mean: `corr(fifa_overall, next move) = −0.55`. Low-rated
players get raised whoever they are, and the gap is mechanically
anti-correlated with `fifa_overall`, so the +0.58 could be nothing but that.
Two independent tests say it is not:

| test | correlation |
|---|---|
| raw | +0.580 |
| partialling out `fifa_overall` | +0.399 |
| partialling out `fifa_overall`, age, minutes | **+0.375** |
| **within players EA rated identically** (same season, same role, same overall; 487 strata, 7,744 players) | **+0.370** |

The model-free test and the regression agree to three decimals, which is the
strongest version of this result available. Among players **EA itself rated
the same number**, the fifth we rate highest gain **+1.70** next edition and
are raised 67.9% of the time; the fifth we rate lowest gain **−0.27** and are
raised 28.4%.

It holds at every age and is strongest where it should be: +0.39 under 23,
+0.36 at 24–28, +0.28 at 29 and over.

**So the honest headline is defensible**: *where we and EA disagree about a
player, EA moves toward us about a third of a standard deviation more than
chance — and nine times in ten for the most extreme disagreements.* That is a
real claim, it is checkable, and it is the kind of thing an interviewer can be
walked through.

## The team half

Fitting points against squad rating **within each league-season** (so the fit
cannot be gamed by league strength) over 762 club-seasons gives a residual sd
of **7.93 points**. The board:

| season | club | squad | points | expected | over |
|---|---|---|---|---|---|
| 2015/16 | **Leicester** | 68.6 | 81 | 45.8 | **+35.2** |
| 2023/24 | Girona | 72.0 | 81 | 54.1 | +26.9 |
| 2018/19 | Lille | 69.8 | 75 | 53.1 | +21.9 |
| 2022/23 | Arsenal | 78.9 | 84 | 65.7 | +18.3 |
| 2019/20 | Sheffield United | 67.6 | 54 | 36.7 | +17.3 |

and the other end: Aston Villa 2015/16 −26.6, Málaga 2017/18 −24.9,
Tottenham 2024/25 −24.1, Napoli 2023/24 −21.6.

Leicester 2015/16 is **+35.2, or 4.4 standard deviations**, the largest of 762.
Their own history makes the point better than the board does — every other
Leicester season sits within ±8:

    1415 +0.8   1516 +35.2   1617 −1.9   1718 −1.1   1819 −2.9
    1920 +3.4   2021 +7.8    2122 −7.3   2223 −14.2  2425 −4.1

This also settles an old tension in the repo honestly. `rating.py` says a
squad-quality rating that *liked* Leicester in 2015/16 would be the broken one
— EA had them 18th of 20 and the bookmakers said 5000-1. This board agrees
with that and makes it the point: the rating was right about the squad, and
the season was the most extreme overachievement in twelve years of data.

## What to build

Three panels, in descending order of how well they are supported.

1. **"We rated them before EA did."** The top positive gaps, with each
   player's EA overall the following edition beside it, so the claim is shown
   being tested rather than asserted. Filterable by season and role. The
   decile table above is the explainer.
2. **"Overachievers and underachievers."** Club-seasons by points above the
   within-league fit, with the club's own history as a sparkline so a single
   spike reads as a spike.
3. **A player's own gap over time**, on the existing player page: two lines,
   ours and EA's, which is where a reader goes after the first panel.

Serving: the gap is a subtraction of two published columns and the team
residual is one `polyfit` per league-season, so both are cheap. The player
panel's EA-next-edition join needs `player_ratings.csv` only. Follow the
pattern already in `app.py`: compute once, cache on `state`, clear on reload.

## What must be filtered, and the evidence for it

**The negative board is polluted and the positive board is not.** Rows whose
EA entry is attached to two different clubs in one season are 1.72% of the
data overall (an upper bound — a mid-season transfer looks identical), but
they are **6 of the top 30 negative gaps and 0 of the top 30 positive**.

The reason is structural: matching a player to a *highly rated* wrong entry
manufactures a huge negative gap, while the matcher's uniqueness guards make
the reverse rare. The top two negative rows are both it — Bruno Saltor
(Brighton) and Bruno González (Getafe) are **both** matched to
`Bruno Soriano Llido`, Villarreal's Bruno, EA 84, giving −33.5 and −32.4.

So: publish the positive board; publish the negative one only behind a guard
that rejects an EA entry attached to two clubs in one season whose names do
not share a token, and say on the page that a low number can be a mismatch.

Two more exclusions, both already documented in `rating.py`:

- **2014/15 and 2015/16** carry missing FBref tables, so `measured_share` is
  as low as 0.38 for centre backs and the performance half is not comparable.
  Exclude them from the player board, or mark them. (The EA-prediction result
  above already excludes them; including them only moves +0.580 to +0.566.)
- A **minutes floor** above the 450 the rating already requires, because the
  gap's tails are where thin seasons land.

## How it gets judged

Declared before building, as `squad_size_sweep` demands:

1. The within-stratum correlation must reproduce at **+0.37 ± 0.03** on the
   vintage the panel ships against. It is the panel's entire claim; if it does
   not reproduce, the panel ships without the claim or not at all.
2. The top-30 positive board must contain **zero** rows whose EA entry is
   attached to two clubs in one season.
3. Leicester 2015/16 must be first on the team board. Not a tuning target — a
   smoke test, because any fit that loses it has broken something.

## Explicitly not in scope

- **No model feature.** This describes the past; it predicts nothing the model
  does not already have, and `EXPERIMENTS.md` has the leave-one-out result
  saying squad strength is worth nothing to RPS anyway.
- **No "EA is wrong" framing.** The measured claim is that EA moves *toward*
  us afterwards, not that we are right and they are wrong. The 0.37 leaves
  most of the variance unexplained and the page should say so.
- **No transfer-value or "most improved" ranking.** Both need data we do not
  have, and "most improved" over two of our own seasons is a difference of two
  noisy numbers.

## Open question for whoever builds it

The gap is mechanically compressed for players whose role-season pool we
measured poorly: `fifa_weight_used` rises to 0.81 for a 2014/15 centre back,
so the published `rating` leans on EA and the *gap* does not. Whether the
board should show the gap as computed or scaled by `measured_share` is a real
decision and it is not obvious; it should be measured against the
EA-next-edition outcome the same way everything else here was.
