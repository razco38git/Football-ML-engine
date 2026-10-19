# Proposal: decouple European validation from the weekly critical path

**Status: investigation only. Nothing implemented, nothing changed.** No
production code, data, config, model, artifact or task registration was
modified, and the run in flight was left alone.

Written 2026-10-19, while a weekly refresh was 229 minutes into a job whose
lock file claims it should never exceed an hour.

---

## 1. What each step reads and writes

| # | step | reads | writes | atomic? |
|---|---|---|---|---|
| 1 | `build_dataset` | football-data CSVs (cached), FBref top-up, Understat xG | `matches_all.csv`, `team_match_history_all.csv` | **no** |
| 2 | `build_players` | Understat, FBref, EA editions | `player_ratings.csv`, `team_strength.csv` | **yes** (`_write_csv`, tmp+replace) |
| 3 | `train` | `team_match_history_all.csv`, `team_strength.csv` | `models/<version>/`, `models/latest.txt` | via `registry.save` |
| 4 | `score_upcoming` | model, features, `matches_all.csv` | prediction store (`store.append`, `store.settle`) | store-managed |
| 5 | `project_season` | model, features | `season_projection.csv` | **no** |
| 6 | `fetch_european` | FBref (headless browser) | `european_matches.csv`, `european_dropped.csv` | **no** |
| 7 | `validate_european` | `european_matches.csv`, `team_match_history_all.csv`, `team_strength.csv`, **`registry.load()`** | `european_features.pkl`, `european_predictions.csv` | **pkl yes**, csv **no** |
| 8 | `fit_league_adjustment` | `european_predictions.csv`, **`registry.load()`** (for the version stamp) | `league_adjustment.json` | **no** |
| 9 | `reload` | — | `POST /admin/reload` | n/a |

## 2. Dependency diagram

```
                 football-data / FBref / Understat / EA
                                  |
                          (1) build_dataset
                                  |
                   matches_all.csv, team_match_history_all.csv
                          |                    |
                  (2) build_players            |
                          |                    |
                   team_strength.csv           |
                          |                    |
                          +--------> (3) train <+
                                        |
                                  models/<version>
                                        |
          +--------------+--------------+------------------+
          |              |              |                  |
   (4) score_upcoming  (5) project   (7) validate_european  |
          |              |            ^        |           |
   prediction store  projection.csv   |  european_predictions.csv
          |              |            |        |           |
          |              |            |  (8) fit_league_adjustment
          |              |            |        |           |
          |              |            |  league_adjustment.json
          |              |            |        |           |
          +--------------+------------|--------+-----------+
                                      |        |
                   (6) fetch_european -+       |
                   european_matches.csv        |
                                               v
                                        (9) reload -> API
```

**The load-bearing edges.** `validate_european` and `fit_league_adjustment`
both read `registry.load()`, and that is the coupling that matters: the offsets
answer *"where is this model wrong when a Spanish club plays an English one"*,
fitted on the model's own residuals as a Poisson offset. A retrain makes them
stale by construction. Everything else about the European branch is
independent of the domestic one.

**The edge that does not exist.** Steps 4, 5 and 9 do not read anything the
European branch writes. `reload` re-reads `european_predictions.csv` and
`league_adjustment.json` if present, but neither gates the domestic model. So
publishing the domestic model before European validation finishes is already
safe in the data-flow sense — the only question is what the API serves in the
window, covered in §5.

---

## 3. Findings

### 3.1 The lock's staleness threshold is wrong, and this is urgent

`pipelines/_lock.py`:

```python
#: The weekly job's longest run on record is 19m 37s. An hour is comfortably
#: past that and still short enough that a crashed run does not block the next
#: morning's.
STALE_AFTER_SECONDS = 3600.0
```

That assumption is obsolete. Recorded weekly runtimes since:

| run | duration |
|---|---|
| 5 Oct | 19m 37s |
| 12 Oct | **190m 37s** |
| 15 Oct | **71m 43s** |
| 19 Oct | **229m and counting** |

Three of the last four exceeded the threshold. The live lock as I write is
**228 minutes old** — 3.8x stale — while its owner is alive and writing:

```
weekly pid=28560 since=2026-10-19T08:25:06+00:00
```

**Consequence:** any daily run firing now would call `_is_stale`, decide the
owner is dead, delete the lock and proceed. Two jobs would then write
`matches_all.csv`, `team_match_history_all.csv` and `season_projection.csv`
concurrently — none of which is written atomically. This is the exact failure
`build_players` added `_write_csv` to prevent, reachable today by a different
route.

It has not fired yet only because the daily trigger is 07:30 and the catch-up
on 19 Oct ran at 09:54, before the weekly started.

**This is independent of the restructuring and should be fixed first.**

### 3.2 Liveness, not age, is the right staleness test

The lock file already records `pid=`. `_is_stale` ignores it and uses mtime
alone. Checking whether that pid is alive turns a guess into a fact and removes
the need to predict a maximum runtime at all — which is what went wrong.

### 3.3 Atomicity is inconsistent

Atomic today: `team_strength.csv`, `player_ratings.csv` (`_write_csv`),
`european_features.pkl` (`_checkpoint`, tmp + `Path.replace`).

Not atomic: `matches_all.csv`, `team_match_history_all.csv`,
`season_projection.csv`, `european_matches.csv`, `european_dropped.csv`,
`european_predictions.csv`, `league_adjustment.json`.

`league_adjustment.json` is the one that matters most for correctness: the API
reads it on reload, and a torn write leaves either a `JSONDecodeError` (handled
— `load` returns `{}`) or, worse, valid JSON missing `model_version`, which
`fitted_model_version` returns as `None` and the API then treats as *"no
version claimed"* and serves without warning.

### 3.4 The expensive cache is already resumable and already atomic

This is the good news, and it narrows the problem.

- `_checkpoint` writes tmp + `Path.replace`, with a docstring explaining that a
  truncated cache carrying a valid fingerprint is worse than no cache.
- `CHECKPOINT_EVERY = 25` matchdays, ~4 minutes of work at risk.
- `build_features` reuses any cached row whose `(Date, HomeTeam, AwayTeam)` key
  is still wanted, so an interrupted run resumes rather than restarts.
- Invalidation is a hash of **completed-season** `team_strength.csv` only, so
  the weekly rebuild of the current season does not invalidate it.

**So interrupting `validate_european` is already safe**, and the work already
done is already preserved. That is what makes design C viable.

### 3.5 Why the cache has not been paying off

Measured today: 130 rows / 75 matchdays cached at the start of a run that needs
441. Yet the fingerprint was **unchanged** across this run
(`f9c69fc5d450` before and after `build_players` rewrote `team_strength.csv`).

So the cache is not being invalidated — it is **incomplete**, because previous
runs were interrupted or failed (12 Oct's `validate_european` ran 148 minutes
and is absent from that run's completed-step list). Each subsequent run has
been paying down a backlog rather than hitting a warm cache.

Once a run completes all 441, a normal week should return to the 30 seconds
observed on 5 Oct. **This weakens the case for restructuring on runtime grounds
alone** and strengthens the case for restructuring on *robustness* grounds: the
step is fragile, and its fragility currently blocks publication.

### 3.6 Measured build rate

12.6s/matchday is the docstring's figure; today's measured rate is **~38s**
(200 → 225 matchdays in 16 minutes). Any runtime estimate should use the
measured rate, not the comment.

### 3.7 API behaviour with the adjustment missing, stale, or mid-replace

| state | behaviour |
|---|---|
| missing | `load` returns `{}`; logs *"No cross-league correction; run ..."*; serves uncorrected |
| corrupt | `JSONDecodeError` caught; same as missing |
| stale (version mismatch) | **serves it anyway**, logs a warning naming both versions |
| mid-replace (torn write) | race: non-atomic `write_text` can be read half-written |
| valid JSON, no `model_version` | `fitted_model_version` returns `None`; the check is `not in (None, version)`, so **no warning** |

The stale path is a deliberate choice and documented: *"a stale correction is
still closer to right than none, and refusing to serve would be worse."* Goal 3
says never serve a mismatched adjustment *silently* — today it is not silent in
the logs, but nothing surfaces it in the API response.

### 3.8 Daily/weekly interaction

`daily` takes the lock with `wait_seconds=0` and gives up immediately if held —
correct, since the weekly supersedes it. `weekly` waits `LOCK_WAIT_SECONDS =
600`. Both behaviours are right; only the staleness test is wrong.

Catch-up (`StartWhenAvailable=True`, now with `WakeToRun=True`) fires both jobs
at the same instant after a missed window — observed at 09:54:55 today. The
lock handled it correctly then, because the weekly had just started.

---

## 4. Designs compared

### A. Move `reload` ahead of European validation

Order becomes 1-5, reload, 6-8, reload again.

| | |
|---|---|
| **runtime on critical path** | ~25 min (domestic only) |
| **failure modes** | second reload can fail silently, leaving the adjustment fitted but unpublished until next week |
| **stale data** | between the two reloads the API serves the new model with the **previous** model's adjustment — exactly the mismatch goal 3 targets, now guaranteed weekly rather than occasional |
| **version compatibility** | knowingly broken for the duration of step 6-8 |
| **recovery** | simple; rerun the weekly |
| **effort** | smallest (reorder a tuple) |

Rejected: it converts an occasional mismatch into a scheduled one.

### B. Separate European job on its own schedule

Steps 6-8 + a reload move to a new scheduled task every 2-3 weeks.

| | |
|---|---|
| **runtime on critical path** | ~25 min |
| **failure modes** | two jobs writing `data/processed`; needs the lock to be correct first |
| **stale data** | adjustment lags the model by up to 3 weeks, i.e. 2 weeks in 3 |
| **version compatibility** | **systematically mismatched**, because the model retrains weekly and the adjustment does not |
| **recovery** | independent; a failed European run does not touch the domestic one |
| **effort** | medium (new task, new entry point, lock changes) |

Rejected on goal 3. The offsets are fitted on *this model's* residuals; a
3-week-old fit describes a model that was retrained twice since.

### C. Split by cost, not by step — **recommended**

Separate the two things `validate_european` currently does:

1. **Feature building** for new matchdays — expensive, fragile (headless
   browser), and *model-independent*: a tie's features are built from the
   domestic matches before it, which is why they are cacheable at all.
2. **Scoring + fitting** — cheap (30s observed on 5 Oct), and *model-dependent*.

| | |
|---|---|
| **runtime on critical path** | ~25 min + ~30s |
| **failure modes** | cache-building job fails → weekly scores whatever is cached, reports coverage, continues |
| **stale data** | adjustment **always** matches the serving model; only *coverage* lags |
| **version compatibility** | correct by construction |
| **recovery** | cache is resumable and atomic already (§3.4) |
| **effort** | medium |

This satisfies all four goals. Goal 1: the critical path loses the expensive
work. Goal 2: order is preserved — fetch → build → score with the intended
model → fit. Goal 3: the fit happens every week against the model just trained,
so a mismatch cannot arise. Goal 4: handled by the lock fix, below.

The honest cost: when the cache is cold, the weekly scores fewer European ties
than exist, so the adjustment is fitted on a subset. That must be **reported,
not hidden** — see the tests.

---

## 5. Recommended design in detail

**New split:**

```
weekly (every Monday, ~25 min)
  build_dataset, build_players, train, score_upcoming, project_season,
  score_european (cached features only, no building),
  fit_league_adjustment,
  reload

european-cache (every 2-3 weeks, hours, optional)
  fetch_european,
  build_european_features   <- the expensive part, resumable
```

`score_european` never builds. If a tie has no cached features it is skipped
and counted. The weekly therefore has a bounded runtime that does not depend on
the UEFA calendar.

### Files and functions that would change

| file | change |
|---|---|
| `pipelines/_lock.py` | `_is_stale` → check pid liveness, not mtime; keep a much larger mtime backstop for a pid that was reused |
| `pipelines/validate_european.py` | split `main` into `build_cache(...)` and `score(...)`; add `--build/--no-build`; make `PREDICTIONS_OUT` atomic |
| `pipelines/weekly.py` | `STEPS`: replace `validate_european` with `score_european` (no build); move `fetch_european` out |
| `pipelines/european_cache.py` | **new**: `fetch_european` + `build_cache`, its own entry point |
| `src/footballml/league_adjust.py` | `save` → atomic tmp+replace; require `model_version`, refuse to save without one |
| `src/footballml/api/app.py` | surface a version mismatch in `/health` (a field, not only a log line) |
| `pipelines/build_dataset.py`, `project_season.py`, `fetch_european.py` | route writes through a shared atomic `write_csv` helper |
| `scripts/register_european.ps1` | **new**: registers the 3-weekly task, staggered away from 07:00/07:30 |

### Tests required

**Lock**
- a live pid holding an old lock is **not** stale
- a dead pid holding a fresh lock **is** stale
- two concurrent `refresh_lock` calls: exactly one wins
- the daily job declines while the weekly holds it, even after an hour

**Atomic publication**
- a write interrupted before `replace` leaves the previous file intact
- `league_adjustment.json` without `model_version` is refused at save time
- a half-written adjustment is never observed by `load`

**Dependency order**
- `score_european` raises if `european_predictions.csv` would be fitted against
  a different model version than `registry.load()`
- `fit_league_adjustment` stamps the version it actually loaded

**Coverage honesty**
- `score_european` with a cold cache reports how many ties it could not score
- the weekly logs that count rather than reporting success unqualified

**No-build guarantee**
- `score_european` performs zero feature builds — asserted by patching the
  builder and requiring it is never called

### Migration, staged

1. **Lock fix alone.** Ship, observe one weekly and one daily. Independently
   valuable and carries no restructuring risk.
2. **Atomic writes alone.** Ship, verify artifact hashes are unchanged across a
   run.
3. **Split `validate_european` into build/score, both still called by the
   weekly.** Behaviour identical; the split is internal. Verify a full weekly
   produces byte-identical `european_predictions.csv`.
4. **Move the build out** into `european_cache.py` and register its task. The
   weekly now scores only. First run with a warm cache so coverage is complete.
5. **Watch one European matchweek** to confirm coverage lag is acceptable.

### Rollback

Each stage is independently revertible. Stage 4 is the only one with a schedule
change: rolling back means re-adding `build_european_features` to the weekly
`STEPS` tuple and unregistering the task — no data migration, because the cache
format does not change at any stage.

---

## 6. What I would do first, regardless

**The lock.** It is a live correctness hazard today, it is unrelated to the
restructuring, and the fix is small. Everything else here is an improvement;
that one is a bug.
