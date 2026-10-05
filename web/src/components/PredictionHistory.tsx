import { useState } from 'react';
import { api, COMPETITION_NAMES, LEAGUE_NAMES, competitionLabel, type Accuracy, type MatchResult, type ModelInfo } from '../api/client';
import { formatDate, pct, teamColor } from '../api/display';
import { ReliabilityDiagram } from './Charts';
import { useAsync } from '../api/hooks';
import { CallVersusResult, FinalScore, OutcomeBar, OutcomeSplit, ResultBadge } from './MatchOutcome';

type Source = 'live' | 'backtest';

/**
 * The two sources make different claims, so they are labelled differently.
 *
 * `live` predictions were stored before kickoff and settled afterwards — a
 * genuine track record, and currently a very short one. `backtest` predictions
 * come from the walk-forward run: the model was trained only on earlier seasons
 * and never saw the match, so they are honestly out-of-sample, but they were
 * generated retrospectively rather than committed to in advance.
 */
const SOURCES: Record<Source, { label: string; blurb: string }> = {
  live: {
    label: 'Live track record',
    blurb:
      'Predictions stored before kickoff and settled once the match finished. The strictest claim we can make — and it grows a few matches per week.',
  },
  backtest: {
    label: 'Backtest history',
    blurb:
      'Walk-forward predictions: each match was forecast by a model trained only on earlier seasons, so it had never seen the result. Out-of-sample, but generated retrospectively rather than committed to in advance.',
  },
};

function ResultRow({ m }: { m: MatchResult }) {
  return (
    <div
      className="rounded-xl p-4"
      style={{
        background: 'var(--card)',
        border: `1px solid ${m.correct ? 'rgba(0,230,118,0.25)' : 'var(--border)'}`,
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs tracking-widest uppercase" style={{ color: 'var(--muted-foreground)' }}>
          {competitionLabel(m.league)}
        </span>
        <div className="flex items-center gap-2">
          <ResultBadge correct={m.correct} />
          <span className="text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
            {formatDate(m.date)}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2" style={{ flex: '1 1 0', minWidth: 0 }}>
          <div
            className="w-7 h-7 rounded flex items-center justify-center text-xs font-display font-bold flex-shrink-0"
            style={{ background: teamColor(m.home_team) + '22', color: teamColor(m.home_team) }}
          >
            {m.home_team[0]}
          </div>
          <span className="text-sm truncate" style={{ color: 'var(--foreground)' }}>{m.home_team}</span>
        </div>

        {/*
          Two numbers from opposite sides of kick-off, stacked. The big one is
          what happened; the small one is what the model expected *before* it.
          Unlabelled, "3-1" above "xG 2.10-1.06" reads as one claim about the
          match, and a reader cannot tell whether the xG is the forecast or the
          chances actually created -- it is the forecast, and the whole page is
          about scoring that forecast.
        */}
        <div className="flex flex-col items-center flex-shrink-0" style={{ minWidth: 104 }}>
          <FinalScore
            homeGoals={m.actual_home_goals}
            awayGoals={m.actual_away_goals}
            expectedHome={m.expected_goals_home}
            expectedAway={m.expected_goals_away}
          />
        </div>

        <div className="flex items-center gap-2 justify-end" style={{ flex: '1 1 0', minWidth: 0 }}>
          <span className="text-sm truncate text-right" style={{ color: 'var(--foreground)' }}>{m.away_team}</span>
          <div
            className="w-7 h-7 rounded flex items-center justify-center text-xs font-display font-bold flex-shrink-0"
            style={{ background: teamColor(m.away_team) + '22', color: teamColor(m.away_team) }}
          >
            {m.away_team[0]}
          </div>
        </div>
      </div>

      <div className="mt-3">
        {/*
          All three, as the Match Predictor shows them. "Predicted Villarreal at
          59%" alone leaves the other 41% unaccounted for, and whether the model
          thought the danger was a draw or an away win is the more interesting
          half of a wrong call.
        */}
        <OutcomeBar home={m.prob_home_win} draw={m.prob_draw} away={m.prob_away_win} />
        <OutcomeSplit
          home={m.home_team}
          away={m.away_team}
          probHome={m.prob_home_win}
          probDraw={m.prob_draw}
          probAway={m.prob_away_win}
          actual={m.actual_result}
          pct={pct}
        />
        <CallVersusResult
          predicted={m.predicted_outcome}
          actual={m.actual_result}
          home={m.home_team}
          away={m.away_team}
          correct={m.correct}
        />
      </div>
    </div>
  );
}

function Stat({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  return (
    <div className="rounded-xl p-5" style={{ background: 'var(--card)', border: '1px solid var(--border)' }}>
      <div className="text-xs font-display uppercase tracking-wider mb-2" style={{ color: 'var(--muted-foreground)' }}>
        {label}
      </div>
      <div className="font-display font-black text-4xl" style={{ color }}>{value}</div>
      <div className="text-xs mt-1" style={{ color: 'var(--muted-foreground)' }}>{sub}</div>
    </div>
  );
}

/**
 * Why ~52% is not a bad number, said on the page rather than in a conversation.
 *
 * A football match has three outcomes, so the reference points are 33% for a
 * guess and the two tiles above: always backing the home side, and the
 * bookmakers on the same fixtures. The calibration strip makes the second, more
 * useful point -- picking one winner is hard, but the *probabilities* hold up,
 * which is what actually matters when reading a prediction.
 */
/**
 * The benchmarks that make our own number mean something.
 *
 * An RPS is not interpretable alone. It only says anything against what
 * guessing would score and against what the people with team news and money on
 * it manage — so all three sit on one row, in order, with the gap stated rather
 * than left for the reader to work out.
 */
function ModelVersusMarket({
  summary,
  confident,
  confidentHits,
}: {
  summary: Accuracy;
  confident: number;
  confidentHits: number;
}) {
  const rows = [
    { label: 'This model', rps: summary.rps, acc: summary.accuracy, note: '', us: true },
    {
      label: 'Bookmakers',
      rps: summary.rps_market,
      acc: summary.accuracy_market,
      note: 'the realistic ceiling — they have team news',
      us: false,
    },
    {
      label: 'Guessing',
      rps: summary.rps_base_rate,
      acc: summary.accuracy_base_rate,
      note: 'always back the home side',
      us: false,
    },
  ];
  const captured =
    summary.rps_market != null && summary.rps_base_rate > summary.rps_market
      ? (summary.rps_base_rate - summary.rps) / (summary.rps_base_rate - summary.rps_market)
      : null;

  return (
    <div
      className="rounded-xl p-5 mb-6"
      style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
    >
      <div
        className="text-xs font-display font-bold uppercase tracking-wider mb-3"
        style={{ color: 'var(--muted-foreground)' }}
      >
        Against the two benchmarks that matter
      </div>
      <table className="w-full text-sm">
        <tbody>
          {rows.map(r => (
            <tr key={r.label}>
              <td className="py-1" style={{ color: 'var(--foreground)' }}>{r.label}</td>
              <td
                className="py-1 text-right font-data font-bold"
                style={{ color: r.us ? '#00e676' : 'var(--muted-foreground)' }}
              >
                {r.rps != null ? r.rps.toFixed(4) : '—'}
              </td>
              <td className="py-1 text-right font-data" style={{ color: 'var(--muted-foreground)' }}>
                {r.acc != null ? pct(r.acc) : '—'}
              </td>
              <td className="py-1 pl-4 text-xs" style={{ color: 'var(--muted-foreground)' }}>
                {r.note}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs mt-3 leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
        {captured != null && (
          <>
            We cover{' '}
            <strong style={{ color: 'var(--foreground)' }}>{pct(captured)}</strong> of
            the distance from guessing to the bookmakers. What is left is mostly
            information we do not have — lineups, injuries, who is fit this weekend.{' '}
          </>
        )}
        {confident > 0 && (
          <>
            Where we called it at 60% or better, we were right{' '}
            <strong style={{ color: 'var(--foreground)' }}>
              {pct(confidentHits / confident)}
            </strong>{' '}
            of the time ({confident} of the matches shown).
          </>
        )}
      </p>
    </div>
  );
}

/**
 * Every trained artifact and how it scored on its held-out season.
 *
 * The question a model you keep retraining has to be able to answer: is this
 * month's better than last month's? The registry has recorded it all along —
 * this is the first thing to read it.
 *
 * Held-out, not walk-forward: each version was scored on the most recent season
 * it had *not* trained on. It is one season rather than twelve, so treat a
 * difference of a few ten-thousandths as noise.
 */
function ModelVersions() {
  const [open, setOpen] = useState(false);
  const { data } = useAsync<ModelInfo[]>(() => api.models(), []);
  if (!data || data.length === 0) return null;

  const shown = [...data]
    .sort((a, b) => b.version.localeCompare(a.version))
    .slice(0, open ? 12 : 4);
  const rps = (m: ModelInfo) => m.metrics?.rps ?? null;
  const best = Math.min(...data.map(m => rps(m) ?? Infinity));

  return (
    <div
      className="rounded-xl p-5 mb-6"
      style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
    >
      <div
        className="text-xs font-display font-bold uppercase tracking-wider mb-3"
        style={{ color: 'var(--muted-foreground)' }}
      >
        Is this version better than the last one?
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr style={{ color: 'var(--muted-foreground)' }}>
            <th className="text-left font-normal pb-1 text-xs">Trained</th>
            <th className="text-right font-normal pb-1 text-xs">Through</th>
            <th className="text-right font-normal pb-1 text-xs">Matches</th>
            <th className="text-right font-normal pb-1 text-xs">Features</th>
            <th className="text-right font-normal pb-1 text-xs">Held-out RPS</th>
          </tr>
        </thead>
        <tbody>
          {shown.map(m => {
            const value = rps(m);
            return (
              <tr key={m.version}>
                <td className="py-1 font-data text-xs" style={{ color: 'var(--foreground)' }}
                    title={m.version}>
                  {m.trained_at.slice(0, 10)}
                </td>
                <td className="py-1 text-right font-data text-xs" style={{ color: 'var(--muted-foreground)' }}>
                  {m.trained_through}
                </td>
                <td className="py-1 text-right font-data text-xs" style={{ color: 'var(--muted-foreground)' }}>
                  {m.n_train.toLocaleString()}
                </td>
                <td className="py-1 text-right font-data text-xs" style={{ color: 'var(--muted-foreground)' }}>
                  {m.n_features}
                </td>
                <td
                  className="py-1 text-right font-data font-bold"
                  style={{ color: value != null && value === best ? '#00e676' : 'var(--muted-foreground)' }}
                >
                  {value != null ? value.toFixed(4) : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {data.length > 4 && (
        <button
          onClick={() => setOpen(o => !o)}
          className="text-xs mt-3"
          style={{ color: 'var(--muted-foreground)' }}
        >
          {open ? 'Show fewer' : `Show more (${data.length} versions)`}
        </button>
      )}
      <p className="text-xs mt-2 leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
        Each version scored on the most recent season it had not trained on. That
        is one season rather than twelve, so a few ten-thousandths between two
        rows is noise, not progress.
      </p>
    </div>
  );
}

function Explainer({ summary }: { summary: Accuracy }) {
  // Every bin, including the small ones. Filtering to the big bands hid the
  // low-probability end — exactly where a model is most often wrong, and where
  // a reader most needs to see it.
  const bins = [...summary.calibration].sort((a, b) => a.mean_predicted - b.mean_predicted);

  return (
    <div
      className="rounded-xl p-5 mb-6"
      style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
    >
      <div
        className="text-xs font-display font-bold uppercase tracking-wider mb-3"
        style={{ color: 'var(--muted-foreground)' }}
      >
        Can the percentages be trusted?
      </div>

      <div className="flex flex-wrap gap-6 items-start">
        {bins.length > 0 && <ReliabilityDiagram bins={bins} />}
        <div style={{ flex: '1 1 300px', minWidth: 260 }}>
          <p className="text-sm leading-relaxed" style={{ color: 'var(--foreground)' }}>
            The question accuracy cannot answer. Each dot is a group of outcomes
            we gave about the same probability; its height is how often they
            actually happened. <strong>On the dashed line means the number can be
            read at face value</strong> &mdash; of the things we called 30%, 30%
            happened. Below it is over-confidence, above it is caution. Dot size
            is how many outcomes are in the group.
          </p>
          <p className="text-sm leading-relaxed mt-3" style={{ color: 'var(--foreground)' }}>
            Accuracy only asks whether the top pick came in. A match has three
            results, so guessing at random gets <strong>33%</strong> and always
            backing the home side gets{' '}
            <strong>{pct(summary.accuracy_base_rate)}</strong>. Draws are the hard
            part: about a quarter of matches end level, but a draw is almost never
            the single likeliest result, so those are nearly impossible to call.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function PredictionHistory() {
  const [source, setSource] = useState<Source>('backtest');
  const [league, setLeague] = useState('All');

  /**
   * The backtest is a walk-forward over the domestic history and contains no
   * European tie, so a competition filter there can only ever return an error
   * — and since the dropdown hides those options under the backtest, there
   * would be no way to clear it from the page.
   *
   * Derived rather than reset on click: a click handler guards the one
   * transition it is attached to, and leaves every other route into this state
   * — a stale tab, a hot reload that kept the old state — stuck on an error.
   */
  const competitionPicked = league in COMPETITION_NAMES;
  const effectiveLeague = source === 'backtest' && competitionPicked ? 'All' : league;

  const { data, loading, error, reload } = useAsync(
    () => api.history(source, effectiveLeague === 'All' ? undefined : effectiveLeague, 60),
    [source, effectiveLeague],
  );

  const summary = data?.summary;
  const matches = data?.matches ?? [];
  const confident = matches.filter(m => m.confidence >= 60);
  const confidentHits = confident.filter(m => m.correct).length;

  return (
    <div>
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Prediction Accuracy
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          {SOURCES[source].blurb}
        </p>
      </div>

      <div className="flex flex-wrap gap-3 mb-5 items-center">
        <div className="flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          {(Object.keys(SOURCES) as Source[]).map(s => (
            <button
              key={s}
              onClick={() => setSource(s)}
              className="px-4 py-2 text-sm font-display font-bold transition-colors"
              style={{
                background: source === s ? 'rgba(0,230,118,0.15)' : 'transparent',
                color: source === s ? '#00e676' : 'var(--muted-foreground)',
              }}
            >
              {SOURCES[s].label}
            </button>
          ))}
        </div>

        <select
          value={effectiveLeague}
          onChange={e => setLeague(e.target.value)}
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          <option value="All">All competitions</option>
          {Object.entries(LEAGUE_NAMES).map(([code, name]) => (
            <option key={code} value={code}>{name}</option>
          ))}
          {/* Live record only. The backtest holds no European tie, so offering
              one here is an option that can only produce an error. The derived
              `effectiveLeague` above is the backstop for state that arrives
              here by some other route. */}
          {source === 'live' &&
            Object.entries(COMPETITION_NAMES).map(([code, name]) => (
              <option key={code} value={code}>{name}</option>
            ))}
        </select>

        {data && (
          <span className="text-xs ml-auto" style={{ color: 'var(--muted-foreground)' }}>
            {data.total.toLocaleString()} settled matches
          </span>
        )}
      </div>

      {loading && !data && (
        <div className="text-center py-16 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Loading results…
        </div>
      )}

      {/* An empty live record is the expected state early on, not a failure. */}
      {error && (
        <div
          className="rounded-xl p-6 text-center"
          style={{ background: 'rgba(255,145,0,0.08)', border: '1px solid rgba(255,145,0,0.3)' }}
        >
          <div className="text-sm mb-2" style={{ color: '#ff9100' }}>{error}</div>
          {source === 'live' && (
            <div className="text-xs mb-4" style={{ color: 'var(--muted-foreground)' }}>
              {competitionPicked ? (
                <>
                  European rounds are a fortnight apart and only ties between big-five clubs
                  can be rated, so this record builds up slowly — a handful of matches per
                  round, and nothing at all between them.
                </>
              ) : (
                <>
                  The live record only counts predictions made before kickoff, so it fills up a
                  few matches at a time.
                </>
              )}
            </div>
          )}
          <button
            onClick={
              competitionPicked
                ? () => setLeague('All')
                : source === 'live'
                  ? () => setSource('backtest')
                  : reload
            }
            className="px-4 py-2 rounded-lg text-sm font-display font-bold"
            style={{ background: 'rgba(0,230,118,0.15)', color: '#00e676' }}
          >
            {competitionPicked
              ? 'Show all competitions'
              : source === 'live'
                ? 'Show backtest history instead'
                : 'Retry'}
          </button>
        </div>
      )}

      {summary && (
        <>
          {/*
            RPS first, because it is the metric every decision in this project
            was actually made on. Accuracy stays beside it because it is the one
            people read — but a model is chosen on the whole distribution it
            publishes, not on whether its top pick came in.
          */}
          <div className="grid gap-4 mb-4" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
            <Stat
              label="RPS"
              value={summary.rps.toFixed(4)}
              sub={`vs ${summary.rps_base_rate.toFixed(4)} guessing · lower is better`}
              color="#00e676"
            />
            <Stat
              label="Accuracy"
              value={pct(summary.accuracy)}
              sub={`over ${summary.n.toLocaleString()} matches`}
              color={summary.accuracy >= 0.55 ? '#00e676' : summary.accuracy >= 0.45 ? '#ffea00' : '#f44336'}
            />
            <Stat
              label="Exact score"
              value={summary.exact_score != null ? pct(summary.exact_score) : '—'}
              sub="likeliest scoreline, exactly right"
              color="#a78bfa"
            />
            <Stat
              label="Log loss"
              value={summary.log_loss.toFixed(4)}
              sub={`Brier ${summary.brier.toFixed(4)}`}
              color="#60a5fa"
            />
          </div>

          <ModelVersusMarket summary={summary} confident={confident.length} confidentHits={confidentHits} />
          <ModelVersions />

          <Explainer summary={summary} />

          {/* Domestic and European kept apart. Pooling them would overstate the
              European ties, which keep about half the model's usual edge. */}
          {Object.keys(summary.by_competition ?? {}).length > 1 && (
            <div
              className="rounded-xl p-5 mb-6"
              style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
            >
              <div
                className="text-xs font-display font-bold uppercase tracking-wider mb-1"
                style={{ color: 'var(--muted-foreground)' }}
              >
                By competition
              </div>
              <p className="text-xs mb-3" style={{ color: 'var(--muted-foreground)' }}>
                Reported separately rather than pooled: a cross-league tie is a harder
                prediction, and one blended figure would flatter it.
              </p>
              <div className="flex flex-wrap gap-x-8 gap-y-3">
                {Object.entries(summary.by_competition)
                  .sort((a, b) => b[1].n - a[1].n)
                  .map(([code, row]) => (
                    <div key={code}>
                      <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                        {code === 'domestic' ? 'Domestic leagues' : competitionLabel(code)}
                        {' · '}{row.n} settled
                      </div>
                      <div className="font-data font-bold text-sm" style={{ color: 'var(--foreground)' }}>
                        {pct(row.accuracy)}
                        <span className="text-xs font-normal ml-2" style={{ color: 'var(--muted-foreground)' }}>
                          RPS {row.rps.toFixed(4)} vs {row.rps_base_rate.toFixed(4)} base
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {Object.keys(summary.by_league).length > 1 && (
            <div
              className="rounded-xl p-5 mb-6"
              style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
            >
              <div
                className="text-xs font-display font-bold uppercase tracking-wider mb-3"
                style={{ color: 'var(--muted-foreground)' }}
              >
                By league
              </div>
              {/*
                RPS beside accuracy, and each league's own base rate beside
                that. Leagues differ in how often the home side wins, so an
                accuracy of 54% means different things in Serie A and Ligue 1;
                without the base rate the column is not comparable across rows.
              */}
              <table className="w-full text-sm">
                <thead>
                  <tr style={{ color: 'var(--muted-foreground)' }}>
                    <th className="text-left font-normal pb-1 text-xs">League</th>
                    <th className="text-right font-normal pb-1 text-xs">Matches</th>
                    <th className="text-right font-normal pb-1 text-xs">Accuracy</th>
                    <th className="text-right font-normal pb-1 text-xs">RPS</th>
                    <th className="text-right font-normal pb-1 text-xs">vs guessing</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary.by_league)
                    .sort((a, b) => a[1].rps - b[1].rps)
                    .map(([code, v]) => (
                      <tr key={code}>
                        <td className="py-1" style={{ color: 'var(--foreground)' }}>
                          {competitionLabel(code)}
                        </td>
                        <td className="py-1 text-right font-data" style={{ color: 'var(--muted-foreground)' }}>
                          {v.n.toLocaleString()}
                        </td>
                        <td className="py-1 text-right font-data" style={{ color: 'var(--foreground)' }}>
                          {pct(v.accuracy)}
                        </td>
                        <td className="py-1 text-right font-data font-bold" style={{ color: '#00e676' }}>
                          {v.rps.toFixed(4)}
                        </td>
                        <td className="py-1 text-right font-data" style={{ color: 'var(--muted-foreground)' }}>
                          {(v.rps_base_rate - v.rps >= 0 ? '−' : '+')}
                          {Math.abs(v.rps_base_rate - v.rps).toFixed(4)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      <div className="flex flex-col gap-3">
        {matches.map((m, i) => (
          <ResultRow key={`${m.league}-${m.date}-${m.home_team}-${i}`} m={m} />
        ))}
      </div>

      {data && matches.length < data.total && (
        <div className="text-center text-xs mt-4" style={{ color: 'var(--muted-foreground)' }}>
          Showing the {matches.length} most recent of {data.total.toLocaleString()}.
        </div>
      )}
    </div>
  );
}
