import { useState } from 'react';
import { api, COMPETITION_NAMES, LEAGUE_NAMES, competitionLabel, type Accuracy, type MatchResult } from '../api/client';
import { formatDate, pct, teamColor } from '../api/display';
import { useAsync } from '../api/hooks';

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

function ResultBadge({ correct }: { correct: boolean }) {
  return (
    <span
      className="text-xs font-display font-bold px-2 py-0.5 rounded"
      style={{
        background: correct ? 'rgba(0,230,118,0.15)' : 'rgba(244,67,54,0.15)',
        color: correct ? '#00e676' : '#f44336',
      }}
    >
      {correct ? '✓' : '✗'}
    </span>
  );
}

function OutcomeBar({ m }: { m: MatchResult }) {
  const segments = [
    { value: m.prob_home_win, color: '#00e676' },
    { value: m.prob_draw, color: '#64748b' },
    { value: m.prob_away_win, color: '#3b82f6' },
  ];
  return (
    <div className="flex h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--secondary)' }}>
      {segments.map((s, i) => (
        <div key={i} style={{ width: `${s.value * 100}%`, background: s.color }} />
      ))}
    </div>
  );
}

function ResultRow({ m }: { m: MatchResult }) {
  const label = (o: string, home: string, away: string) =>
    o === 'H' ? home : o === 'A' ? away : 'Draw';

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
          <div
            className="text-xs uppercase tracking-wider"
            style={{ color: 'var(--muted-foreground)', fontSize: 9 }}
          >
            final score
          </div>
          <div className="font-display font-black text-xl" style={{ color: 'var(--foreground)' }}>
            {m.actual_home_goals}–{m.actual_away_goals}
          </div>
          {m.expected_goals_home != null && (
            <div
              className="text-xs font-data"
              style={{ color: 'var(--muted-foreground)' }}
              title="Goals the model expected each side to score, from before kick-off. Not the chances they actually created."
            >
              we forecast {m.expected_goals_home.toFixed(2)}–{m.expected_goals_away?.toFixed(2)}
            </div>
          )}
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
        <OutcomeBar m={m} />
        {/*
          All three, as the Match Predictor shows them. "Predicted Villarreal at
          59%" alone leaves the other 41% unaccounted for, and whether the model
          thought the danger was a draw or an away win is the more interesting
          half of a wrong call.
        */}
        <div
          className="flex justify-between mt-1.5 text-xs font-data"
          style={{ color: 'var(--muted-foreground)' }}
        >
          <span style={{ color: m.actual_result === 'H' ? 'var(--foreground)' : undefined }}>
            {m.home_team} {pct(m.prob_home_win)}
          </span>
          <span style={{ color: m.actual_result === 'D' ? 'var(--foreground)' : undefined }}>
            Draw {pct(m.prob_draw)}
          </span>
          <span style={{ color: m.actual_result === 'A' ? 'var(--foreground)' : undefined }}>
            {m.away_team} {pct(m.prob_away_win)}
          </span>
        </div>
        <div className="flex justify-between mt-1 text-xs" style={{ color: 'var(--muted-foreground)' }}>
          <span>
            We said{' '}
            <span style={{ color: m.correct ? '#00e676' : '#f44336', fontWeight: 500 }}>
              {label(m.predicted_outcome, m.home_team, m.away_team)}
            </span>
          </span>
          <span>
            It was{' '}
            <span style={{ color: 'var(--foreground)' }}>
              {label(m.actual_result, m.home_team, m.away_team)}
            </span>
          </span>
        </div>
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
function Explainer({ summary }: { summary: Accuracy }) {
  // Only the bands with enough matches to mean anything, most confident first.
  const bands = summary.calibration
    .filter(b => b.n >= 100)
    .sort((a, b) => b.mean_predicted - a.mean_predicted)
    .slice(0, 4);

  return (
    <div
      className="rounded-xl p-5 mb-6"
      style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
    >
      <div
        className="text-xs font-display font-bold uppercase tracking-wider mb-3"
        style={{ color: 'var(--muted-foreground)' }}
      >
        What does {pct(summary.accuracy)} mean?
      </div>

      <p className="text-sm leading-relaxed" style={{ color: 'var(--foreground)' }}>
        A match has three results, so guessing at random gets{' '}
        <strong>33%</strong>. Always backing the home side gets{' '}
        <strong>{pct(summary.accuracy_base_rate)}</strong>
        {summary.accuracy_market != null && (
          <>
            , and the bookmakers&rsquo; favourite wins{' '}
            <strong>{pct(summary.accuracy_market)}</strong> of these same matches
          </>
        )}
        . Draws are the hard part: about a quarter of matches end level, but a draw
        is almost never any single most likely result, so those games are nearly
        impossible to call.
      </p>

      {bands.length > 0 && (
        <>
          <p className="text-sm leading-relaxed mt-3" style={{ color: 'var(--foreground)' }}>
            Accuracy only asks whether the top pick came in. The more useful
            question is whether the percentages can be trusted &mdash; and they can:
          </p>
          <div className="flex flex-wrap gap-x-6 gap-y-2 mt-3">
            {bands.map(b => (
              <div key={b.bin_lower} className="flex items-center gap-2">
                <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                  says {pct(b.mean_predicted)}
                </span>
                <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>&rarr;</span>
                <span className="font-data font-bold text-sm" style={{ color: '#00e676' }}>
                  {pct(b.observed_rate)}
                </span>
                <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                  ({b.n.toLocaleString()})
                </span>
              </div>
            ))}
          </div>
        </>
      )}
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
          <div className="grid gap-4 mb-6" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
            <Stat
              label="Accuracy"
              value={pct(summary.accuracy)}
              sub={`over ${summary.n.toLocaleString()} matches`}
              color={summary.accuracy >= 0.55 ? '#00e676' : summary.accuracy >= 0.45 ? '#ffea00' : '#f44336'}
            />
            <Stat
              label="Always pick home"
              value={pct(summary.accuracy_base_rate)}
              sub="the simplest possible rule"
              color="#64748b"
            />
            <Stat
              label="Bookmakers"
              value={summary.accuracy_market != null ? pct(summary.accuracy_market) : '—'}
              sub="favourite wins, same matches"
              color="#f59e0b"
            />
            <Stat
              label="When confident"
              value={confident.length ? pct(confidentHits / confident.length) : '—'}
              sub={`≥60% picks (${confident.length} of last ${matches.length})`}
              color="#a78bfa"
            />
          </div>

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

          {Object.keys(summary.by_league_accuracy).length > 1 && (
            <div
              className="rounded-xl p-5 mb-6"
              style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
            >
              <div
                className="text-xs font-display font-bold uppercase tracking-wider mb-3"
                style={{ color: 'var(--muted-foreground)' }}
              >
                Accuracy by league
              </div>
              <div className="flex flex-wrap gap-x-8 gap-y-2">
                {Object.entries(summary.by_league_accuracy)
                  .sort((a, b) => b[1] - a[1])
                  .map(([code, value]) => (
                    <div key={code} className="flex items-center gap-2">
                      <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                        {competitionLabel(code)}
                      </span>
                      <span className="font-data font-bold text-sm" style={{ color: 'var(--foreground)' }}>
                        {pct(value)}
                      </span>
                    </div>
                  ))}
              </div>
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
