import { useState } from 'react';
import { api, LEAGUE_NAMES, type MatchResult } from '../api/client';
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
          {LEAGUE_NAMES[m.league] ?? m.league}
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

        <div className="flex flex-col items-center flex-shrink-0" style={{ minWidth: 86 }}>
          <div className="font-display font-black text-xl" style={{ color: 'var(--foreground)' }}>
            {m.actual_home_goals}–{m.actual_away_goals}
          </div>
          {m.expected_goals_home != null && (
            <div className="text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
              xG {m.expected_goals_home.toFixed(2)}–{m.expected_goals_away?.toFixed(2)}
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
        <div className="flex justify-between mt-1.5 text-xs" style={{ color: 'var(--muted-foreground)' }}>
          <span>
            Predicted{' '}
            <span style={{ color: m.correct ? '#00e676' : '#f44336', fontWeight: 500 }}>
              {label(m.predicted_outcome, m.home_team, m.away_team)}
            </span>{' '}
            at {m.confidence.toFixed(0)}%
          </span>
          <span>
            Actual{' '}
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

export default function PredictionHistory() {
  const [source, setSource] = useState<Source>('backtest');
  const [league, setLeague] = useState('All');

  const { data, loading, error, reload } = useAsync(
    () => api.history(source, league === 'All' ? undefined : league, 60),
    [source, league],
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
          value={league}
          onChange={e => setLeague(e.target.value)}
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          <option value="All">All leagues</option>
          {Object.entries(LEAGUE_NAMES).map(([code, name]) => (
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
              The live record only counts predictions made before kickoff, so it fills up a few
              matches at a time.
            </div>
          )}
          <button
            onClick={source === 'live' ? () => setSource('backtest') : reload}
            className="px-4 py-2 rounded-lg text-sm font-display font-bold"
            style={{ background: 'rgba(0,230,118,0.15)', color: '#00e676' }}
          >
            {source === 'live' ? 'Show backtest history instead' : 'Retry'}
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
              label="RPS"
              value={summary.rps.toFixed(4)}
              sub={`vs ${summary.rps_base_rate.toFixed(4)} base rate — lower is better`}
              color={summary.rps < summary.rps_base_rate ? '#00e676' : '#f44336'}
            />
            <Stat
              label="Log loss"
              value={summary.log_loss.toFixed(3)}
              sub="penalises confident mistakes"
              color="#3b82f6"
            />
            <Stat
              label="When confident"
              value={confident.length ? pct(confidentHits / confident.length) : '—'}
              sub={`≥60% picks (${confident.length} of last ${matches.length})`}
              color="#a78bfa"
            />
          </div>

          {Object.keys(summary.by_league).length > 1 && (
            <div
              className="rounded-xl p-5 mb-6"
              style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
            >
              <div
                className="text-xs font-display font-bold uppercase tracking-wider mb-3"
                style={{ color: 'var(--muted-foreground)' }}
              >
                RPS by league — lower is better
              </div>
              <div className="flex flex-wrap gap-x-8 gap-y-2">
                {Object.entries(summary.by_league)
                  .sort((a, b) => a[1] - b[1])
                  .map(([code, rps]) => (
                    <div key={code} className="flex items-center gap-2">
                      <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                        {LEAGUE_NAMES[code] ?? code}
                      </span>
                      <span className="font-data font-bold text-sm" style={{ color: 'var(--foreground)' }}>
                        {rps.toFixed(4)}
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
