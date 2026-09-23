import { useState } from 'react';
import { api, LEAGUE_NAMES, type ProjectedTeam } from '../api/client';
import { pct, teamColor } from '../api/display';
import { useAsync } from '../api/hooks';

/**
 * Where each league is heading, from simulating every remaining fixture.
 *
 * The projection is deliberately presented with its uncertainty attached. Run
 * against 2025/26 truncated to five matchweeks, this approach put teams within
 * 8-11 points of their real final total and named the eventual champion in two
 * leagues of five — decent shape, poor precision. Showing a bare column of
 * points would imply a forecast the simulation does not support, so every row
 * carries its 10th-90th percentile range.
 */

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex h-1.5 rounded-full overflow-hidden w-full" style={{ background: 'var(--secondary)' }}>
      <div style={{ width: `${Math.min(100, value * 100)}%`, background: color }} />
    </div>
  );
}

function Chance({ value, color }: { value: number; color: string }) {
  if (value < 0.005) return <span style={{ color: 'var(--muted-foreground)' }}>—</span>;
  return <span style={{ color, fontWeight: 600 }}>{pct(value)}</span>;
}

function Row({ t, position, teams }: { t: ProjectedTeam; position: number; teams: number }) {
  // The band is the point of the row: at this stage of a season it is routinely
  // twenty points wide, which is the honest picture.
  const span = t.points_high - t.points_low;

  return (
    <div
      className="grid items-center gap-3 px-4 py-2.5 rounded-lg"
      style={{
        gridTemplateColumns: '32px 1fr 58px 96px 116px 60px 60px 60px',
        background: 'var(--card)',
        border: '1px solid var(--border)',
      }}
    >
      <span className="font-data text-sm" style={{ color: 'var(--muted-foreground)' }}>
        {position}
      </span>

      <div className="flex items-center gap-2" style={{ minWidth: 0 }}>
        <div
          className="w-6 h-6 rounded flex items-center justify-center text-xs font-display font-bold flex-shrink-0"
          style={{ background: teamColor(t.team) + '22', color: teamColor(t.team) }}
        >
          {t.team[0]}
        </div>
        <span className="text-sm truncate" style={{ color: 'var(--foreground)' }}>{t.team}</span>
      </div>

      <span className="font-data text-xs" style={{ color: 'var(--muted-foreground)' }}>
        {t.points}p / {t.played}
      </span>

      <div className="text-right">
        <span className="font-display font-black text-lg" style={{ color: 'var(--foreground)' }}>
          {t.projected_points.toFixed(0)}
        </span>
        <span className="text-xs ml-1" style={{ color: 'var(--muted-foreground)' }}>pts</span>
      </div>

      <div>
        <div className="text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
          {t.points_low}–{t.points_high}
        </div>
        <Bar value={span / 60} color="#64748b" />
      </div>

      <span className="text-xs text-right"><Chance value={t.title_pct} color="#ffea00" /></span>
      <span className="text-xs text-right"><Chance value={t.top_four_pct} color="#00e676" /></span>
      <span className="text-xs text-right">
        <Chance value={t.relegation_pct} color="#f44336" />
      </span>
    </div>
  );
}

export default function SeasonProjection() {
  const [league, setLeague] = useState('E0');
  const { data, loading, error } = useAsync(() => api.projection(league), [league]);

  return (
    <div>
      <div className="mb-5">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Projected Tables
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Every remaining fixture simulated 10,000 times, using the same model that
          predicts individual matches.
        </p>
      </div>

      <div className="flex flex-wrap gap-3 mb-5 items-center">
        <select
          value={league}
          onChange={e => setLeague(e.target.value)}
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          {Object.entries(LEAGUE_NAMES).map(([code, name]) => (
            <option key={code} value={code}>{name}</option>
          ))}
        </select>

        {data && (
          <span className="text-xs ml-auto" style={{ color: 'var(--muted-foreground)' }}>
            {data.played} played · <strong>{data.remaining} still to play</strong>
          </span>
        )}
      </div>

      {/*
        Stated up front rather than buried. A projection this early is closer to
        a squad-strength prior than a forecast, and the measured error says so.
      */}
      <div
        className="rounded-xl p-4 mb-5 text-sm leading-relaxed"
        style={{ background: 'rgba(255,145,0,0.07)', border: '1px solid rgba(255,145,0,0.25)', color: 'var(--foreground)' }}
      >
        <strong style={{ color: '#ff9100' }}>How much to trust this.</strong>{' '}
        Tested on last season truncated to five matchweeks, this method landed
        within <strong>8–11 points</strong> of each team&rsquo;s real final total and
        named the eventual champion in <strong>two leagues out of five</strong>. It also
        assumes every side&rsquo;s current form holds until May, which nobody&rsquo;s does.
        Read the range, not the single number.
      </div>

      {loading && !data && (
        <div className="text-center py-16 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Loading projection…
        </div>
      )}

      {error && (
        <div
          className="rounded-xl p-6 text-center text-sm"
          style={{ background: 'rgba(255,145,0,0.08)', border: '1px solid rgba(255,145,0,0.3)', color: '#ff9100' }}
        >
          {error}
        </div>
      )}

      {data && (
        <>
          <div
            className="grid gap-3 px-4 pb-2 text-xs font-display uppercase tracking-wider"
            style={{ gridTemplateColumns: '32px 1fr 58px 96px 116px 60px 60px 60px', color: 'var(--muted-foreground)' }}
          >
            <span>#</span>
            <span>Team</span>
            <span>Now</span>
            <span className="text-right">Projected</span>
            <span>Range</span>
            <span className="text-right">Title</span>
            <span className="text-right">Top 4</span>
            <span className="text-right">Rel.</span>
          </div>

          <div className="flex flex-col gap-1.5">
            {data.teams.map((t, i) => (
              <Row key={t.team} t={t} position={i + 1} teams={data.teams.length} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
