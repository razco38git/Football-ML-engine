import { useState } from 'react';
import { api, LEAGUE_NAMES, seasonLabel, type PlayerRating, type TeamStrength } from '../api/client';
import { getRatingBg, getRatingTextColor, teamColor } from '../api/display';
import { useAsync } from '../api/hooks';
import { LeagueStrengthPanel } from './LeagueStrength';

const LINES: { key: keyof TeamStrength; label: string }[] = [
  { key: 'strength_goalkeeper', label: 'GK' },
  { key: 'strength_defence', label: 'DEF' },
  { key: 'strength_midfield', label: 'MID' },
  { key: 'strength_attack', label: 'ATT' },
];

function Badge({ value, size = 34 }: { value: number | null; size?: number }) {
  if (value == null) {
    return (
      <span
        className="inline-flex items-center justify-center rounded font-display"
        style={{
          width: size, height: size * 0.78, background: 'var(--secondary)',
          color: 'var(--muted-foreground)', fontSize: size * 0.36,
        }}
      >
        –
      </span>
    );
  }
  const rounded = Math.round(value);
  return (
    <span
      className="inline-flex items-center justify-center rounded font-display font-bold"
      style={{
        width: size, height: size * 0.78, fontSize: size * 0.38,
        background: getRatingBg(rounded), color: getRatingTextColor(rounded),
      }}
    >
      {rounded}
    </span>
  );
}

function SquadPanel({ team, onClose }: { team: TeamStrength; onClose: () => void }) {
  const { data, loading, error } = useAsync(() => api.squad(team.team), [team.team]);
  const color = teamColor(team.team);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.8)' }}
      onClick={onClose}
    >
      <div
        className="rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto"
        style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
        onClick={e => e.stopPropagation()}
      >
        <div
          className="p-6 relative"
          style={{ background: 'linear-gradient(135deg, #0d1a2e 0%, #1a0d2e 100%)', borderBottom: '1px solid var(--border)' }}
        >
          <button
            onClick={onClose}
            className="absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center text-sm hover:bg-white/10"
            style={{ color: 'var(--muted-foreground)' }}
          >
            ✕
          </button>
          <div className="flex items-center gap-4">
            <div
              className="w-16 h-16 rounded-xl flex items-center justify-center text-xl font-display font-black"
              style={{ background: color + '33', border: `2px solid ${color}66`, color }}
            >
              {team.team.slice(0, 2).toUpperCase()}
            </div>
            <div className="flex-1">
              <div className="font-display font-black text-2xl" style={{ color: 'var(--foreground)' }}>
                {team.team}
              </div>
              <div className="text-sm" style={{ color: 'var(--muted-foreground)' }}>
                {LEAGUE_NAMES[team.league] ?? team.league} · {team.season.slice(0, 2)}/{team.season.slice(2)}
                {' · '}built from {team.n_players} players
              </div>
            </div>
            <Badge value={team.strength_overall} size={54} />
          </div>
        </div>

        <div className="p-6">
          <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: 'var(--muted-foreground)' }}>
            Squad by minutes played
          </div>

          {loading && (
            <div className="text-sm py-6 text-center" style={{ color: 'var(--muted-foreground)' }}>Loading…</div>
          )}
          {error && <div className="text-sm py-6 text-center" style={{ color: '#f44336' }}>{error}</div>}

          {data && (
            <div className="flex flex-col gap-1">
              {data.slice(0, 22).map((p: PlayerRating, i) => (
                <div
                  key={`${p.player}-${i}`}
                  className="flex items-center gap-3 py-1.5"
                  style={{ borderBottom: i < Math.min(data.length, 22) - 1 ? '1px solid var(--border)' : undefined }}
                >
                  <Badge value={p.rating} size={30} />
                  <span
                    className="text-xs font-display font-bold px-1.5 py-0.5 rounded"
                    style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)', minWidth: 32, textAlign: 'center' }}
                  >
                    {p.position}
                  </span>
                  <span className="text-sm flex-1 truncate" style={{ color: 'var(--foreground)' }}>{p.player}</span>
                  <span className="text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
                    {p.minutes.toLocaleString()} min
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="mt-4 pt-3 text-xs" style={{ borderTop: '1px solid var(--border)', color: 'var(--muted-foreground)' }}>
            The rating uses the likely starting eleven — the highest-minute player in each
            formation slot. Minutes are the best available evidence of who a manager picks:
            predicted line-ups for unplayed matches are not published anywhere readable.
          </div>
        </div>
      </div>
    </div>
  );
}

export default function TeamStrength() {
  const [league, setLeague] = useState('All');
  const [season, setSeason] = useState('');
  const [selected, setSelected] = useState<TeamStrength | null>(null);

  const { data, loading, error, reload } = useAsync(
    () => api.teams(league === 'All' ? undefined : league, 100, season || undefined),
    [league, season],
  );

  const rows = data?.teams ?? [];
  const best = rows[0]?.strength_overall ?? 100;
  // A squad is rated from the season's own players, so an earlier season shows
  // that season's team -- which is the point of being able to pick one, and
  // worth saying out loud rather than leaving the reader to infer it.
  const isPrevious = Boolean(
    data?.season && data.seasons.length > 0 && data.seasons[0] !== data.season,
  );

  return (
    <div>
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Team Strength
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Built from the likely starting eleven — the highest-minute player in each formation
          slot, rated and weighted by line. Click a team to see the squad behind it, or a
          league to filter to it.
        </p>
      </div>

      {/* Clicking a league filters the table below it. */}
      <LeagueStrengthPanel onPick={setLeague} />

      <div className="flex flex-wrap gap-3 mb-5 items-center">
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
        <select
          value={data?.season ?? season}
          onChange={e => setSeason(e.target.value)}
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          {(data?.seasons ?? []).map(s => (
            <option key={s} value={s}>{seasonLabel(s)}</option>
          ))}
        </select>
        {rows.length > 0 && (
          <span className="text-xs ml-auto" style={{ color: 'var(--muted-foreground)' }}>
            {rows.length} teams{isPrevious ? ` · ${seasonLabel(data!.season!)}` : ''}
          </span>
        )}
      </div>

      {error && (
        <div className="rounded-xl p-6 text-center" style={{ background: 'rgba(244,67,54,0.08)', border: '1px solid rgba(244,67,54,0.25)' }}>
          <div className="text-sm mb-3" style={{ color: '#f44336' }}>{error}</div>
          <button
            onClick={reload}
            className="px-4 py-2 rounded-lg text-sm font-display font-bold"
            style={{ background: 'rgba(244,67,54,0.15)', color: '#f44336' }}
          >
            Retry
          </button>
        </div>
      )}

      {loading && !data && (
        <div className="text-center py-16 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Loading team ratings…
        </div>
      )}

      {data && (
        <div className="flex flex-col gap-2">
          {rows.map((t, i) => (
            <div
              key={`${t.team}-${i}`}
              onClick={() => setSelected(t)}
              className="rounded-xl p-4 cursor-pointer transition-colors hover:bg-white/5"
              style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
            >
              <div className="flex items-center gap-4">
                <span className="text-xs font-data w-6 text-right" style={{ color: 'var(--muted-foreground)' }}>
                  {i + 1}
                </span>
                <div
                  className="w-9 h-9 rounded-lg flex items-center justify-center text-xs font-display font-bold flex-shrink-0"
                  style={{ background: teamColor(t.team) + '22', color: teamColor(t.team) }}
                >
                  {t.team.slice(0, 2).toUpperCase()}
                </div>
                <div style={{ minWidth: 0, flex: '0 0 180px' }}>
                  <div className="font-display font-bold text-base truncate" style={{ color: 'var(--foreground)' }}>
                    {t.team}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                    {LEAGUE_NAMES[t.league] ?? t.league}
                  </div>
                </div>

                <div className="flex-1 min-w-0">
                  <div className="h-2 rounded-full" style={{ background: 'var(--secondary)' }}>
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(t.strength_overall / best) * 100}%`,
                        background: 'linear-gradient(90deg, #00e676, #3b82f6)',
                      }}
                    />
                  </div>
                </div>

                <div className="flex items-center gap-3 flex-shrink-0">
                  {LINES.map(line => (
                    <div key={String(line.key)} className="flex flex-col items-center gap-0.5">
                      <Badge value={t[line.key] as number | null} size={30} />
                      <span style={{ fontSize: 9, color: 'var(--muted-foreground)' }}>{line.label}</span>
                    </div>
                  ))}
                  <div className="flex flex-col items-center gap-0.5 ml-1">
                    <Badge value={t.strength_overall} size={40} />
                    <span style={{ fontSize: 9, color: 'var(--muted-foreground)' }}>OVR</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {selected && <SquadPanel team={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
