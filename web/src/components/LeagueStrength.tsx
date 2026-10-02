import { api, LEAGUE_NAMES, type LeagueStrength } from '../api/client';
import { useAsync } from '../api/hooks';

const ACCENT = '#00e676';

/**
 * Why these numbers are comparable at all, in one place so both callers say
 * the same thing. A rating is not league-relative — EA's overall is a global
 * scale, and the performance percentiles rank a player against everyone in his
 * role across all five leagues — so the means can be read side by side.
 */
export const COMPARABILITY_NOTE =
  'Ratings are not league-relative: EA’s overall is a global scale and performance ' +
  'is ranked against every player in the same role across all five leagues. One ' +
  'caveat — per-90 output is easier to accumulate against weaker opponents, which ' +
  'flatters the weaker leagues, so the real gaps are if anything wider than these.';

/** Scale the bars against the range on show, not against zero. */
function bounds(rows: LeagueStrength[]) {
  const means = rows.map(r => r.mean_strength);
  const low = Math.min(...means) - 1.5;
  const high = Math.max(...means) + 0.5;
  return { low, high, span: Math.max(high - low, 0.1) };
}

/** The full five-league comparison. */
export function LeagueStrengthPanel({ onPick }: { onPick?: (league: string) => void }) {
  const { data, loading, error } = useAsync(() => api.leagueStrength(), []);

  if (loading || error || !data || data.length === 0) return null;

  const { low, span } = bounds(data);
  const top = data[0].mean_strength;

  return (
    <div
      className="rounded-xl border p-4 mb-6"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-baseline justify-between mb-3 gap-3 flex-wrap">
        <span
          className="text-xs font-display font-bold tracking-widest uppercase"
          style={{ color: 'var(--muted-foreground)' }}
        >
          League strength · average squad rating
        </span>
        <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
          {data[0].season.slice(0, 2)}/{data[0].season.slice(2)}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {data.map(row => {
          const width = ((row.mean_strength - low) / span) * 100;
          const behind = top - row.mean_strength;
          return (
            <button
              key={row.league}
              onClick={onPick ? () => onPick(row.league) : undefined}
              className="grid items-center gap-3 text-left w-full"
              style={{
                gridTemplateColumns: 'minmax(96px, 1fr) minmax(0, 3fr) auto',
                cursor: onPick ? 'pointer' : 'default',
                background: 'transparent',
              }}
              title={
                `${row.strongest_team} ${row.strongest_strength} · ` +
                `${row.weakest_team} ${row.weakest_strength} · ` +
                `spread ${row.spread} across ${row.n_teams} clubs`
              }
            >
              <span
                className="text-sm font-display font-bold truncate"
                style={{ color: 'var(--foreground)' }}
              >
                {LEAGUE_NAMES[row.league] ?? row.league}
              </span>

              <span
                className="h-2 rounded-full overflow-hidden"
                style={{ background: 'rgba(255,255,255,0.06)' }}
              >
                <span
                  className="block h-full rounded-full"
                  style={{
                    width: `${Math.max(width, 2)}%`,
                    background: `linear-gradient(90deg, ${ACCENT}, #00b0ff)`,
                  }}
                />
              </span>

              <span className="text-sm font-display font-bold tabular-nums whitespace-nowrap">
                <span style={{ color: 'var(--foreground)' }}>{row.mean_strength.toFixed(1)}</span>
                <span className="text-xs ml-2" style={{ color: 'var(--muted-foreground)' }}>
                  {behind < 0.05 ? '—' : `−${behind.toFixed(1)}`}
                </span>
              </span>
            </button>
          );
        })}
      </div>

      <p className="mt-3 text-xs leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
        Mean squad rating across each league's clubs, with the gap to the strongest league.
        Hover a row for its best and worst side and how far apart its clubs are.{' '}
        {COMPARABILITY_NOTE}
      </p>
    </div>
  );
}

/** Just the two leagues in play, for a cross-league What-If pairing. */
export function LeagueGap({ home, away }: { home: string; away: string }) {
  const { data } = useAsync(() => api.leagueStrength(), []);
  if (!data || home === away) return null;

  const a = data.find(r => r.league === home);
  const b = data.find(r => r.league === away);
  if (!a || !b) return null;

  const gap = Math.abs(a.mean_strength - b.mean_strength);
  const stronger = a.mean_strength >= b.mean_strength ? a : b;

  return (
    <div
      className="rounded-xl border p-4 mb-4"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center justify-center gap-4 flex-wrap text-center">
        {[a, b].map(row => (
          <div key={row.league} className="min-w-[120px]">
            <div className="text-xs uppercase tracking-widest" style={{ color: 'var(--muted-foreground)' }}>
              {LEAGUE_NAMES[row.league] ?? row.league}
            </div>
            <div
              className="font-display font-black text-2xl tabular-nums"
              style={{ color: row.league === stronger.league ? ACCENT : 'var(--foreground)' }}
            >
              {row.mean_strength.toFixed(1)}
            </div>
            <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
              avg squad · {row.n_teams} clubs
            </div>
          </div>
        ))}
      </div>

      <p className="mt-3 text-xs leading-relaxed text-center" style={{ color: 'var(--muted-foreground)' }}>
        {gap < 0.05 ? (
          <>These two leagues rate level on average.</>
        ) : (
          <>
            {LEAGUE_NAMES[stronger.league] ?? stronger.league} rates{' '}
            <strong style={{ color: ACCENT }}>{gap.toFixed(1)}</strong> higher per club on
            average — worth remembering, since the model has never seen these divisions meet.
          </>
        )}
      </p>
    </div>
  );
}
