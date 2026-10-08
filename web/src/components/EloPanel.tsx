/**
 * The Elo leaderboards, and the difference between Elo and squad strength.
 *
 * These two numbers sit on the same page and disagree often, so the panel leads
 * with *why* rather than making the reader reconcile them. The short version:
 * squad strength asks how good the players are, Elo asks how good the results
 * have been. A club that buys well and loses rates high on one and low on the
 * other, and neither is wrong.
 *
 * Two boards rather than one because "best now" and "best ever" are different
 * questions with mostly different answers — the all-time list is a tour of the
 * last decade's great sides, and burying it inside the current table would hide
 * that.
 */
import { api, LEAGUE_NAMES, seasonLabel, type EloRankEntry } from '../api/client';
import { useAsync } from '../api/hooks';

/** Rating relative to the 1500 everyone starts at, which is what a reader can act on. */
function fromStart(elo: number, start: number): string {
  const d = Math.round(elo - start);
  return d >= 0 ? `+${d}` : `−${Math.abs(d)}`;
}

function Board({
  title,
  caption,
  rows,
  start,
  showSeason,
}: {
  title: string;
  caption: string;
  rows: EloRankEntry[];
  start: number;
  showSeason: boolean;
}) {
  const top = rows[0]?.elo ?? start;
  return (
    <div className="flex-1 min-w-[260px]">
      <div
        className="text-xs font-display font-bold uppercase tracking-wider"
        style={{ color: 'var(--muted-foreground)' }}
      >
        {title}
      </div>
      <div className="text-xs mb-3 mt-0.5" style={{ color: 'var(--muted-foreground)', opacity: 0.8 }}>
        {caption}
      </div>
      <div className="flex flex-col gap-1">
        {rows.map((r, i) => (
          <div key={`${r.team}-${r.date}`} className="flex items-center gap-2 text-sm">
            <span
              className="font-data w-5 text-right shrink-0"
              style={{ color: 'var(--muted-foreground)', fontSize: 11 }}
            >
              {i + 1}
            </span>
            <span className="truncate flex-1" style={{ color: 'var(--foreground)' }}>
              {r.team}
            </span>
            {showSeason && (
              <span
                className="font-data shrink-0"
                style={{ color: 'var(--muted-foreground)', fontSize: 11 }}
                title={r.date}
              >
                {seasonLabel(r.season)}
              </span>
            )}
            <span
              className="font-data shrink-0 text-xs"
              style={{ color: 'var(--muted-foreground)' }}
              title={`${LEAGUE_NAMES[r.league] ?? r.league} · rating going into ${r.date}`}
            >
              {fromStart(r.elo, start)}
            </span>
            {/* Bar is relative to the leader, so the gaps are readable rather than
                all four digits looking alike. */}
            <span
              className="hidden sm:block h-1.5 rounded-full shrink-0"
              style={{
                width: 54,
                background: 'var(--secondary)',
              }}
            >
              <span
                className="block h-full rounded-full"
                style={{
                  width: `${Math.max(6, (100 * (r.elo - start)) / Math.max(1, top - start))}%`,
                  background: '#00e676',
                }}
              />
            </span>
            <span
              className="font-data font-bold shrink-0 w-11 text-right"
              style={{ color: 'var(--foreground)' }}
            >
              {Math.round(r.elo)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function EloPanel() {
  const { data, loading, error } = useAsync(() => api.eloRanking(12), []);

  return (
    <div
      className="rounded-xl p-5 mb-6"
      style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
    >
      <div className="font-display font-bold text-lg mb-1" style={{ color: 'var(--foreground)' }}>
        Elo — the other rating
      </div>
      <p className="text-sm mb-1" style={{ color: 'var(--muted-foreground)' }}>
        The squad numbers on this page ask <strong style={{ color: 'var(--foreground)' }}>how
        good are the players</strong>, from minutes and performance, once a season. Elo asks{' '}
        <strong style={{ color: 'var(--foreground)' }}>how good have the results been</strong>.
        It knows nothing about who is in the squad: it starts everyone at 1500 and moves only
        when a result disagrees with what the rating expected, so beating a strong side is
        worth more than beating a weak one.
      </p>
      <p className="text-sm mb-4" style={{ color: 'var(--muted-foreground)' }}>
        They disagree often, and neither is wrong — a club that has bought well but lost rates
        high on squad and low on Elo. Two differences worth knowing: Elo updates{' '}
        <strong style={{ color: 'var(--foreground)' }}>every match</strong> where squad strength
        is fixed from one August to the next, and Elo includes European nights where squad
        strength is purely domestic. Elo is also the single largest input to the match
        predictor, so it is the closest thing here to what the model currently thinks.
      </p>

      {loading && (
        <div className="text-sm py-4" style={{ color: 'var(--muted-foreground)' }}>Loading…</div>
      )}
      {error && <div className="text-sm py-4" style={{ color: '#f44336' }}>{error}</div>}

      {data && (
        <>
          <div className="flex flex-wrap gap-8">
            <Board
              title="Highest now"
              caption="Rating going into each side's most recent match"
              rows={data.current}
              start={data.start}
              showSeason={false}
            />
            <Board
              title="Highest ever"
              caption="Best any side has rated since 2010/11"
              rows={data.peak}
              start={data.start}
              showSeason
            />
          </div>
          <p className="text-xs mt-4" style={{ color: 'var(--muted-foreground)', opacity: 0.8 }}>
            Every rating shown is the one a side carried <em>into</em> that match, which is
            exactly what the model was given — so a current rating has not yet seen the result
            of the match it precedes. The number beside each club is its distance from the
            1500 everyone starts at.
          </p>
        </>
      )}
    </div>
  );
}
