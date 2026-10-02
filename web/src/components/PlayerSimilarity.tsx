import { useState } from 'react';
import { api, type SimilarPlayer, type SimilarPlayers } from '../api/client';
import { teamColor } from '../api/display';
import { useAsync } from '../api/hooks';

/**
 * Who plays like this player?
 *
 * Two answers, shown separately because they are different questions. EA's
 * attributes share one scale across outfield positions, so they survive a
 * cross-position search. Our own sub-ratings are percentiles *within* a
 * position, so widening the search drops that axis entirely rather than
 * comparing a centre-back's 80 with a winger's 80.
 *
 * Both scores are anchored so that 50 means "no more alike than two random
 * players in this position". Either can be missing, and missing is not zero —
 * A small share of rated players have no EA entry at all — 1.6% of the
 * current season, 3.9% across all of them.
 */

/** Short labels for the radar, keyed by the API's attribute names. */
const ATTRIBUTE_LABELS: Record<string, string> = {
  pace: 'PAC',
  shooting: 'SHO',
  passing: 'PAS',
  dribbling: 'DRI',
  defending: 'DEF',
  physical: 'PHY',
  gk_diving: 'DIV',
  gk_handling: 'HAN',
  gk_kicking: 'KIC',
  gk_positioning: 'POS',
  gk_reflexes: 'REF',
};

const SUB_LABELS: Record<string, string> = {
  sub_finishing: 'Finishing',
  sub_creation: 'Creation',
  sub_involvement: 'Involvement',
  sub_volume: 'Volume',
  sub_defending: 'Defending',
  sub_shot_stopping: 'Shot stopping',
  sub_reliability: 'Reliability',
  sub_workload: 'Workload',
  sub_penalties: 'Penalties',
};

const SUBJECT_COLOR = '#00e676';
const CANDIDATE_COLOR = '#00b0ff';

function scoreColor(score: number): string {
  // Banded against what the scale means: 50 is a random pair, so anything
  // under it is actively dissimilar and should not look encouraging.
  if (score >= 80) return '#00e676';
  if (score >= 65) return '#76ff03';
  if (score >= 50) return '#ffea00';
  return '#ff9100';
}

function Score({ label, value, hint }: { label: string; value: number | null; hint: string }) {
  if (value === null) {
    return (
      <div className="text-center" title={hint}>
        <div className="font-data font-bold text-sm" style={{ color: 'var(--muted-foreground)' }}>
          n/a
        </div>
        <div className="text-xs uppercase tracking-wider" style={{ color: 'var(--muted-foreground)', fontSize: 9 }}>
          {label}
        </div>
      </div>
    );
  }
  return (
    <div className="text-center" title={hint}>
      <div className="font-data font-bold text-sm" style={{ color: scoreColor(value) }}>
        {value.toFixed(0)}
      </div>
      <div className="text-xs uppercase tracking-wider" style={{ color: 'var(--muted-foreground)', fontSize: 9 }}>
        {label}
      </div>
    </div>
  );
}

function SimilarityCircle({ score }: { score: number | null }) {
  const r = 18;
  const circ = 2 * Math.PI * r;
  const value = score ?? 0;
  const dash = (value / 100) * circ;
  const color = score === null ? '#64748b' : scoreColor(score);

  return (
    <div className="relative flex items-center justify-center flex-shrink-0" style={{ width: 48, height: 48 }}>
      <svg width="48" height="48" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="24" cy="24" r={r} fill="none" stroke="#1e2d45" strokeWidth="3" />
        <circle
          cx="24" cy="24" r={r} fill="none"
          stroke={color} strokeWidth="3"
          strokeDasharray={`${dash} ${circ - dash}`}
          strokeLinecap="round"
        />
      </svg>
      <span className="absolute font-data font-bold text-sm" style={{ color }}>
        {score === null ? '–' : score.toFixed(0)}
      </span>
    </div>
  );
}

/**
 * The attribute radar, optionally with a second player laid over the first.
 *
 * Takes the attribute names from the API rather than hardcoding the six, so a
 * goalkeeper renders diving/handling/kicking/positioning/reflexes without a
 * separate component.
 */
function RadarChart({
  names,
  subject,
  candidate,
}: {
  names: string[];
  subject: Record<string, number>;
  candidate?: Record<string, number>;
}) {
  const n = names.length;
  const cx = 80;
  const cy = 80;
  const r = 60;

  const angleAt = (i: number) => (i / n) * 2 * Math.PI - Math.PI / 2;

  const shape = (values: Record<string, number>) =>
    names
      .map((name, i) => {
        const angle = angleAt(i);
        const fr = ((values[name] ?? 0) / 99) * r;
        return `${cx + fr * Math.cos(angle)},${cy + fr * Math.sin(angle)}`;
      })
      .join(' ');

  const grids = [0.25, 0.5, 0.75, 1].map(f =>
    names
      .map((_, i) => `${cx + f * r * Math.cos(angleAt(i))},${cy + f * r * Math.sin(angleAt(i))}`)
      .join(' ')
  );

  return (
    <svg viewBox="0 0 160 160" className="w-full max-w-48">
      {grids.map((pts, i) => (
        <polygon key={i} points={pts} fill="none" stroke="#1e2d45" strokeWidth="0.5" />
      ))}
      {names.map((_, i) => (
        <line
          key={i}
          x1={cx} y1={cy}
          x2={cx + r * Math.cos(angleAt(i))} y2={cy + r * Math.sin(angleAt(i))}
          stroke="#1e2d45" strokeWidth="0.5"
        />
      ))}

      {candidate && (
        <polygon
          points={shape(candidate)}
          fill="rgba(0,176,255,0.12)"
          stroke={CANDIDATE_COLOR}
          strokeWidth="1.5"
          strokeDasharray="3 2"
        />
      )}
      <polygon points={shape(subject)} fill="rgba(0,230,118,0.15)" stroke={SUBJECT_COLOR} strokeWidth="1.5" />

      {names.map((name, i) => {
        const angle = angleAt(i);
        const fr = ((subject[name] ?? 0) / 99) * r;
        return (
          <g key={name}>
            <circle cx={cx + fr * Math.cos(angle)} cy={cy + fr * Math.sin(angle)} r="2.5" fill={SUBJECT_COLOR} />
            <text
              x={cx + (r + 16) * Math.cos(angle)}
              y={cy + (r + 16) * Math.sin(angle)}
              textAnchor="middle" dominantBaseline="middle"
              fontSize="8" fill="#94a3b8" fontFamily="'Barlow Condensed', sans-serif" fontWeight="700"
            >
              {ATTRIBUTE_LABELS[name] ?? name.slice(0, 3).toUpperCase()}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** The attribute-by-attribute gaps, so a score can be checked rather than trusted. */
function Gaps({
  names,
  subject,
  candidate,
  labels,
}: {
  names: string[];
  subject: Record<string, number>;
  candidate: Record<string, number>;
  labels: Record<string, string>;
}) {
  const shared = names.filter(n => subject[n] !== undefined && candidate[n] !== undefined);
  if (shared.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1.5">
      {shared.map(name => {
        const gap = candidate[name] - subject[name];
        return (
          <div
            key={name}
            className="rounded px-2 py-1 text-xs font-data"
            style={{ background: 'var(--secondary)', border: '1px solid var(--border)' }}
          >
            <span style={{ color: 'var(--muted-foreground)' }}>{labels[name] ?? name} </span>
            <span style={{ color: 'var(--foreground)' }}>{candidate[name].toFixed(0)}</span>
            <span style={{ color: gap === 0 ? 'var(--muted-foreground)' : gap > 0 ? '#00e676' : '#ff9100' }}>
              {' '}{gap > 0 ? '+' : ''}{gap.toFixed(0)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ResultCard({
  match,
  data,
  onSelect,
  expanded,
  onToggle,
}: {
  match: SimilarPlayer;
  data: SimilarPlayers;
  onSelect: () => void;
  expanded: boolean;
  onToggle: () => void;
}) {
  const subLabels = Object.keys(data.player_sub_ratings);

  return (
    <div
      className="rounded-xl p-4"
      style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
    >
      <div className="flex items-center gap-3">
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center text-base font-display font-black flex-shrink-0"
          style={{ background: teamColor(match.team) + '22', color: teamColor(match.team) }}
        >
          {match.player[0]}
        </div>

        <div className="flex-1 min-w-0">
          <button
            onClick={onSelect}
            className="font-display font-bold text-sm truncate block text-left"
            style={{ color: 'var(--foreground)', background: 'transparent' }}
          >
            {match.player}
          </button>
          <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
            {match.position} · {match.team} · rated {match.rating ?? '–'}
          </div>
        </div>

        <div className="flex items-center gap-3 flex-shrink-0">
          <Score
            label="EA"
            value={match.fifa_similarity}
            hint={
              match.fifa_similarity === null
                ? 'No EA entry for this player, so the attribute axis cannot be measured.'
                : "Similarity across EA's attributes."
            }
          />
          <Score
            label="Pctile"
            value={match.percentile_similarity}
            hint={
              match.percentile_similarity === null
                ? 'Our sub-ratings are ranked within a position, so they are not compared across positions.'
                : 'Similarity across our own percentile sub-ratings.'
            }
          />
          <SimilarityCircle score={match.combined} />
          <button
            onClick={onToggle}
            className="text-xs px-2 py-1 rounded font-display font-bold"
            style={{ background: 'var(--secondary)', border: '1px solid var(--border)', color: 'var(--muted-foreground)' }}
          >
            {expanded ? 'Hide' : 'Why'}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 grid gap-4" style={{ borderTop: '1px solid var(--border)', gridTemplateColumns: 'auto 1fr' }}>
          <RadarChart
            names={data.attribute_names}
            subject={data.player_attributes}
            candidate={match.attributes}
          />
          <div className="flex flex-col gap-3 min-w-0">
            <div>
              <div className="text-xs uppercase tracking-wider mb-1.5" style={{ color: 'var(--muted-foreground)' }}>
                EA attributes, against {data.player.player}
              </div>
              {Object.keys(match.attributes).length > 0 ? (
                <Gaps
                  names={data.attribute_names}
                  subject={data.player_attributes}
                  candidate={match.attributes}
                  labels={ATTRIBUTE_LABELS}
                />
              ) : (
                <p className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                  No EA entry for this player, so there is nothing to compare on this axis.
                </p>
              )}
            </div>

            <div>
              <div className="text-xs uppercase tracking-wider mb-1.5" style={{ color: 'var(--muted-foreground)' }}>
                Our percentile sub-ratings
              </div>
              {Object.keys(match.sub_ratings).length > 0 ? (
                <Gaps
                  names={subLabels}
                  subject={data.player_sub_ratings}
                  candidate={match.sub_ratings}
                  labels={SUB_LABELS}
                />
              ) : !data.same_role ? (
                <p className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                  Not shown while searching across positions. These are ranks <em>within</em> a
                  position, so scoring the same-position candidates on them and everyone else
                  without would put two different scales in one ranked list.
                </p>
              ) : (
                <p className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                  Not comparable: these are ranks within a position, and this player plays a
                  different one.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function PlayerSimilarity() {
  const [name, setName] = useState('Mohamed Salah');
  const [query, setQuery] = useState('');
  const [sameRole, setSameRole] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  const { data, loading, error } = useAsync(
    () => api.similar(name, { limit: 20, same_role: sameRole }),
    [name, sameRole]
  );

  const search = useAsync(
    () => (query.length < 2 ? Promise.resolve(null) : api.players({ search: query, limit: 8 })),
    [query]
  );

  const select = (player: string) => {
    setName(player);
    setQuery('');
    setExpanded(null);
  };

  const subject = data?.player;
  const keeper = subject?.position === 'GK';
  const noEaEntry = data && Object.keys(data.player_attributes).length === 0;

  return (
    <div>
      <div className="mb-5">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Player Similarity
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Who plays like this player, measured two ways: EA&rsquo;s attributes, and our own
          percentile sub-ratings.
        </p>
      </div>

      {/* Search */}
      <div
        className="rounded-xl p-5 mb-5"
        style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
      >
        <div className="flex items-center gap-4 flex-wrap">
          <div className="relative flex-1 min-w-48">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search player…"
              className="w-full rounded-lg px-3 py-2 text-sm outline-none"
              style={{ background: 'var(--secondary)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
            />
            {search.data && search.data.players.length > 0 && (
              <div
                className="absolute top-full left-0 right-0 z-10 mt-1 rounded-lg overflow-hidden max-h-56 overflow-y-auto"
                style={{ background: 'var(--card)', border: '1px solid var(--border)', boxShadow: '0 8px 32px rgba(0,0,0,0.5)' }}
              >
                {search.data.players.map(p => (
                  <button
                    key={`${p.player}-${p.team}`}
                    className="px-3 py-2.5 flex items-center gap-2 w-full text-left"
                    style={{ borderBottom: '1px solid var(--border)', background: 'transparent' }}
                    onClick={() => select(p.player)}
                  >
                    <div>
                      <div className="text-sm font-display font-bold" style={{ color: 'var(--foreground)' }}>
                        {p.player}
                      </div>
                      <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                        {p.position} · {p.team}
                      </div>
                    </div>
                    <span className="ml-auto font-data text-sm" style={{ color: 'var(--muted-foreground)' }}>
                      {p.rating ?? '–'}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: 'var(--muted-foreground)' }}>
            <input type="checkbox" checked={sameRole} onChange={e => setSameRole(e.target.checked)} />
            Same position only
          </label>
        </div>
      </div>

      {/*
        The scale is the thing a reader will misjudge, so it is stated rather
        than left to be inferred from the colours.
      */}
      <div
        className="rounded-xl p-4 mb-5 text-sm leading-relaxed"
        style={{ background: 'rgba(0,176,255,0.06)', border: '1px solid rgba(0,176,255,0.25)', color: 'var(--foreground)' }}
      >
        <strong style={{ color: CANDIDATE_COLOR }}>How to read the scores.</strong>{' '}
        <strong>50 means &ldquo;no more alike than two random players in this position&rdquo;</strong> —
        the scale is pinned to the typical gap between two players, not to the range of the
        numbers. The <strong>EA</strong> score compares EA&rsquo;s attributes, which share one
        scale across outfield positions. The <strong>Pctile</strong> score compares our own
        sub-ratings, which are ranks <em>within</em> a position, so it is blank whenever the
        comparison crosses one. A blank is &ldquo;not measured&rdquo;, never &ldquo;nothing alike&rdquo;.
      </div>

      {loading && !data && (
        <div className="text-center py-16 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Finding similar players…
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

      {data && subject && (
        <>
          {/* Subject */}
          <div
            className="rounded-xl p-6 mb-5 grid gap-6 items-center"
            style={{
              background: 'linear-gradient(135deg, #0d1a2e 0%, #1a0d2e 100%)',
              border: '1px solid var(--border)',
              gridTemplateColumns: 'auto 1fr auto',
            }}
          >
            <div
              className="w-20 h-20 rounded-xl flex items-center justify-center text-4xl font-display font-black"
              style={{ background: teamColor(subject.team) + '33', color: teamColor(subject.team) }}
            >
              {subject.player[0]}
            </div>

            <div>
              <div className="font-display font-black text-3xl leading-tight" style={{ color: 'var(--foreground)' }}>
                {subject.player.toUpperCase()}
              </div>
              <div className="text-sm mb-3" style={{ color: 'var(--muted-foreground)' }}>
                {subject.position} · {subject.team} · {subject.league} · {subject.season} ·{' '}
                {subject.minutes.toLocaleString()} min
              </div>
              <div className="flex gap-3 flex-wrap">
                <div className="rounded-lg px-3 py-2" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
                  <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: 'var(--muted-foreground)' }}>
                    Our rating
                  </div>
                  <div className="text-sm font-display font-bold text-white">{subject.rating ?? '–'}</div>
                </div>
                <div className="rounded-lg px-3 py-2" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
                  <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: 'var(--muted-foreground)' }}>
                    Compared against
                  </div>
                  <div className="text-sm font-display font-bold text-white">
                    {data.same_role ? `other ${subject.position}s` : 'every outfield position'}
                  </div>
                </div>
              </div>
            </div>

            {noEaEntry ? (
              <div className="text-xs max-w-48" style={{ color: 'var(--muted-foreground)' }}>
                No EA entry for {subject.player}
                {keeper && subject.season === '2324'
                  ? ' — the FC24 export carries no goalkeeper attributes at all.'
                  : ', so only our own sub-ratings are available.'}
              </div>
            ) : (
              <RadarChart names={data.attribute_names} subject={data.player_attributes} />
            )}
          </div>

          <div className="mb-3">
            <h3 className="font-display font-bold text-xl" style={{ color: 'var(--foreground)' }}>
              {data.results.length} most similar
            </h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--muted-foreground)' }}>
              {data.same_role
                ? `Same position, same season. Ranked on both axes combined.`
                : 'Every outfield position, ranked on EA attributes alone — our percentiles are not comparable across positions.'}
            </p>
          </div>

          {data.results.length === 0 ? (
            <div className="text-center py-12 text-sm" style={{ color: 'var(--muted-foreground)' }}>
              Nobody comparable in this season.
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {data.results.map(match => (
                <ResultCard
                  key={`${match.player}-${match.team}`}
                  match={match}
                  data={data}
                  onSelect={() => select(match.player)}
                  expanded={expanded === match.player}
                  onToggle={() => setExpanded(expanded === match.player ? null : match.player)}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
