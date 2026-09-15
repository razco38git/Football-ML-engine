import { players, getRatingBg, getRatingTextColor } from '../data/footballData';
import DemoBanner from './DemoBanner';

interface TeamData {
  name: string;
  league: string;
  players: typeof players;
  top11Avg: number;
  squadDepth: number;
  attackRating: number;
  midfieldRating: number;
  defenceRating: number;
  overallStrength: number;
}

function computeTeamData(teamName: string, league: string): TeamData {
  const squad = players.filter(p => p.team === teamName);
  if (squad.length === 0) return {
    name: teamName, league, players: [],
    top11Avg: 0, squadDepth: 0, attackRating: 0, midfieldRating: 0, defenceRating: 0, overallStrength: 0,
  };

  const sorted = [...squad].sort((a, b) => b.overall - a.overall);
  const top11 = sorted.slice(0, Math.min(11, sorted.length));
  const top11Avg = top11.reduce((s, p) => s + p.overall, 0) / top11.length;
  const squadDepth = sorted.length > 11
    ? sorted.slice(11).reduce((s, p) => s + p.overall, 0) / (sorted.length - 11)
    : top11Avg - 5;

  const attackers = squad.filter(p => ['ST', 'CF', 'LW', 'RW'].includes(p.position));
  const mids = squad.filter(p => ['CAM', 'CM', 'CDM'].includes(p.position));
  const defenders = squad.filter(p => ['CB', 'RB', 'LB', 'GK'].includes(p.position));

  const avg = (arr: typeof players) =>
    arr.length ? arr.reduce((s, p) => s + p.overall, 0) / arr.length : top11Avg;

  const overallStrength = top11Avg * 0.65 + squadDepth * 0.15 + (top11.length >= 8 ? 8 : 0);

  return {
    name: teamName, league, players: squad,
    top11Avg: Math.round(top11Avg * 10) / 10,
    squadDepth: Math.round(squadDepth * 10) / 10,
    attackRating: Math.round(avg(attackers) * 10) / 10,
    midfieldRating: Math.round(avg(mids) * 10) / 10,
    defenceRating: Math.round(avg(defenders) * 10) / 10,
    overallStrength: Math.round(overallStrength * 10) / 10,
  };
}

const teamEntries: { name: string; league: string }[] = [
  { name: 'Real Madrid', league: 'LaLiga EA SPORTS' },
  { name: 'Manchester City', league: 'Premier League' },
  { name: 'Liverpool', league: 'Premier League' },
  { name: 'FC Barcelona', league: 'LaLiga EA SPORTS' },
  { name: 'Bayern München', league: 'Bundesliga' },
  { name: 'Arsenal', league: 'Premier League' },
  { name: 'Atlético Madrid', league: 'LaLiga EA SPORTS' },
];

function RatingBadge({ value }: { value: number }) {
  return (
    <span
      className="inline-flex items-center justify-center font-display font-bold rounded"
      style={{ background: getRatingBg(value), color: getRatingTextColor(value), width: 44, height: 36, fontSize: 15 }}
    >
      {value.toFixed(0)}
    </span>
  );
}

function StrengthBar({ value, max = 95 }: { value: number; max?: number }) {
  const pct = (value / max) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'var(--secondary)' }}>
        <div
          className="h-full rounded-full"
          style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${getRatingBg(value)}, ${getRatingBg(Math.min(99, value + 5))})` }}
        />
      </div>
      <span className="text-xs font-data font-bold w-8 text-right" style={{ color: getRatingBg(value) }}>
        {value.toFixed(1)}
      </span>
    </div>
  );
}

export default function TeamStrength() {
  const teams = teamEntries.map(t => computeTeamData(t.name, t.league))
    .sort((a, b) => b.overallStrength - a.overallStrength);

  const maxStrength = teams[0]?.overallStrength || 95;

  return (
    <div>
      <DemoBanner reason="Squad strength is sample data. It derives from player ratings, which are not built yet." />
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Team Strength
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Aggregated squad ratings derived from player ML ratings. Used as a key input to the match predictor.
        </p>
      </div>

      {/* Method explainer */}
      <div
        className="rounded-xl p-4 mb-6"
        style={{ background: 'rgba(59,130,246,0.06)', border: '1px solid rgba(59,130,246,0.2)' }}
      >
        <div className="text-xs font-display font-bold uppercase tracking-wider mb-2" style={{ color: '#3b82f6' }}>
          Strength Calculation Method
        </div>
        <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
          <strong className="text-white">Overall Strength</strong> = Top 11 Avg Rating × 0.65 + Squad Depth Avg × 0.15 + Team Coverage Bonus.
          Attack, Midfield, and Defence ratings are computed from position-grouped players in each squad.
          This rating feeds directly into the Match Predictor model.
        </div>
      </div>

      {/* League table */}
      <div className="grid gap-3">
        {teams.map((team, rank) => (
          <div
            key={team.name}
            className="rounded-xl overflow-hidden"
            style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
          >
            <div className="p-5">
              <div className="flex items-center gap-4">
                {/* Rank */}
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center font-display font-black text-xl flex-shrink-0"
                  style={{
                    background: rank === 0 ? '#f59e0b22' : rank === 1 ? '#9ca3af22' : rank === 2 ? '#cd7c3222' : 'var(--secondary)',
                    color: rank === 0 ? '#f59e0b' : rank === 1 ? '#9ca3af' : rank === 2 ? '#cd7c32' : 'var(--muted-foreground)',
                    border: `1px solid ${rank === 0 ? '#f59e0b44' : rank === 1 ? '#9ca3af44' : rank === 2 ? '#cd7c3244' : 'var(--border)'}`,
                  }}
                >
                  {rank + 1}
                </div>

                {/* Team info */}
                <div className="flex-1">
                  <div className="font-display font-bold text-lg leading-tight" style={{ color: 'var(--foreground)' }}>
                    {team.name}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                    {team.league} · {team.players.length} players in database
                  </div>
                </div>

                {/* Overall strength */}
                <div className="flex flex-col items-center gap-1">
                  <div
                    className="font-display font-black text-3xl"
                    style={{ color: getRatingBg(team.overallStrength) }}
                  >
                    {team.overallStrength.toFixed(1)}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Overall</div>
                </div>

                {/* Rating breakdown */}
                <div className="flex gap-3">
                  {[
                    { label: 'ATT', value: team.attackRating },
                    { label: 'MID', value: team.midfieldRating },
                    { label: 'DEF', value: team.defenceRating },
                    { label: 'TOP11', value: team.top11Avg },
                  ].map(s => (
                    <div key={s.label} className="flex flex-col items-center gap-1">
                      {s.value > 0 ? <RatingBadge value={s.value} /> : (
                        <span className="text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>N/A</span>
                      )}
                      <span className="text-xs font-display font-bold" style={{ color: 'var(--muted-foreground)' }}>{s.label}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Strength bars */}
              <div className="mt-4 grid gap-2" style={{ gridTemplateColumns: '80px 1fr' }}>
                <span className="text-xs self-center" style={{ color: 'var(--muted-foreground)' }}>Squad Power</span>
                <StrengthBar value={team.overallStrength} max={maxStrength + 2} />
                <span className="text-xs self-center" style={{ color: 'var(--muted-foreground)' }}>Attack</span>
                {team.attackRating > 0
                  ? <StrengthBar value={team.attackRating} max={maxStrength + 2} />
                  : <div className="text-xs flex items-center" style={{ color: 'var(--muted-foreground)' }}>Not enough data</div>
                }
                <span className="text-xs self-center" style={{ color: 'var(--muted-foreground)' }}>Midfield</span>
                {team.midfieldRating > 0
                  ? <StrengthBar value={team.midfieldRating} max={maxStrength + 2} />
                  : <div className="text-xs flex items-center" style={{ color: 'var(--muted-foreground)' }}>Not enough data</div>
                }
                <span className="text-xs self-center" style={{ color: 'var(--muted-foreground)' }}>Defence</span>
                {team.defenceRating > 0
                  ? <StrengthBar value={team.defenceRating} max={maxStrength + 2} />
                  : <div className="text-xs flex items-center" style={{ color: 'var(--muted-foreground)' }}>Not enough data</div>
                }
              </div>

              {/* Squad players */}
              {team.players.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {[...team.players].sort((a, b) => b.overall - a.overall).map(p => (
                    <div
                      key={p.id}
                      className="flex items-center gap-1.5 rounded-lg px-2 py-1"
                      style={{ background: 'var(--secondary)', border: '1px solid var(--border)' }}
                      title={p.name}
                    >
                      <span className="text-xs font-display font-bold" style={{ color: 'var(--muted-foreground)' }}>
                        {p.position}
                      </span>
                      <span className="text-xs font-display" style={{ color: 'var(--foreground)' }}>{p.lastName}</span>
                      <span
                        className="text-xs font-data font-bold rounded px-1"
                        style={{ background: getRatingBg(p.overall), color: getRatingTextColor(p.overall) }}
                      >
                        {p.overall}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
