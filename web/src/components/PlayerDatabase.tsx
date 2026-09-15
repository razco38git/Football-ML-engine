import { useState } from 'react';
import { players, leagues, positions, getRatingBg, getRatingTextColor, type Player } from '../data/footballData';
import DemoBanner from './DemoBanner';

function RatingBadge({ value, size = 'sm' }: { value: number; size?: 'sm' | 'md' | 'lg' }) {
  const bg = getRatingBg(value);
  const color = getRatingTextColor(value);
  const dims = size === 'lg' ? { w: 52, h: 44, fs: 20 } : size === 'md' ? { w: 44, h: 36, fs: 16 } : { w: 36, h: 28, fs: 13 };
  return (
    <span
      className="inline-flex items-center justify-center font-display font-bold rounded"
      style={{ background: bg, color, width: dims.w, height: dims.h, fontSize: dims.fs, flexShrink: 0 }}
    >
      {value}
    </span>
  );
}

function StatBar({ label, value }: { label: string; value: number }) {
  const bg = getRatingBg(value);
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs w-28 shrink-0" style={{ color: 'var(--muted-foreground)' }}>{label}</span>
      <div className="flex-1 h-1.5 rounded-full" style={{ background: 'var(--secondary)' }}>
        <div className="h-full rounded-full" style={{ width: `${value}%`, background: bg }} />
      </div>
      <span className="text-xs font-data font-bold w-6 text-right" style={{ color: bg }}>{value}</span>
    </div>
  );
}

function PlayerDetailPanel({ player, onClose }: { player: Player; onClose: () => void }) {
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
        {/* Header */}
        <div
          className="p-6 relative"
          style={{ background: 'linear-gradient(135deg, #0d1a2e 0%, #1a0d2e 100%)', borderBottom: '1px solid var(--border)' }}
        >
          <button
            onClick={onClose}
            className="absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center text-sm transition-colors hover:bg-white/10"
            style={{ color: 'var(--muted-foreground)' }}
          >
            ✕
          </button>
          <div className="flex items-start gap-5">
            <div
              className="w-20 h-20 rounded-xl flex items-center justify-center text-4xl font-display font-black"
              style={{ background: player.photoColor, border: '2px solid rgba(255,255,255,0.1)' }}
            >
              {player.firstName.slice(0, 1)}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-1">
                <span className="text-2xl">{player.flag}</span>
                <div>
                  <div className="font-display font-black text-2xl leading-tight" style={{ color: 'var(--foreground)' }}>
                    {player.name}
                  </div>
                  <div className="text-sm" style={{ color: 'var(--muted-foreground)' }}>
                    {player.team} · {player.league}
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mt-3">
                <span className="text-xs px-2 py-0.5 rounded font-display font-bold" style={{ background: '#00e67622', color: '#00e676', border: '1px solid #00e67644' }}>
                  {player.position}
                </span>
                {player.altPosition !== player.position && (
                  <span className="text-xs px-2 py-0.5 rounded font-display" style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}>
                    {player.altPosition}
                  </span>
                )}
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}>
                  Age {player.age}
                </span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}>
                  {player.height} cm
                </span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}>
                  {'★'.repeat(player.skillMoves)}{'☆'.repeat(5 - player.skillMoves)} SM
                </span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}>
                  {'★'.repeat(player.weakFoot)}{'☆'.repeat(5 - player.weakFoot)} WF
                </span>
              </div>
            </div>
            <div className="flex flex-col items-center gap-1">
              <RatingBadge value={player.overall} size="lg" />
              <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>OVR</span>
              <RatingBadge value={player.potential} size="md" />
              <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>POT</span>
            </div>
          </div>

          {/* 6 main stats */}
          <div className="grid grid-cols-6 gap-2 mt-5">
            {[
              { label: 'PAC', value: player.pac },
              { label: 'SHO', value: player.sho },
              { label: 'PAS', value: player.pas },
              { label: 'DRI', value: player.dri },
              { label: 'DEF', value: player.def },
              { label: 'PHY', value: player.phy },
            ].map(s => (
              <div key={s.label} className="flex flex-col items-center gap-1">
                <RatingBadge value={s.value} size="md" />
                <span className="text-xs font-display font-bold" style={{ color: 'var(--muted-foreground)' }}>{s.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Sub-stats */}
        <div className="p-6 grid gap-5" style={{ gridTemplateColumns: '1fr 1fr' }}>
          <div>
            <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: '#00e676' }}>Attacking</div>
            <div className="flex flex-col gap-2">
              <StatBar label="Crossing" value={player.crossing} />
              <StatBar label="Finishing" value={player.finishing} />
              <StatBar label="Heading Acc." value={player.headingAccuracy} />
              <StatBar label="Short Passing" value={player.shortPassing} />
              <StatBar label="Volleys" value={player.volleys} />
            </div>
            <div className="text-xs font-display font-bold uppercase tracking-wider mb-3 mt-4" style={{ color: '#3b82f6' }}>Defending</div>
            <div className="flex flex-col gap-2">
              <StatBar label="Def. Awareness" value={player.defensiveAwareness} />
              <StatBar label="Stand. Tackle" value={player.standingTackle} />
              <StatBar label="Sliding Tackle" value={player.slidingTackle} />
              <StatBar label="Interceptions" value={player.interceptions} />
            </div>
          </div>
          <div>
            <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: '#f59e0b' }}>Skill</div>
            <div className="flex flex-col gap-2">
              <StatBar label="Dribbling" value={player.dribbling} />
              <StatBar label="Curve" value={player.curve} />
              <StatBar label="FK Accuracy" value={player.fkAccuracy} />
              <StatBar label="Long Passing" value={player.longPassing} />
              <StatBar label="Ball Control" value={player.ballControl} />
            </div>
            <div className="text-xs font-display font-bold uppercase tracking-wider mb-3 mt-4" style={{ color: '#a78bfa' }}>Power & Mentality</div>
            <div className="flex flex-col gap-2">
              <StatBar label="Shot Power" value={player.shotPower} />
              <StatBar label="Jumping" value={player.jumping} />
              <StatBar label="Stamina" value={player.stamina} />
              <StatBar label="Strength" value={player.strength} />
              <StatBar label="Vision" value={player.vision} />
              <StatBar label="Composure" value={player.composure} />
            </div>
          </div>
          <div>
            <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: '#fb7185' }}>Movement</div>
            <div className="flex flex-col gap-2">
              <StatBar label="Acceleration" value={player.acceleration} />
              <StatBar label="Sprint Speed" value={player.sprintSpeed} />
              <StatBar label="Agility" value={player.agility} />
              <StatBar label="Reactions" value={player.reactions} />
              <StatBar label="Balance" value={player.balance} />
            </div>
          </div>
          <div>
            <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: '#34d399' }}>Data Sources</div>
            <div className="flex flex-col gap-3">
              <div className="rounded-lg p-3" style={{ background: 'var(--secondary)', border: '1px solid var(--border)' }}>
                <div className="text-xs mb-1" style={{ color: 'var(--muted-foreground)' }}>EA FC Rating</div>
                <RatingBadge value={player.overall} size="md" />
              </div>
              <div className="rounded-lg p-3" style={{ background: 'var(--secondary)', border: '1px solid var(--border)' }}>
                <div className="text-xs mb-1" style={{ color: 'var(--muted-foreground)' }}>Football Manager (FM) Rating</div>
                <RatingBadge value={player.fmRating} size="md" />
              </div>
              <div className="rounded-lg p-3" style={{ background: 'var(--secondary)', border: '1px solid var(--border)' }}>
                <div className="text-xs mb-2" style={{ color: 'var(--muted-foreground)' }}>ScoutLab Percentile</div>
                <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--muted)' }}>
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${player.scoutPercentile}%`, background: getRatingBg(player.scoutPercentile) }}
                  />
                </div>
                <div className="text-xs font-data font-bold mt-1" style={{ color: '#00e676' }}>
                  P{player.scoutPercentile}
                </div>
              </div>
            </div>
          </div>
        </div>

        {player.position === 'GK' && (
          <div className="px-6 pb-6">
            <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: '#fbbf24' }}>Goalkeeper</div>
            <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
              {[
                { label: 'Diving', value: player.gkDiving },
                { label: 'Handling', value: player.gkHandling },
                { label: 'Kicking', value: player.gkKicking },
                { label: 'Positioning', value: player.gkPositioning },
                { label: 'Reflexes', value: player.gkReflexes },
              ].map(s => (
                <div key={s.label} className="flex flex-col items-center gap-1">
                  <RatingBadge value={s.value} size="sm" />
                  <span className="text-xs text-center" style={{ color: 'var(--muted-foreground)' }}>{s.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function PlayerDatabase() {
  const [selectedLeague, setSelectedLeague] = useState('All');
  const [selectedPosition, setSelectedPosition] = useState('All');
  const [minRating, setMinRating] = useState(0);
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<keyof Player>('overall');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);

  const filtered = players
    .filter(p =>
      (selectedLeague === 'All' || p.league === selectedLeague) &&
      (selectedPosition === 'All' || p.position === selectedPosition || p.altPosition === selectedPosition) &&
      p.overall >= minRating &&
      (search === '' || p.name.toLowerCase().includes(search.toLowerCase()) || p.team.toLowerCase().includes(search.toLowerCase()))
    )
    .sort((a, b) => {
      const av = a[sortBy] as number;
      const bv = b[sortBy] as number;
      return sortDir === 'desc' ? bv - av : av - bv;
    });

  const handleSort = (col: keyof Player) => {
    if (sortBy === col) setSortDir(d => (d === 'desc' ? 'asc' : 'desc'));
    else { setSortBy(col); setSortDir('desc'); }
  };

  const SortHeader = ({ col, label }: { col: keyof Player; label: string }) => (
    <th
      className="px-2 py-3 text-xs font-display font-bold uppercase tracking-wider cursor-pointer select-none transition-colors hover:text-white"
      style={{ color: sortBy === col ? '#00e676' : 'var(--muted-foreground)', whiteSpace: 'nowrap' }}
      onClick={() => handleSort(col)}
    >
      {label}{sortBy === col ? (sortDir === 'desc' ? ' ↓' : ' ↑') : ''}
    </th>
  );

  return (
    <div>
      <DemoBanner reason="Ratings shown are sample data. The 0-99 rating model needs the FBref player pipeline, which is not built yet." />
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Player Ratings
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          ML-aggregated ratings combining EA FC stats, Football Manager attributes, and ScoutLab percentiles.
        </p>
      </div>

      {/* Filters */}
      <div
        className="rounded-xl p-4 mb-5 flex flex-wrap gap-3 items-center"
        style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
      >
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search player or club..."
          className="rounded-lg px-3 py-2 text-sm outline-none flex-1 min-w-48"
          style={{ background: 'var(--secondary)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        />
        <select
          value={selectedLeague}
          onChange={e => setSelectedLeague(e.target.value)}
          className="rounded-lg px-3 py-2 text-sm outline-none"
          style={{ background: 'var(--secondary)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          {leagues.map(l => <option key={l} value={l}>{l}</option>)}
        </select>
        <select
          value={selectedPosition}
          onChange={e => setSelectedPosition(e.target.value)}
          className="rounded-lg px-3 py-2 text-sm outline-none"
          style={{ background: 'var(--secondary)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          {positions.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Min OVR</span>
          <input
            type="range" min={0} max={90} step={5} value={minRating}
            onChange={e => setMinRating(Number(e.target.value))}
            className="w-20"
          />
          <span className="text-xs font-data font-bold w-6" style={{ color: '#00e676' }}>{minRating || 'All'}</span>
        </div>
        <span className="text-xs ml-auto" style={{ color: 'var(--muted-foreground)' }}>
          {filtered.length} players
        </span>
      </div>

      {/* Table */}
      <div
        className="rounded-xl overflow-hidden"
        style={{ border: '1px solid var(--border)' }}
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr style={{ background: 'var(--muted)', borderBottom: '1px solid var(--border)' }}>
                <th className="px-4 py-3 text-left text-xs font-display font-bold uppercase tracking-wider" style={{ color: 'var(--muted-foreground)' }}>
                  Player
                </th>
                <SortHeader col="position" label="POS" />
                <SortHeader col="overall" label="OVR" />
                <SortHeader col="potential" label="POT" />
                <SortHeader col="age" label="AGE" />
                <SortHeader col="pac" label="PAC" />
                <SortHeader col="sho" label="SHO" />
                <SortHeader col="pas" label="PAS" />
                <SortHeader col="dri" label="DRI" />
                <SortHeader col="def" label="DEF" />
                <SortHeader col="phy" label="PHY" />
                <SortHeader col="fmRating" label="FM" />
                <SortHeader col="scoutPercentile" label="SCOUT" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((p, i) => (
                <tr
                  key={p.id}
                  className="cursor-pointer transition-colors"
                  style={{
                    background: i % 2 === 0 ? 'var(--card)' : 'rgba(255,255,255,0.015)',
                    borderBottom: '1px solid rgba(30,45,69,0.5)',
                  }}
                  onClick={() => setSelectedPlayer(p)}
                  onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,230,118,0.06)')}
                  onMouseLeave={e => (e.currentTarget.style.background = i % 2 === 0 ? 'var(--card)' : 'rgba(255,255,255,0.015)')}
                >
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-display font-black flex-shrink-0"
                        style={{ background: p.photoColor, border: '1px solid rgba(255,255,255,0.1)' }}
                      >
                        {p.firstName.slice(0, 1)}
                      </div>
                      <div>
                        <div className="font-display font-bold text-sm leading-tight" style={{ color: 'var(--foreground)' }}>
                          {p.flag} {p.name}
                        </div>
                        <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>{p.team}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-2 py-2.5 text-center">
                    <span
                      className="text-xs font-display font-bold px-2 py-0.5 rounded"
                      style={{ background: 'var(--secondary)', color: '#00e676' }}
                    >
                      {p.position}
                    </span>
                  </td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.overall} /></td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.potential} /></td>
                  <td className="px-2 py-2.5 text-center text-sm font-data" style={{ color: 'var(--muted-foreground)' }}>{p.age}</td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.pac} /></td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.sho} /></td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.pas} /></td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.dri} /></td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.def} /></td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.phy} /></td>
                  <td className="px-2 py-2.5 text-center"><RatingBadge value={p.fmRating} /></td>
                  <td className="px-2 py-2.5 text-center">
                    <span className="text-xs font-data font-bold" style={{ color: getRatingBg(p.scoutPercentile) }}>
                      P{p.scoutPercentile}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selectedPlayer && (
        <PlayerDetailPanel player={selectedPlayer} onClose={() => setSelectedPlayer(null)} />
      )}
    </div>
  );
}
