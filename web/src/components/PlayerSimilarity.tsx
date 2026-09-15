import { useState } from 'react';
import { players, computeSimilarity, getRatingBg, getRatingTextColor, type Player } from '../data/footballData';
import DemoBanner from './DemoBanner';

function RatingBadge({ value }: { value: number }) {
  return (
    <span
      className="inline-flex items-center justify-center font-display font-bold rounded"
      style={{ background: getRatingBg(value), color: getRatingTextColor(value), width: 36, height: 28, fontSize: 13 }}
    >
      {value}
    </span>
  );
}

function SimilarityCircle({ score }: { score: number }) {
  const r = 18;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  const color = score >= 80 ? '#00e676' : score >= 65 ? '#76ff03' : score >= 50 ? '#ffea00' : '#ff9100';

  return (
    <div className="relative flex items-center justify-center" style={{ width: 48, height: 48 }}>
      <svg width="48" height="48" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="24" cy="24" r={r} fill="none" stroke="#1e2d45" strokeWidth="3" />
        <circle
          cx="24" cy="24" r={r} fill="none"
          stroke={color} strokeWidth="3"
          strokeDasharray={`${dash} ${circ - dash}`}
          strokeLinecap="round"
        />
      </svg>
      <span
        className="absolute font-data font-bold text-sm"
        style={{ color }}
      >
        {score}
      </span>
    </div>
  );
}

function RadarChart({ player }: { player: Player }) {
  const stats = [
    { label: 'PAC', value: player.pac },
    { label: 'SHO', value: player.sho },
    { label: 'PAS', value: player.pas },
    { label: 'DRI', value: player.dri },
    { label: 'DEF', value: player.def },
    { label: 'PHY', value: player.phy },
  ];
  const n = stats.length;
  const cx = 80;
  const cy = 80;
  const r = 60;

  const points = stats.map((s, i) => {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    const fr = (s.value / 99) * r;
    return {
      x: cx + fr * Math.cos(angle),
      y: cy + fr * Math.sin(angle),
      lx: cx + (r + 16) * Math.cos(angle),
      ly: cy + (r + 16) * Math.sin(angle),
      label: s.label,
      value: s.value,
    };
  });

  const polygon = points.map(p => `${p.x},${p.y}`).join(' ');

  // grid circles
  const grids = [0.25, 0.5, 0.75, 1].map(f => {
    const gridPts = stats.map((_, i) => {
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
      return `${cx + f * r * Math.cos(angle)},${cy + f * r * Math.sin(angle)}`;
    });
    return gridPts.join(' ');
  });

  // axes
  const axes = stats.map((_, i) => {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    return { x1: cx, y1: cy, x2: cx + r * Math.cos(angle), y2: cy + r * Math.sin(angle) };
  });

  return (
    <svg viewBox="0 0 160 160" className="w-full max-w-48">
      {grids.map((pts, i) => (
        <polygon key={i} points={pts} fill="none" stroke="#1e2d45" strokeWidth="0.5" />
      ))}
      {axes.map((ax, i) => (
        <line key={i} x1={ax.x1} y1={ax.y1} x2={ax.x2} y2={ax.y2} stroke="#1e2d45" strokeWidth="0.5" />
      ))}
      <polygon points={polygon} fill="rgba(0,230,118,0.15)" stroke="#00e676" strokeWidth="1.5" />
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="2.5" fill="#00e676" />
          <text
            x={p.lx} y={p.ly}
            textAnchor="middle" dominantBaseline="middle"
            fontSize="8" fill="#94a3b8" fontFamily="'Barlow Condensed', sans-serif" fontWeight="700"
          >
            {p.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default function PlayerSimilarity() {
  const [selectedId, setSelectedId] = useState<number>(1);
  const [search, setSearch] = useState('');

  const selected = players.find(p => p.id === selectedId)!;

  const similar = players
    .filter(p => p.id !== selectedId)
    .map(p => ({ player: p, score: computeSimilarity(selected, p) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 20);

  const searchResults = players.filter(
    p => search.length > 0 && p.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <DemoBanner reason="Similarity is computed over sample players. Real results need the player rating vectors." />
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Player Similarity
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Find the most similar players using cosine similarity across all ML-rated attributes.
        </p>
      </div>

      {/* Player selector */}
      <div
        className="rounded-xl p-5 mb-6"
        style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
      >
        <div className="flex items-center gap-4 flex-wrap">
          <div className="relative flex-1 min-w-48">
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search player..."
              className="w-full rounded-lg px-3 py-2 text-sm outline-none"
              style={{ background: 'var(--secondary)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
            />
            {searchResults.length > 0 && (
              <div
                className="absolute top-full left-0 right-0 z-10 mt-1 rounded-lg overflow-hidden max-h-48 overflow-y-auto"
                style={{ background: 'var(--card)', border: '1px solid var(--border)', boxShadow: '0 8px 32px rgba(0,0,0,0.5)' }}
              >
                {searchResults.map(p => (
                  <div
                    key={p.id}
                    className="px-3 py-2.5 flex items-center gap-2 cursor-pointer transition-colors"
                    style={{ borderBottom: '1px solid var(--border)' }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,230,118,0.06)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                    onClick={() => { setSelectedId(p.id); setSearch(''); }}
                  >
                    <span>{p.flag}</span>
                    <div>
                      <div className="text-sm font-display font-bold">{p.name}</div>
                      <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>{p.position} · {p.team}</div>
                    </div>
                    <span
                      className="ml-auto inline-flex items-center justify-center font-display font-bold rounded text-xs"
                      style={{ background: getRatingBg(p.overall), color: getRatingTextColor(p.overall), width: 32, height: 24 }}
                    >
                      {p.overall}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <span className="text-sm" style={{ color: 'var(--muted-foreground)' }}>or pick:</span>
          <div className="flex flex-wrap gap-2">
            {players.slice(0, 8).map(p => (
              <button
                key={p.id}
                onClick={() => { setSelectedId(p.id); setSearch(''); }}
                className="text-xs px-3 py-1.5 rounded-lg font-display font-bold transition-all"
                style={{
                  background: selectedId === p.id ? '#00e676' : 'var(--secondary)',
                  color: selectedId === p.id ? '#000' : 'var(--muted-foreground)',
                  border: `1px solid ${selectedId === p.id ? '#00e676' : 'var(--border)'}`,
                }}
              >
                {p.lastName}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Selected player hero */}
      <div
        className="rounded-xl p-6 mb-6"
        style={{ background: 'linear-gradient(135deg, #0d1a2e 0%, #1a0d2e 100%)', border: '1px solid var(--border)' }}
      >
        <div className="grid gap-6" style={{ gridTemplateColumns: 'auto 1fr auto' }}>
          <div
            className="w-20 h-20 rounded-xl flex items-center justify-center text-4xl font-display font-black"
            style={{ background: selected.photoColor, border: '2px solid rgba(255,255,255,0.1)' }}
          >
            {selected.firstName.slice(0, 1)}
          </div>
          <div>
            <div className="flex items-center gap-3 mb-1">
              <span className="text-xl">{selected.flag}</span>
              <div>
                <div className="font-display font-black text-3xl leading-tight" style={{ color: 'var(--foreground)' }}>
                  {selected.name.toUpperCase()}
                </div>
                <div className="text-sm" style={{ color: 'var(--muted-foreground)' }}>
                  {selected.position} · {selected.team} · {selected.league} · {selected.age} yrs · {selected.height} cm
                </div>
              </div>
            </div>
            <div className="flex gap-4 mt-3 flex-wrap">
              {[
                { label: 'NATION', value: selected.nationality },
                { label: 'CLUB', value: selected.team },
                { label: 'AGE', value: `${selected.age} yrs` },
                { label: 'HEIGHT', value: `${selected.height} cm` },
              ].map(f => (
                <div key={f.label} className="rounded-lg px-3 py-2" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
                  <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: 'var(--muted-foreground)' }}>{f.label}</div>
                  <div className="text-sm font-display font-bold text-white">{f.value}</div>
                </div>
              ))}
            </div>
          </div>
          <RadarChart player={selected} />
        </div>
      </div>

      {/* Similar players */}
      <div className="mb-3">
        <h3 className="font-display font-bold text-xl" style={{ color: 'var(--foreground)' }}>
          20 Most Similar Players
        </h3>
        <p className="text-xs mt-0.5" style={{ color: 'var(--muted-foreground)' }}>
          Similarity score based on cosine similarity across PAC · SHO · PAS · DRI · DEF · PHY
        </p>
      </div>

      <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
        {similar.map((s, i) => (
          <div
            key={s.player.id}
            className="rounded-xl p-4 flex items-center gap-3 cursor-pointer transition-all"
            style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
            onClick={() => { setSelectedId(s.player.id); setSearch(''); }}
            onMouseEnter={e => (e.currentTarget.style.borderColor = '#00e67644')}
            onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
          >
            <span className="text-sm font-display font-bold w-6 text-center" style={{ color: 'var(--muted-foreground)' }}>
              {i + 1}
            </span>
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center text-base font-display font-black flex-shrink-0"
              style={{ background: s.player.photoColor, border: '1px solid rgba(255,255,255,0.1)' }}
            >
              {s.player.firstName.slice(0, 1)}
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-display font-bold text-sm truncate" style={{ color: 'var(--foreground)' }}>
                {s.player.flag} {s.player.name}
              </div>
              <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
                {s.player.age}, {s.player.position}, {s.player.team}
              </div>
            </div>
            <RatingBadge value={s.player.overall} />
            <SimilarityCircle score={s.score} />
          </div>
        ))}
      </div>
    </div>
  );
}
