import { useState } from 'react';
import { api, LEAGUE_NAMES, seasonLabel, type PlayerRating } from '../api/client';
import { getRatingBg, getRatingTextColor, teamColor } from '../api/display';
import { useAsync } from '../api/hooks';

const POSITIONS: Record<string, string> = {
  GK: 'Goalkeeper',
  CB: 'Centre back',
  FB: 'Full back',
  DM: 'Defensive midfielder',
  MID: 'Central midfielder',
  AMW: 'Attacking mid / winger',
  FWD: 'Forward',
};

/**
 * Sub-ratings shown per position -- only what we actually measure for that role.
 *
 * Deliberately not the FIFA six (pace, shooting, passing, dribbling, defending,
 * physical). Nothing in the source data supports a pace or dribbling number, and
 * inventing one to fill a column would undermine every honest figure beside it.
 */
const ATTRIBUTES: Record<string, { key: keyof PlayerRating; label: string; short: string }[]> = {
  GK: [
    { key: 'sub_shot_stopping', label: 'Shot stopping', short: 'STP' },
    { key: 'sub_reliability', label: 'Clean sheets', short: 'CS' },
    { key: 'sub_workload', label: 'Workload faced', short: 'WRK' },
    { key: 'sub_penalties', label: 'Penalties', short: 'PEN' },
  ],
  CB: [
    { key: 'sub_defending', label: 'Defending', short: 'DEF' },
    { key: 'sub_involvement', label: 'Build-up', short: 'BLD' },
    { key: 'sub_creation', label: 'Creation', short: 'CRE' },
    { key: 'sub_finishing', label: 'Finishing', short: 'FIN' },
  ],
  FB: [
    { key: 'sub_defending', label: 'Defending', short: 'DEF' },
    { key: 'sub_creation', label: 'Creation & crossing', short: 'CRE' },
    { key: 'sub_involvement', label: 'Build-up', short: 'BLD' },
    { key: 'sub_finishing', label: 'Finishing', short: 'FIN' },
  ],
  // A holder leads with winning the ball back; a number eight with creating.
  // Same four bars, ordered by what the role is for.
  DM: [
    { key: 'sub_defending', label: 'Defending', short: 'DEF' },
    { key: 'sub_involvement', label: 'Build-up', short: 'BLD' },
    { key: 'sub_creation', label: 'Creation', short: 'CRE' },
    { key: 'sub_finishing', label: 'Finishing', short: 'FIN' },
  ],
  MID: [
    { key: 'sub_creation', label: 'Creation', short: 'CRE' },
    { key: 'sub_involvement', label: 'Build-up', short: 'BLD' },
    { key: 'sub_defending', label: 'Defending', short: 'DEF' },
    { key: 'sub_finishing', label: 'Finishing', short: 'FIN' },
  ],
  AMW: [
    { key: 'sub_creation', label: 'Creation', short: 'CRE' },
    { key: 'sub_finishing', label: 'Finishing', short: 'FIN' },
    { key: 'sub_involvement', label: 'Build-up', short: 'BLD' },
    { key: 'sub_volume', label: 'Shot volume', short: 'VOL' },
  ],
  FWD: [
    { key: 'sub_finishing', label: 'Finishing', short: 'FIN' },
    { key: 'sub_creation', label: 'Creation', short: 'CRE' },
    { key: 'sub_involvement', label: 'Build-up', short: 'BLD' },
    { key: 'sub_volume', label: 'Shot volume', short: 'VOL' },
  ],
};

const SLOTS = [0, 1, 2, 3];

function RatingBadge({ value, size = 'sm' }: { value: number | null; size?: 'sm' | 'md' | 'lg' }) {
  const dims =
    size === 'lg' ? { w: 52, h: 44, fs: 20 } : size === 'md' ? { w: 44, h: 36, fs: 16 } : { w: 36, h: 28, fs: 13 };

  if (value == null) {
    return (
      <span
        className="inline-flex items-center justify-center font-display rounded"
        style={{
          background: 'var(--secondary)', color: 'var(--muted-foreground)',
          width: dims.w, height: dims.h, fontSize: dims.fs, flexShrink: 0,
        }}
      >
        –
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center justify-center font-display font-bold rounded"
      style={{
        background: getRatingBg(value), color: getRatingTextColor(value),
        width: dims.w, height: dims.h, fontSize: dims.fs, flexShrink: 0,
      }}
    >
      {value}
    </span>
  );
}

function StatBar({ label, value }: { label: string; value: number | null }) {
  // An attribute this position is scored on, with no value, is a gap in the
  // source rather than an attribute that does not apply -- FBref published no
  // defensive data at all for 2014/15, so no centre-back that season has a
  // defending score. Hiding the row left a defender with three bars and no
  // hint that a fourth was missing; showing it empty says which one.
  if (value == null) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs w-32 shrink-0" style={{ color: 'var(--muted-foreground)' }}>{label}</span>
        <div className="flex-1 h-1.5 rounded-full" style={{ background: 'var(--secondary)' }} />
        <span className="text-xs font-data w-6 text-right" style={{ color: 'var(--muted-foreground)' }} title="Not published for this season">—</span>
      </div>
    );
  }
  const bg = getRatingBg(value);
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs w-32 shrink-0" style={{ color: 'var(--muted-foreground)' }}>{label}</span>
      <div className="flex-1 h-1.5 rounded-full" style={{ background: 'var(--secondary)' }}>
        <div className="h-full rounded-full" style={{ width: `${value}%`, background: bg }} />
      </div>
      <span className="text-xs font-data font-bold w-6 text-right" style={{ color: bg }}>{value}</span>
    </div>
  );
}

function Underlying({ player }: { player: PlayerRating }) {
  const rows: [string, string | number][] = [];
  const show = (label: string, value: number | null | undefined, digits = 2) => {
    if (value != null) rows.push([label, digits === 0 ? value : value.toFixed(digits)]);
  };

  if (player.position === 'GK') {
    show('Save percentage', player.save_pct, 1);
    show('Goals against per 90', player.goals_against_per90);
  } else {
    show('Goals', player.goals, 0);
    show('Assists', player.assists, 0);
    show('Non-penalty xG', player.np_xg);
    show('Expected assists', player.xa);
    show('Key passes per 90', player.key_passes_per90);
    show('Interceptions per 90', player.interceptions_per90);
    show('Tackles won per 90', player.tackles_won_per90);
  }

  return (
    <div className="grid gap-x-6 gap-y-1" style={{ gridTemplateColumns: '1fr 1fr' }}>
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between text-xs py-0.5">
          <span style={{ color: 'var(--muted-foreground)' }}>{label}</span>
          <span className="font-data font-bold text-white">{value}</span>
        </div>
      ))}
    </div>
  );
}

function PlayerDetailPanel({ player, onClose }: { player: PlayerRating; onClose: () => void }) {
  const attributes = ATTRIBUTES[player.position] ?? [];
  const color = teamColor(player.team);

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
          <div className="flex items-start gap-5">
            <div
              className="w-20 h-20 rounded-xl flex items-center justify-center text-2xl font-display font-black"
              style={{ background: color + '33', border: `2px solid ${color}66`, color }}
            >
              {player.player.split(' ').map(w => w[0]).join('').slice(0, 2)}
            </div>
            <div className="flex-1">
              <div className="font-display font-black text-2xl" style={{ color: 'var(--foreground)' }}>
                {player.player}
              </div>
              <div className="text-sm mt-0.5" style={{ color: 'var(--muted-foreground)' }}>
                {player.team} · {LEAGUE_NAMES[player.league] ?? player.league} ·{' '}
                {POSITIONS[player.position] ?? player.position}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--muted-foreground)' }}>
                {player.season.slice(0, 2)}/{player.season.slice(2)} · {player.minutes.toLocaleString()} minutes
              </div>
            </div>
            <RatingBadge value={player.rating} size="lg" />
          </div>
        </div>

        <div className="p-6">
          <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: 'var(--muted-foreground)' }}>
            Rating breakdown
          </div>
          <div className="flex flex-col gap-2 mb-6">
            {attributes.map(a => (
              <StatBar key={String(a.key)} label={a.label} value={player[a.key] as number | null} />
            ))}
          </div>

          <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: 'var(--muted-foreground)' }}>
            Underlying numbers
          </div>
          <Underlying player={player} />

          <div className="mt-5 pt-4 text-xs" style={{ borderTop: '1px solid var(--border)', color: 'var(--muted-foreground)' }}>
            Ranked against {POSITIONS[player.position]?.toLowerCase()}s in the same season, then pulled
            toward the positional average in proportion to minutes played.
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * How much of the rating EA is carrying, said out loud when it is not half.
 *
 * It is half for every season from 2016/17 on, which is why the explainer can
 * talk about a 50/50 split. It is not half for 2014/15 or 2015/16: FBref
 * served no `misc` table, so a centre back's defending -- 62% of what the role
 * means -- could not be computed, and the blend leans on EA in proportion to
 * what was missing rather than pretending the gap was average. Saying "the
 * 50/50 blend" on those rows would be a plain untruth, and they are exactly
 * the rows a reader is most likely to be querying.
 */
function eaShare(p: PlayerRating): string {
  const used = p.fifa_weight_used;
  if (used == null || Math.abs(used - 0.5) < 0.01) return '';
  const share = p.measured_share;
  const measured = share == null ? '' : ` — only ${Math.round(share * 100)}% of what this position is scored on was recorded that season`;
  return `, at ${Math.round(used * 100)}% EA rather than the usual 50${measured}`;
}

export default function PlayerDatabase() {
  // Empty means "whatever the server defaults to" -- early in a campaign that
  // is last season, because too few players have the minutes to be rated yet.
  const [season, setSeason] = useState('');
  const [league, setLeague] = useState('All');
  const [position, setPosition] = useState('All');
  const [minRating, setMinRating] = useState(0);
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState('rating');
  const [descending, setDescending] = useState(true);
  const [selected, setSelected] = useState<PlayerRating | null>(null);

  const { data, loading, error, reload } = useAsync(
    () =>
      api.players({
        season: season || undefined,
        league: league === 'All' ? undefined : league,
        position: position === 'All' ? undefined : position,
        search: search || undefined,
        min_rating: minRating,
        sort,
        descending,
        // Each row renders five rating badges plus an avatar, so a large page
        // is a lot of DOM for little benefit -- filters are the way to find
        // someone, not scrolling.
        limit: 50,
      }),
    [season, league, position, minRating, search, sort, descending],
  );

  // True when the newest season with ratings is not the newest season that
  // exists -- i.e. the campaign is under way but too little has been played.
  const isPreviousSeason = Boolean(
    data?.season && data.seasons.length > 0 && data.seasons[0] !== data.season,
  );

  // Empty while "All positions" is selected, which is what makes the attribute
  // headers unsortable there.
  const selectedAttributes = position === 'All' ? [] : (ATTRIBUTES[position] ?? []);

  // Changing position can strand the sort on an attribute the new position does
  // not have -- forwards have no `sub_defending` -- which would quietly rank
  // everyone by a column of dashes. Fall back to the overall rating.
  const changePosition = (next: string) => {
    setPosition(next);
    if (sort.startsWith('sub_')) {
      const allowed = next === 'All' ? [] : (ATTRIBUTES[next] ?? []);
      if (!allowed.some(a => a.key === sort)) {
        setSort('rating');
        setDescending(true);
      }
    }
  };

  const toggleSort = (col: string) => {
    if (sort === col) setDescending(d => !d);
    else {
      setSort(col);
      setDescending(true);
    }
  };

  const SortHeader = ({ col, label }: { col: string; label: string }) => (
    <th
      onClick={() => toggleSort(col)}
      className="px-2 py-2 text-xs font-display font-bold uppercase tracking-wider cursor-pointer select-none"
      style={{ color: sort === col ? '#00e676' : 'var(--muted-foreground)', whiteSpace: 'nowrap' }}
    >
      {label}
      {sort === col ? (descending ? ' ↓' : ' ↑') : ''}
    </th>
  );

  return (
    <div>
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Player Ratings
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Ratings out of 99 built from per-90 output, ranked against positional peers.
          Click any player for the breakdown.
          {data?.season && (
            <>
              {' '}Showing <strong style={{ color: 'var(--foreground)' }}>
                {seasonLabel(data.season)}
              </strong>
              {isPreviousSeason && (
                <> — clubs are that season's, so a summer transfer is not reflected here.
                  A player needs 450 minutes before he can be rated, so the current
                  campaign only appears once enough of it has been played.</>
              )}
            </>
          )}
        </p>

        {/*
          How OVR is made, stated once. The table prints EA, Perf and OVR side
          by side and OVR is plainly not the average of the other two -- 63 and
          78 give 64 -- which reads as a broken sum until you know the two
          scales are not the same measurement.
        */}
        <details className="mt-3 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          <summary className="cursor-pointer" style={{ color: 'var(--muted-foreground)' }}>
            How OVR is built from EA and Perf
          </summary>
          <div className="mt-2 leading-relaxed rounded-xl p-4"
               style={{ background: 'var(--card)', border: '1px solid var(--border)' }}>
            <strong style={{ color: 'var(--foreground)' }}>OVR is half EA, half us</strong> —
            but the two are not on the same scale, so they are put on one before
            being averaged. EA&rsquo;s number is global: a 63 means the same thing
            to EA whoever it belongs to. Ours is a <em>rank against positional
            peers in that season</em>, so it spreads across the whole 0–99 range
            by construction while EA&rsquo;s clusters in the sixties and
            seventies.
            <br /><br />
            Averaging them raw would let whichever is wider dominate, so EA&rsquo;s
            overall is first converted to its percentile within the player&rsquo;s
            position and season, then mapped onto our scale, and it is that
            converted number the blend uses — not the EA score in the table.
            A 63 among 181 defensive midfielders is the 1st percentile, which is
            49 on our scale, so 49 and 78 average to 64. Hover the EA column to
            see what any row&rsquo;s score converts to.
            <br /><br />
            Where the two disagree is the interesting part: well above EA means a
            player is outperforming his reputation this season, well below means
            he is coasting on it.
            <br /><br />
            <strong style={{ color: 'var(--foreground)' }}>Half, except where
            we could not measure our half.</strong> 2014/15 and 2015/16 have no
            tackles or interceptions in the source at all, and defending is 62%
            of what a centre back is scored on. Rather than rank him on the
            passing and goals that happen to be left and call it half the
            rating, those seasons lean on EA in proportion to what is missing
            &mdash; a centre back there is 81% EA, a full back 69%. Hovering the
            EA column on one of those rows says so. Every season from 2016/17
            on is an even split.
          </div>
        </details>

        {/*
          The limit that matters most, stated on the page rather than left for
          a reader to discover by disagreeing with a defender's rating.
          Measuring it was the point: whether the rating is weaker for
          defenders is not a matter of opinion, and the number says it is.
        */}
        <details className="mt-2 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          <summary className="cursor-pointer" style={{ color: 'var(--muted-foreground)' }}>
            Why defenders and midfielders are rated less well than attackers
          </summary>
          <div className="mt-2 leading-relaxed rounded-xl p-4"
               style={{ background: 'var(--card)', border: '1px solid var(--border)' }}>
            <strong style={{ color: 'var(--foreground)' }}>This rating is better
            at attackers, because the data is.</strong> Our source publishes
            full shooting and chance-creation numbers for every player, but
            serves its tackling, possession and passing tables with the player
            rows stripped out &mdash; headers present, cells empty. What
            survives for a defender is interceptions, tackles won, fouls and
            crosses. There is no passing accuracy, no duels, no recoveries, no
            clearances, and no aerials.
            <br /><br />
            So a striker is judged on most of his job and a centre back on a
            corner of his. That is measurable rather than arguable. Scoring each
            season against how EA moved the player&rsquo;s rating the following
            year &mdash; a fair proxy for &ldquo;did he have a good season&rdquo;
            &mdash; our rating tracks it like this, for players with 1,500+
            minutes since 2016/17:
            <div className="mt-3 font-data text-xs" style={{ color: 'var(--foreground)' }}>
              <div>Goalkeepers&nbsp;&nbsp;&nbsp;&nbsp;0.23</div>
              <div>Forwards&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.15</div>
              <div style={{ opacity: 0.75 }}>Full backs&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.07</div>
              <div style={{ opacity: 0.75 }}>Midfielders&nbsp;&nbsp;&nbsp;&nbsp;0.05&ndash;0.07</div>
              <div style={{ opacity: 0.75 }}>Centre backs&nbsp;&nbsp;&nbsp;0.05</div>
            </div>
            <br />
            Three times better at the positions the data covers. Read a
            defender&rsquo;s rating as a decent guide to what he did on the ball
            and a weak one to how well he defended &mdash; and lean on the EA
            half of his OVR, which is set by people who watch him.
            <br /><br />
            Two seasons are worse still. 2014/15 and 2015/16 carry no tackling
            data at all, which is what the previous note is about.
          </div>
        </details>
      </div>

      <div className="flex flex-wrap gap-3 mb-5 items-center">
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search players…"
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)', minWidth: 190 }}
        />
        <select
          value={data?.season ?? season}
          onChange={e => setSeason(e.target.value)}
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          {(data?.seasons ?? []).map(code => (
            <option key={code} value={code}>{seasonLabel(code)}</option>
          ))}
        </select>
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
          value={position}
          onChange={e => changePosition(e.target.value)}
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          <option value="All">All positions</option>
          {Object.entries(POSITIONS).map(([code, name]) => (
            <option key={code} value={code}>{name}</option>
          ))}
        </select>
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Min {minRating}</span>
          <input
            type="range"
            min={0}
            max={95}
            step={1}
            value={minRating}
            onChange={e => setMinRating(Number(e.target.value))}
            style={{ width: 110 }}
          />
        </div>
        {data && (
          <span className="text-xs ml-auto" style={{ color: 'var(--muted-foreground)' }}>
            {data.total.toLocaleString()} players
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
          Loading players…
        </div>
      )}

      {data && (
        <div className="rounded-xl overflow-hidden" style={{ background: 'var(--card)', border: '1px solid var(--border)' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'rgba(255,255,255,0.03)', borderBottom: '1px solid var(--border)' }}>
                  <th
                    className="px-3 py-2 text-left text-xs font-display font-bold uppercase tracking-wider"
                    style={{ color: 'var(--muted-foreground)' }}
                  >
                    Player
                  </th>
                  <SortHeader col="position_group" label="Pos" />
                  <SortHeader col="rating" label="OVR" />
                  <SortHeader col="fifa_overall" label="EA score" />
                  <SortHeader col="performance_rating" label="Perf" />
                  <SortHeader col="minutes" label="Min" />
                  {/*
                    Sortable only once a position is chosen. These columns are
                    position-dependent -- slot 1 is DEF for a centre-back and
                    FIN for a forward -- so sorting one across a mixed table
                    would rank a defender's tackling against a striker's
                    finishing and present it as a single ranking.
                  */}
                  {SLOTS.map(i => {
                    const attribute = selectedAttributes[i];
                    return attribute ? (
                      <SortHeader key={i} col={attribute.key} label={attribute.short} />
                    ) : (
                      <th
                        key={i}
                        className="px-2 py-2 text-xs font-display font-bold uppercase tracking-wider"
                        style={{ color: 'var(--muted-foreground)' }}
                        title="Pick a position to sort by this attribute — it means something different for each one"
                      >
                        {`Attr ${i + 1}`}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {data.players.map((p, i) => {
                  const attributes = ATTRIBUTES[p.position] ?? [];
                  return (
                    <tr
                      key={`${p.player}-${p.team}-${i}`}
                      onClick={() => setSelected(p)}
                      className="cursor-pointer transition-colors hover:bg-white/5"
                      style={{ borderBottom: '1px solid var(--border)' }}
                    >
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-7 h-7 rounded flex items-center justify-center text-xs font-display font-bold flex-shrink-0"
                            style={{ background: teamColor(p.team) + '22', color: teamColor(p.team) }}
                          >
                            {p.player[0]}
                          </div>
                          <div style={{ minWidth: 0 }}>
                            <div className="text-sm font-medium truncate" style={{ color: 'var(--foreground)' }}>
                              {p.player}
                            </div>
                            <div className="text-xs truncate" style={{ color: 'var(--muted-foreground)' }}>
                              {p.team}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-2 py-2 text-center">
                        <span
                          className="text-xs font-display font-bold px-1.5 py-0.5 rounded"
                          style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}
                        >
                          {p.position}
                        </span>
                      </td>
                      <td className="px-2 py-2 text-center">
                        <RatingBadge value={p.rating} />
                      </td>
                      {/*
                        Muted, not badged. These are the inputs to OVR, not
                        peers of it, and three equally loud numbers in a row
                        would leave a reader unsure which one the site means.
                        A dash where EA has no entry, which is under 2% of the current season.
                      */}
                      {/*
                        EA's own number, and only that. The blend does not use
                        it directly -- it uses the positional percentile it maps
                        to -- so a row can read EA 63, Perf 78, OVR 64 and look
                        like bad arithmetic. The conversion is on hover and in
                        the explainer above rather than in the cell, which the
                        second number made unreadable.
                      */}
                      <td
                        className="px-2 py-2 text-center text-xs font-data"
                        style={{ color: 'var(--muted-foreground)' }}
                        title={
                          p.fifa_on_our_scale != null && p.fifa_overall != null
                            ? `EA rates him ${p.fifa_overall} on their global scale. Among ${p.position}s that season that is worth ${p.fifa_on_our_scale} on ours, and ${p.fifa_on_our_scale} is what the blend with the performance rating uses${eaShare(p)}.`
                            : undefined
                        }
                      >
                        {p.fifa_overall ?? '—'}
                      </td>
                      <td className="px-2 py-2 text-center text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
                        {p.performance_rating ?? '—'}
                      </td>
                      <td className="px-2 py-2 text-center text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
                        {p.minutes.toLocaleString()}
                      </td>
                      {SLOTS.map(idx => {
                        const attribute = attributes[idx];
                        const value = attribute ? (p[attribute.key] as number | null) : null;
                        return (
                          <td key={idx} className="px-2 py-2 text-center">
                            <div className="flex flex-col items-center gap-0.5">
                              <RatingBadge value={value} />
                              {attribute && (
                                <span style={{ fontSize: 9, color: 'var(--muted-foreground)' }}>
                                  {attribute.short}
                                </span>
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {data.players.length === 0 && (
            <div className="text-center py-10 text-sm" style={{ color: 'var(--muted-foreground)' }}>
              No players match those filters.
            </div>
          )}
        </div>
      )}

      <div className="mt-3 text-xs" style={{ color: 'var(--muted-foreground)' }}>
        Attribute columns differ by position — the label under each badge says which is which.
        There is no pace or dribbling score: nothing in the underlying data measures them.
      </div>

      {selected && <PlayerDetailPanel player={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
