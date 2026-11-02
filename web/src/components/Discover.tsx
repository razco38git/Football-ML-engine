import { useState } from 'react';
import {
  api,
  competitionLabel,
  seasonLabel,
  type AchievementPoint,
  type ClubAchievement,
  type Prediction,
} from '../api/client';
import { confidence, formatDate, pct, teamColor, verdict } from '../api/display';
import { useAsync } from '../api/hooks';

/**
 * The things worth noticing in this week's fixtures.
 *
 * Everything here is derived from predictions the site already serves — no new
 * endpoint, no new model. The point is that a list of forty fixtures hides its
 * own most interesting rows, and the three questions below are the ones worth
 * asking of a prediction set: where do we disagree with the market, where are
 * we certain, and where are we backing an outsider.
 *
 * Every row is laid out the same way and every number is labelled where it
 * sits. The first version put the fixture on one line and the numbers on
 * another with no words attached — so club names were truncated to fit and a
 * reader was left to guess whether "64% v 79%" meant us against them, home
 * against away, or before against after. A percentage with no noun is not
 * information.
 *
 * Deliberately absent: a "biggest rating risers" section. Player ratings are
 * whole-season numbers and do not move until August, so a weekly risers list
 * would either be empty or be last summer's news dressed as this week's.
 */

/** Market probability for one side, with the bookmakers' margin removed. */
function marketProb(p: Prediction, side: 'home' | 'draw' | 'away'): number | null {
  const raw = [p.market_prob_home, p.market_prob_draw, p.market_prob_away];
  if (raw.some(v => v == null)) return null;
  const total = (raw as number[]).reduce((a, b) => a + b, 0);
  if (total <= 0) return null;
  const index = side === 'home' ? 0 : side === 'draw' ? 1 : 2;
  return (raw as number[])[index] / total;
}

function Card({
  title,
  blurb,
  children,
}: {
  title: string;
  blurb: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="rounded-xl p-5"
      style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
    >
      <div
        className="text-xs font-display font-bold uppercase tracking-wider"
        style={{ color: 'var(--muted-foreground)' }}
      >
        {title}
      </div>
      <div className="text-xs mb-4 mt-1 leading-relaxed" style={{ color: 'var(--muted-foreground)', opacity: 0.85 }}>
        {blurb}
      </div>
      <div className="flex flex-col gap-3">{children}</div>
    </div>
  );
}

/**
 * One fixture: who is playing, what we said, and the numbers behind it.
 *
 * Three lines rather than one. Club names get a full line of their own and are
 * never truncated — "Elche v Real Ma…" reads as a bug, and squeezing a fixture,
 * a verdict and two percentages onto one row is what forced it.
 */
function FixtureRow({ p, children }: { p: Prediction; children: React.ReactNode }) {
  const { label, color } = verdict(p);
  return (
    <div
      className="rounded-lg px-3 py-2"
      style={{ background: 'var(--secondary)', border: '1px solid var(--border)' }}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span
          className="font-data px-1.5 py-0.5 rounded"
          style={{ background: 'var(--card)', color: 'var(--muted-foreground)', fontSize: 9 }}
        >
          {competitionLabel(p.competition && p.competition !== 'domestic' ? p.competition : p.league)}
        </span>
        <span className="font-data" style={{ color: 'var(--muted-foreground)', fontSize: 9 }}>
          {formatDate(p.date)}
        </span>
      </div>

      <div className="text-sm leading-snug mb-1.5" style={{ color: 'var(--foreground)' }}>
        <span style={{ color: teamColor(p.home_team) }}>{p.home_team}</span>
        <span style={{ color: 'var(--muted-foreground)' }}> v </span>
        <span style={{ color: teamColor(p.away_team) }}>{p.away_team}</span>
      </div>

      <div className="text-xs mb-1" style={{ color: 'var(--muted-foreground)' }}>
        We said <span style={{ color, fontWeight: 600 }}>{label}</span>
      </div>

      {children}
    </div>
  );
}

/** A labelled number. The label is the point: a bare percentage says nothing. */
function Figure({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <span className="inline-flex items-baseline gap-1">
      <span style={{ color: 'var(--muted-foreground)', fontSize: 10 }}>{label}</span>
      <span className="font-data font-bold text-xs" style={{ color: color ?? 'var(--foreground)' }}>
        {value}
      </span>
    </span>
  );
}

/**
 * Two questions about the ratings, answered with measurements rather than prose.
 *
 * These are not weekly rows and do not belong beside the fixture cards: they
 * are properties of the project, they change only when the data does, and each
 * one is the answer to a question a reader actually asked. They sit below the
 * fixture zone for that reason.
 *
 * Both answers are deflationary, and that is why they are worth publishing. A
 * site that shows player ratings and match probabilities invites the inference
 * that the two are closely connected. They are not, and the numbers say so.
 */
/**
 * One club's run of residuals, as a baseline with a bar per season.
 *
 * The sparkline is the honest half of the row. A club shown at +35 reads as a
 * verdict on the club; the run shows it is an event — Leicester's other nine
 * seasons all sit inside eight points of their line, and the shape says that
 * faster than any sentence could.
 */
function AchievementRun({
  history,
  season,
  scale,
}: {
  history: AchievementPoint[];
  season: string;
  scale: number;
}) {
  if (history.length < 2) return null;
  return (
    <div className="flex items-center gap-px" style={{ height: 22 }} aria-hidden>
      {history.map(point => {
        const here = point.season === season;
        const height = Math.max(1, (Math.abs(point.over) / scale) * 10);
        return (
          <div
            key={point.season}
            title={`${seasonLabel(point.season)} · ${point.over >= 0 ? '+' : '−'}${Math.abs(point.over).toFixed(1)}`}
            className="flex flex-col justify-center"
            style={{ width: 5, height: 22 }}
          >
            {/* Two halves around a shared midline, so above and below read
                as opposite directions rather than as different sizes. */}
            <div style={{ height: 11, display: 'flex', alignItems: 'flex-end' }}>
              <div
                style={{
                  width: '100%',
                  height: point.over > 0 ? height : 0,
                  background: here ? '#00e676' : 'rgba(0,230,118,0.4)',
                }}
              />
            </div>
            <div style={{ height: 11 }}>
              <div
                style={{
                  width: '100%',
                  height: point.over < 0 ? height : 0,
                  background: here ? '#f44336' : 'rgba(244,67,54,0.4)',
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AchievementRow({ club, scale }: { club: ClubAchievement; scale: number }) {
  const good = club.over > 0;
  return (
    <div
      className="flex items-center gap-3 py-1.5"
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      <div style={{ minWidth: 0, flex: '1 1 auto' }}>
        <div className="text-sm truncate" style={{ color: 'var(--foreground)' }}>
          {club.team}
        </div>
        <div style={{ fontSize: 10, color: 'var(--muted-foreground)' }}>
          {seasonLabel(club.season)} · squad {club.squad_rating.toFixed(1)} · {club.points} pts,
          {' '}expected {club.expected.toFixed(0)}
        </div>
      </div>
      <AchievementRun history={club.history} season={club.season} scale={scale} />
      <span
        className="font-data text-sm font-bold text-right flex-shrink-0"
        style={{ color: good ? '#00e676' : '#f44336', width: 54 }}
      >
        {good ? '+' : '−'}
        {Math.abs(club.over).toFixed(1)}
      </span>
    </div>
  );
}

/**
 * Clubs that took more — or fewer — points than their squad rating explains.
 *
 * The squad rating says how good the players are and the table says what
 * happened. This is the gap, and it is fitted inside each league-season
 * because ratings are not league-relative but points are: a 70-rated side in
 * Ligue 1 plays different opponents from a 70-rated side in England, and
 * eighteen clubs take fewer points between them than twenty do.
 */
function AchievementCard() {
  const [end, setEnd] = useState<'over' | 'under'>('over');
  const { data } = useAsync(() => api.achievement(8), []);
  if (!data || data.over.length === 0) return null;

  const shown = end === 'over' ? data.over : data.under;
  // One scale across both ends, so a bar means the same thing in either list.
  const scale = Math.max(
    ...[...data.over, ...data.under].flatMap(c => c.history.map(h => Math.abs(h.over))),
    1,
  );

  return (
    <Card
      title="Overachievers — and the other kind"
      blurb="Points a club took against the points a squad of its rating took in that same division that season. The line is fitted inside each league-season, so neither a division's scoring level nor a particular year's spread can pass for a club overachieving."
    >
      <div className="flex gap-1 mb-1">
        {(['over', 'under'] as const).map(option => (
          <button
            key={option}
            onClick={() => setEnd(option)}
            className="px-2 py-0.5 rounded text-xs font-display"
            style={{
              background: option === end ? 'rgba(0,230,118,0.15)' : 'var(--secondary)',
              color: option === end ? '#00e676' : 'var(--muted-foreground)',
            }}
          >
            {option === 'over' ? 'Overachieved' : 'Underachieved'}
          </button>
        ))}
      </div>

      <div className="flex flex-col">
        {shown.map(club => (
          <AchievementRow key={`${club.league}-${club.season}-${club.team}`} club={club} scale={scale} />
        ))}
      </div>

      <div className="text-xs leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
        Across {data.club_seasons.toLocaleString()} finished club-seasons the spread of this
        number is{' '}
        <strong style={{ color: 'var(--foreground)' }}>
          {data.residual_sd.toFixed(1)} points
        </strong>
        , so Leicester&rsquo;s +35.2 is four and a half of those — the largest in the data by a
        distance. The bars beside each row are that club&rsquo;s other seasons, and they are
        there to stop the row reading as a verdict on the club: every other Leicester season
        sits inside eight points of its own line. Our rating had that squad{' '}
        <strong style={{ color: 'var(--foreground)' }}>15th of 20</strong>, roughly where EA
        had them — 18th — and where the bookmakers had them at 5000-1. The rating was
        right about the players; the season was the outlier.
      </div>
    </Card>
  );
}

/** The roles a reader can narrow the gems board to, in pitch order. */
const GEM_ROLES = ['All', 'GK', 'CB', 'FB', 'DM', 'MID', 'AMW', 'FWD'];

/**
 * Seasons we rated far above EA, and what EA did about it a year later.
 *
 * The board is the easy half and the claim under it is the point. A list of
 * players "we rated higher than EA" proves nothing on its own — any two
 * rating systems disagree, and EA raises low-rated players anyway, so the
 * disagreements would look prescient whether or not they were. What is
 * published here is the comparison that rules that out: among players EA
 * rated *identically*, the ones we rated higher are the ones EA raises.
 *
 * Each row carries its own test result rather than the card asserting one,
 * which is why EA's next-edition number sits on the row.
 */
function HiddenGemsCard() {
  const [role, setRole] = useState('All');
  const { data } = useAsync(() => api.hiddenGems(10, role), [role]);
  if (!data || data.players.length === 0) return null;

  const claim = data.claim;

  return (
    <Card
      title="Hidden gems — where we disagreed with EA, and who was right"
      blurb="Every player rating is half what we measured from the season and half EA's own, mapped onto one scale. These are the seasons where our half was furthest above theirs — and beside each one, the rating EA gave that player in the next edition."
    >
      <div className="flex flex-wrap gap-1 mb-1">
        {GEM_ROLES.map(option => (
          <button
            key={option}
            onClick={() => setRole(option)}
            className="px-2 py-0.5 rounded text-xs font-display"
            style={{
              background: option === role ? 'rgba(0,230,118,0.15)' : 'var(--secondary)',
              color: option === role ? '#00e676' : 'var(--muted-foreground)',
            }}
          >
            {option}
          </button>
        ))}
      </div>

      <div className="flex flex-col gap-1.5">
        {data.players.map(p => (
          <div
            key={`${p.season}-${p.player}-${p.team}`}
            className="flex items-center gap-3 py-1.5"
            style={{ borderBottom: '1px solid var(--border)' }}
          >
            <div style={{ minWidth: 0, flex: '1 1 auto' }}>
              <div className="text-sm truncate" style={{ color: 'var(--foreground)' }}>
                {p.player}
              </div>
              <div style={{ fontSize: 10, color: 'var(--muted-foreground)' }}>
                {p.team} · {seasonLabel(p.season)} · {p.role}
              </div>
            </div>

            <div className="flex items-baseline gap-1.5 font-data text-xs flex-shrink-0">
              <span title="EA's rating that season" style={{ color: '#f59e0b' }}>
                {p.fifa_overall}
              </span>
              <span style={{ color: 'var(--muted-foreground)' }}>→</span>
              <span title="what we measured, on EA's scale" style={{ color: '#00e676' }}>
                {p.performance_rating}
              </span>
            </div>

            {/* The test, on the row: what EA did next edition. */}
            <div className="text-right flex-shrink-0" style={{ width: 92 }}>
              {p.ea_next == null ? (
                <span style={{ fontSize: 10, color: 'var(--muted-foreground)' }}>
                  no next entry
                </span>
              ) : (
                <>
                  <span
                    className="font-data text-xs font-bold"
                    style={{ color: (p.ea_move ?? 0) > 0 ? '#00e676' : '#f44336' }}
                  >
                    EA {p.ea_next}
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--muted-foreground)' }}>
                    {' '}({(p.ea_move ?? 0) >= 0 ? '+' : '−'}
                    {Math.abs(p.ea_move ?? 0)})
                  </span>
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      {claim && (
        <div className="text-xs leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
          <strong style={{ color: 'var(--foreground)' }}>Does that mean anything?</strong>{' '}
          On its own, no — EA raises low-rated players whoever they are, so any disagreement
          would look prescient. So the test compares only players{' '}
          <strong style={{ color: 'var(--foreground)' }}>EA rated identically</strong>: same
          season, same position, same EA number, {claim.strata.toLocaleString()} such groups
          holding {claim.players.toLocaleString()} players. Within them, the fifth we rated
          highest gained{' '}
          <strong style={{ color: '#00e676' }}>
            {claim.top_fifth_move >= 0 ? '+' : '−'}
            {Math.abs(claim.top_fifth_move).toFixed(2)}
          </strong>{' '}
          next edition and were raised {(claim.top_fifth_raised * 100).toFixed(0)}% of the
          time; the fifth we rated lowest gained{' '}
          <strong style={{ color: '#f44336' }}>
            {claim.bottom_fifth_move >= 0 ? '+' : '−'}
            {Math.abs(claim.bottom_fifth_move).toFixed(2)}
          </strong>{' '}
          and were raised {(claim.bottom_fifth_raised * 100).toFixed(0)}%. Correlation{' '}
          <strong style={{ color: 'var(--foreground)' }}>
            {claim.correlation.toFixed(2)}
          </strong>{' '}
          — real, and far from everything: most of what moves an EA rating is not in our data.
          Seasons {data.excluded_seasons.join(' and ')} are left out entirely, because FBref
          served no defensive or goalkeeping tables for them, and a season under{' '}
          {data.min_minutes.toLocaleString()} minutes is too thin to put here.
        </div>
      )}
    </Card>
  );
}

function InsightsZone() {
  const { data } = useAsync(() => api.ratingInsights(), []);
  if (!data) return null;

  const { lines, overall, one_player: player } = data;
  // The lines ranked by what a reader means by "matters": next season's points.
  const ranked = [...lines].sort((a, b) => b.vs_points - a.vs_points);
  const scale = Math.max(...ranked.map(l => l.vs_points), overall?.vs_points ?? 0, 0.001);

  return (
    <div className="mt-8">
      <h3 className="font-display font-bold text-xl mb-1" style={{ color: 'var(--foreground)' }}>
        Insights
      </h3>
      <p className="text-sm mb-4" style={{ color: 'var(--muted-foreground)' }}>
        Questions worth asking of a rating system, answered on this one&rsquo;s own numbers.
        Both answers are smaller than they sound.
      </p>

      <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))' }}>
        {player && (
          <Card
            title="Can one player decide a match?"
            blurb="Two measurements, and they point the same way. First: take each club's single highest-rated player out and rebuild the rating from the next fifteen. Second: take the squad rating away from the model entirely and re-run the whole walk-forward backtest."
          >
            <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
              <Figure
                label="a club's rating loses, median"
                value={`${player.median_drop.toFixed(2)} pts`}
                color="#00e676"
              />
              <Figure
                label="middle half of clubs"
                value={`${player.q1_drop.toFixed(2)}–${player.q3_drop.toFixed(2)}`}
              />
              <Figure
                label={`most, ${player.max_team} ${seasonLabel(player.max_season)}`}
                value={`${player.max_drop.toFixed(2)} pts`}
                color="#f59e0b"
              />
            </div>
            <div className="text-xs leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
              On a 0&ndash;99 scale, across {player.n.toLocaleString()} club-seasons. Losing the
              best player in the squad moves the team rating by about a point &mdash; and the model
              barely reads the team rating at all: removing squad strength from it costs{' '}
              <strong style={{ color: 'var(--foreground)' }}>
                {player.squad_rps_delta >= 0 ? '+' : '−'}
                {Math.abs(player.squad_rps_delta).toFixed(5)} RPS
              </strong>{' '}
              [{player.squad_rps_low.toFixed(5)}, {player.squad_rps_high.toFixed(5)}] over{' '}
              {player.squad_rps_matches.toLocaleString()} matches, an interval that includes zero.
              So: on these numbers, no. Form, home advantage and Elo decide a prediction, and one
              player reaches none of them.
            </div>
          </Card>
        )}

        {ranked.length > 0 && (
          <Card
            title="Is there a position that matters more?"
            blurb="Each line's rating against the points that club takes the following season — next season, not this one, because that is the only direction a rating is ever read in, and a same-season figure would flatter a rating built from those results."
          >
            <div className="flex flex-col gap-2">
              {[...ranked, ...(overall ? [overall] : [])].map(line => {
                const isOverall = line.line === 'overall';
                return (
                  <div key={line.line} className="flex items-center gap-3">
                    <span
                      className="text-xs w-32 flex-shrink-0"
                      style={{
                        color: isOverall ? 'var(--foreground)' : 'var(--muted-foreground)',
                        fontWeight: isOverall ? 700 : 400,
                      }}
                    >
                      {line.label}
                    </span>
                    <div className="flex-1 h-2 rounded-full" style={{ background: 'var(--secondary)' }}>
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${(line.vs_points / scale) * 100}%`,
                          background: isOverall ? '#a78bfa' : 'linear-gradient(90deg, #00e676, #3b82f6)',
                        }}
                      />
                    </div>
                    <span
                      className="font-data text-xs w-12 text-right flex-shrink-0"
                      style={{ color: 'var(--foreground)' }}
                    >
                      {line.vs_points.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="text-xs leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
              The attack leads, narrowly, and the goalkeeper trails &mdash; but{' '}
              <strong style={{ color: 'var(--foreground)' }}>no single line beats the four together</strong>,
              which is the real answer. Each line is also best at predicting its own job: the attack
              tracks next season&rsquo;s goals scored at{' '}
              {ranked.find(l => l.line === 'attack')?.vs_goals_for.toFixed(2) ?? '—'} and the
              defence tracks goals conceded at{' '}
              {ranked.find(l => l.line === 'defence')?.vs_goals_against.toFixed(2) ?? '—'}. Measured
              over {ranked[0].n.toLocaleString()} club-seasons; a difference under 0.003 is below
              what this data can resolve.
            </div>
          </Card>
        )}

        <HiddenGemsCard />

        <AchievementCard />
      </div>
    </div>
  );
}

export default function Discover() {
  const { data, loading, error } = useAsync(() => api.upcoming('All', false), []);
  // Odds reach the fixture feed only days before kick-off, and never at all for
  // UEFA competitions, so the disagreement card would be empty most of the
  // week. Settled matches always carry them — and they are the better subject
  // anyway, because the result is there to say which of us was right.
  const { data: recent } = useAsync(() => api.matches('All', 60), []);

  if (loading) return <div className="text-sm" style={{ color: 'var(--muted-foreground)' }}>Loading…</div>;
  if (error) return <div className="text-sm" style={{ color: '#f44336' }}>{error}</div>;
  if (!data || data.length === 0) {
    return (
      <div className="text-sm" style={{ color: 'var(--muted-foreground)' }}>
        Nothing scheduled in the next few days. This fills up when the next round does.
      </div>
    );
  }

  const disagreements = (recent ?? [])
    .map(p => {
      const side = p.predicted_outcome === 'H' ? 'home' : p.predicted_outcome === 'A' ? 'away' : 'draw';
      const ours =
        side === 'home' ? p.prob_home_win : side === 'away' ? p.prob_away_win : p.prob_draw;
      const theirs = marketProb(p, side);
      return theirs == null ? null : { p, gap: ours - theirs, ours, theirs };
    })
    .filter((x): x is { p: Prediction; gap: number; ours: number; theirs: number } => x != null)
    .sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap))
    .slice(0, 5);

  const confident = [...data]
    .filter(p => verdict(p).decisive)
    .sort((a, b) => confidence(b) - confidence(a))
    .slice(0, 5);

  const upsets = data
    .filter(
      p =>
        p.strength_home_overall != null &&
        p.strength_away_overall != null &&
        p.predicted_outcome !== 'D',
    )
    .map(p => {
      const backingHome = p.predicted_outcome === 'H';
      const ours = backingHome ? p.prob_home_win : p.prob_away_win;
      const underdog = backingHome ? p.home_team : p.away_team;
      const favourite = backingHome ? p.away_team : p.home_team;
      const theirStrength = backingHome ? p.strength_home_overall! : p.strength_away_overall!;
      const otherStrength = backingHome ? p.strength_away_overall! : p.strength_home_overall!;
      return { p, deficit: otherStrength - theirStrength, ours, underdog, favourite };
    })
    .filter(x => x.deficit > 0)
    .sort((a, b) => b.deficit - a.deficit)
    .slice(0, 5);

  return (
    <div>
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Discover
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          The rows worth noticing, pulled out of a fixture list that hides them.
          Nothing here is a new prediction &mdash; it is the same numbers the Match
          Predictor shows, sorted by the questions worth asking.
        </p>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))' }}>
        {disagreements.length > 0 && (
          <Card
            title="Where we disagreed with the bookmakers"
            blurb="Recent matches where our probability for our own pick was furthest from the market's probability for that same outcome. The tick says whether the pick came in — the market usually wins these. A match only appears once we have a closing price for it, so this list is shorter than Recent results rather than out of date."
          >
            {disagreements.map(({ p, gap, ours, theirs }) => {
              const settled = p.actual_result !== null;
              const right = settled && p.actual_result === p.predicted_outcome;
              return (
                <FixtureRow key={`${p.date}-${p.home_team}-${p.away_team}`} p={p}>
                  <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                    <Figure label="we gave it" value={pct(ours)} color="#00e676" />
                    <Figure label="bookmakers gave it" value={pct(theirs)} color="#f59e0b" />
                    <Figure
                      label={gap > 0 ? 'we were higher by' : 'we were lower by'}
                      value={pct(Math.abs(gap))}
                    />
                  </div>
                  {settled && (
                    <div className="text-xs mt-1.5" style={{ color: right ? '#00e676' : '#f44336' }}>
                      {right ? '✓ it came in' : '✗ it did not'}
                      <span style={{ color: 'var(--muted-foreground)' }}>
                        {' — '}final score {p.actual_home_goals}–{p.actual_away_goals}
                      </span>
                    </div>
                  )}
                </FixtureRow>
              );
            })}
          </Card>
        )}

        {confident.length > 0 && (
          <Card
            title="Where we are most certain"
            blurb="The coming round's strongest calls: the highest probability we have given to any single outcome. Nothing here is close to a certainty — our firmest view of a football match is usually around four in five."
          >
            {confident.map(p => (
              <FixtureRow key={`${p.date}-${p.home_team}-${p.away_team}`} p={p}>
                <Figure label="we give that" value={`${confidence(p)}%`} color="#00e676" />
              </FixtureRow>
            ))}
          </Card>
        )}

        {upsets.length > 0 && (
          <Card
            title="Outsiders we are backing"
            blurb="The coming round, where we favour the side with the weaker squad on paper — because recent form, home advantage and Elo outvoted the squad rating. Squad rating is one input of several, and only about 5% of what drives a prediction."
          >
            {upsets.map(({ p, deficit, ours, underdog, favourite }) => (
              <FixtureRow key={`${p.date}-${p.home_team}-${p.away_team}`} p={p}>
                <Figure label="we give that" value={pct(ours)} color="#a78bfa" />
                <div className="text-xs mt-1.5" style={{ color: 'var(--muted-foreground)' }}>
                  {underdog}&rsquo;s squad rates{' '}
                  <strong style={{ color: 'var(--foreground)' }}>{deficit.toFixed(1)} lower</strong>{' '}
                  than {favourite}&rsquo;s
                </div>
              </FixtureRow>
            ))}
          </Card>
        )}
      </div>

      <InsightsZone />

      <p className="text-xs mt-5 leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
        No &ldquo;biggest risers&rdquo; list, deliberately: player ratings are
        whole-season numbers and do not move until August, so a weekly risers
        board would be last summer&rsquo;s news dressed as this week&rsquo;s. Team
        Elo does move every match &mdash; see the chart on any club in Team
        Strength.
      </p>
    </div>
  );
}
