import { api, competitionLabel, type Prediction } from '../api/client';
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
            blurb="Recent matches where our probability for our own pick was furthest from the market's probability for that same outcome. The tick says whether the pick came in — the market usually wins these."
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
