import { api, competitionLabel, type Prediction } from '../api/client';
import { confidence, pct, teamColor, verdict } from '../api/display';
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
      <div className="text-xs mb-3 mt-1" style={{ color: 'var(--muted-foreground)', opacity: 0.8 }}>
        {blurb}
      </div>
      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}

function Row({ p, right }: { p: Prediction; right: React.ReactNode }) {
  const { label, color } = verdict(p);
  return (
    <div className="flex items-center gap-3">
      <span
        className="text-xs px-1.5 py-0.5 rounded shrink-0 font-data"
        style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)', fontSize: 9 }}
      >
        {competitionLabel(p.competition && p.competition !== 'domestic' ? p.competition : p.league)}
      </span>
      <span className="text-sm truncate" style={{ color: 'var(--foreground)', flex: '1 1 0', minWidth: 0 }}>
        <span style={{ color: teamColor(p.home_team) }}>{p.home_team}</span>
        <span style={{ color: 'var(--muted-foreground)' }}> v </span>
        <span style={{ color: teamColor(p.away_team) }}>{p.away_team}</span>
      </span>
      <span className="text-xs shrink-0 truncate" style={{ color, maxWidth: 110 }}>
        {label}
      </span>
      <span className="font-data font-bold text-sm shrink-0 text-right" style={{ width: 86 }}>
        {right}
      </span>
    </div>
  );
}

export default function Discover() {
  const { data, loading, error } = useAsync(() => api.upcoming('All', false), []);
  // Odds reach the fixture feed only days before kick-off, and never at all for
  // UEFA competitions, so the disagreement card would be empty most of the
  // week. Settled matches always carry them -- and they are the better subject
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

  // Where we and the market most disagree about the same outcome. Only the
  // fixtures that carry odds can appear, which is domestic ones: football-data
  // publishes no prices for UEFA competitions.
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
    .slice(0, 6);

  const confident = [...data]
    .filter(p => verdict(p).decisive)
    .sort((a, b) => confidence(b) - confidence(a))
    .slice(0, 6);

  // An outsider we like: the side with the weaker squad given the better chance.
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
      const theirStrength = backingHome ? p.strength_home_overall! : p.strength_away_overall!;
      const otherStrength = backingHome ? p.strength_away_overall! : p.strength_home_overall!;
      return { p, deficit: otherStrength - theirStrength, ours };
    })
    .filter(x => x.deficit > 0)
    .sort((a, b) => b.deficit - a.deficit)
    .slice(0, 6);

  return (
    <div>
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Discover
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          The rows worth noticing in the coming round, pulled out of a fixture list
          that hides them. Nothing here is a new prediction &mdash; it is the same
          numbers the Match Predictor shows, sorted by the questions worth asking.
        </p>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))' }}>
        {disagreements.length > 0 && (
          <Card
            title="Where we disagreed with the bookmakers"
            blurb="Recent matches, our probability for our own pick against theirs — with a tick if we were right. The market usually wins these."
          >
            {disagreements.map(({ p, gap, ours, theirs }) => {
              const settled = p.actual_result !== null;
              const right = settled && p.actual_result === p.predicted_outcome;
              return (
                <Row
                  key={`${p.league}-${p.date}-${p.home_team}`}
                  p={p}
                  right={
                    <span style={{ color: gap > 0 ? '#00e676' : '#f59e0b' }}>
                      {gap > 0 ? '+' : '−'}
                      {pct(Math.abs(gap))}
                      {settled && (
                        <span style={{ color: right ? '#00e676' : '#f44336' }}> {right ? '✓' : '✗'}</span>
                      )}
                      <span
                        className="font-data block"
                        style={{ color: 'var(--muted-foreground)', fontSize: 9, fontWeight: 400 }}
                      >
                        {pct(ours)} v {pct(theirs)}
                      </span>
                    </span>
                  }
                />
              );
            })}
          </Card>
        )}

        {confident.length > 0 && (
          <Card
            title="Where we are most certain"
            blurb="The highest probability assigned to any single outcome this round."
          >
            {confident.map(p => (
              <Row
                key={`${p.date}-${p.home_team}-${p.away_team}`}
                p={p}
                right={<span style={{ color: '#00e676' }}>{confidence(p)}%</span>}
              />
            ))}
          </Card>
        )}

        {upsets.length > 0 && (
          <Card
            title="Outsiders we are backing"
            blurb="The weaker squad on paper, favoured anyway — because form, venue and Elo outvoted the squad rating."
          >
            {upsets.map(({ p, deficit, ours }) => (
              <Row
                key={`${p.date}-${p.home_team}-${p.away_team}`}
                p={p}
                right={
                  <span style={{ color: '#a78bfa' }}>
                    {pct(ours)}
                    <span
                      className="font-data block"
                      style={{ color: 'var(--muted-foreground)', fontSize: 9, fontWeight: 400 }}
                    >
                      −{deficit.toFixed(1)} squad
                    </span>
                  </span>
                }
              />
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
