/**
 * The pieces that say how a forecast turned out, shared by the two pages that
 * show settled matches.
 *
 * Prediction Accuracy and Match Predictor's "Recent results" are the same
 * statement about the same match, and they were drawn twice from two different
 * types — `MatchResult` and `Prediction`. They had already drifted: the
 * probability bar was `h-1.5` on one page and `h-2` on the other, the
 * correct/incorrect badge was a component in one file and inline markup in the
 * other, and only one of them named the teams under the bar or said which
 * outcome actually happened.
 *
 * So these take the four or five fields they need rather than either row type,
 * which is why neither `MatchResult` nor `Prediction` is imported here.
 */

import { NO_FAVOURITE_BELOW } from '../api/display';

type Outcome = 'H' | 'D' | 'A';

/** Which side an outcome letter refers to, in words a reader recognises. */
export function outcomeLabel(o: Outcome | string, home: string, away: string): string {
  return o === 'H' ? home : o === 'A' ? away : 'Draw';
}

/** The tick or cross. Green for a call that came off, red for one that did not. */
export function ResultBadge({ correct }: { correct: boolean }) {
  return (
    <span
      className="text-xs font-display font-bold px-2 py-0.5 rounded"
      style={{
        background: correct ? 'rgba(0,230,118,0.15)' : 'rgba(244,67,54,0.15)',
        color: correct ? '#00e676' : '#f44336',
      }}
    >
      {correct ? '✓' : '✗'}
    </span>
  );
}

/** The three-way probability bar: home green, draw grey, away blue. */
export function OutcomeBar({
  home,
  draw,
  away,
}: {
  home: number;
  draw: number;
  away: number;
}) {
  const segments = [
    { value: home, color: '#00e676' },
    { value: draw, color: '#64748b' },
    { value: away, color: '#3b82f6' },
  ];
  return (
    <div
      className="flex h-1.5 rounded-full overflow-hidden"
      style={{ background: 'var(--secondary)' }}
    >
      {segments.map((s, i) => (
        <div key={i} style={{ width: `${s.value * 100}%`, background: s.color }} />
      ))}
    </div>
  );
}

/**
 * The percentages under the bar, named and with the real outcome picked out.
 *
 * "Home 59% Draw 22% Away 18%" makes the reader map two of the three onto
 * clubs themselves, and says nothing about which one happened. Where the match
 * is settled, the outcome that did happen is brightened — that is the only
 * thing on the row that answers the question the reader came with.
 */
export function OutcomeSplit({
  home,
  away,
  probHome,
  probDraw,
  probAway,
  actual,
  pct,
}: {
  home: string;
  away: string;
  probHome: number;
  probDraw: number;
  probAway: number;
  actual?: Outcome | string | null;
  pct: (v: number) => string;
}) {
  const cells: Array<[Outcome, string, number]> = [
    ['H', home, probHome],
    ['D', 'Draw', probDraw],
    ['A', away, probAway],
  ];
  return (
    <div
      className="flex justify-between mt-1.5 text-xs font-data"
      style={{ color: 'var(--muted-foreground)' }}
    >
      {cells.map(([key, name, value]) => (
        <span
          key={key}
          className="truncate"
          style={actual === key ? { color: 'var(--foreground)' } : undefined}
        >
          {name} {pct(value)}
        </span>
      ))}
    </div>
  );
}

/**
 * "We said Villarreal — It was Villarreal."
 *
 * Two words each, and they do the work a column header cannot: "predicted" and
 * "actual" beside three numbers left the reader to work out which of them was
 * the claim and which was the result.
 */
export function CallVersusResult({
  predicted,
  actual,
  home,
  away,
  correct,
  probHome,
  probDraw,
  probAway,
}: {
  predicted: Outcome | string;
  actual: Outcome | string;
  home: string;
  away: string;
  correct: boolean;
  /** The three probabilities, so this can agree with the headline above it. */
  probHome?: number;
  probDraw?: number;
  probAway?: number;
}) {
  // A card showed "Too close to call" in the analysis and "We said Mainz" in
  // the footer, on 38 / 24 / 38. Both came from the same prediction: the
  // headline asks `verdict`, which refuses to name a winner below
  // NO_FAVOURITE_BELOW, while this line read `predicted_outcome` raw -- and
  // that is an argmax, which breaks a dead heat by whichever side it happens
  // to see first. The footer now asks the same question as the headline.
  const known = probHome != null && probDraw != null && probAway != null;
  const backed = !known || Math.max(probHome, probDraw, probAway) >= NO_FAVOURITE_BELOW;

  return (
    <div
      className="flex justify-between mt-1 text-xs"
      style={{ color: 'var(--muted-foreground)' }}
    >
      <span>
        {backed ? (
          <>
            We said{' '}
            <span style={{ color: correct ? '#00e676' : '#f44336', fontWeight: 500 }}>
              {outcomeLabel(predicted, home, away)}
            </span>
          </>
        ) : (
          // Deliberately not coloured by `correct`. We did not back anyone, so
          // there is nothing here to be right or wrong about -- the tick beside
          // it is the published accuracy metric, which scores the top
          // probability whether or not it amounts to a favourite.
          <span title={`No outcome reached ${Math.round(NO_FAVOURITE_BELOW * 100)}%, so the model named no favourite. The tick scores its highest probability anyway, which is how accuracy is counted.`}>
            We called it <span style={{ fontWeight: 500 }}>too close</span>
          </span>
        )}
      </span>
      <span>
        It was{' '}
        <span style={{ color: 'var(--foreground)' }}>
          {outcomeLabel(actual, home, away)}
        </span>
      </span>
    </div>
  );
}

/**
 * The headline of a settled match: what happened, and what we had said would.
 *
 * The forecast is deliberately the smaller line. A card that leads with its own
 * prediction and footnotes the result is answering a question nobody asked once
 * the match has been played.
 */
export function FinalScore({
  homeGoals,
  awayGoals,
  expectedHome,
  expectedAway,
}: {
  homeGoals: number;
  awayGoals: number;
  expectedHome?: number | null;
  expectedAway?: number | null;
}) {
  return (
    <>
      <div
        className="uppercase tracking-wider"
        style={{ color: 'var(--muted-foreground)', fontSize: 9 }}
      >
        Final score
      </div>
      <div className="font-display font-black text-xl" style={{ color: 'var(--foreground)' }}>
        {homeGoals}–{awayGoals}
      </div>
      {expectedHome != null && (
        <div
          className="text-xs font-data"
          style={{ color: 'var(--muted-foreground)' }}
          title="Goals the model expected each side to score, from before kick-off. Not the chances they actually created."
        >
          we forecast {expectedHome.toFixed(2)}–{expectedAway?.toFixed(2)}
        </div>
      )}
    </>
  );
}
