/** Presentation helpers for turning API values into things the design expects. */

const PALETTE = [
  '#00e676', '#3b82f6', '#f44336', '#ffea00', '#e91e63',
  '#00bcd4', '#ff9100', '#9c27b0', '#8bc34a', '#ff5722',
];

/**
 * A stable colour per team.
 *
 * The mock data carried a hand-picked colour for each club. The API has no such
 * field and inventing a crest palette for ~200 teams is not worth it, so derive
 * one from the name: same team, same colour, every render and every session.
 */
export function teamColor(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) | 0;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

/** Short label for cramped spots. Keeps distinctive words, drops filler. */
export function shortName(name: string): string {
  const cleaned = name.replace(/\b(FC|AFC|CF|AC|SC|SV|RC|US|SS)\b/g, '').trim();
  const words = cleaned.split(/\s+/).filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 3).toUpperCase();
  return words.map(w => w[0]).join('').slice(0, 3).toUpperCase();
}

export function mean(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Format a probability as a whole percentage. */
export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  // Pinned to en-GB rather than the system locale: the rest of the interface is
  // English, and an undefined locale renders dates in whatever language the
  // viewer's OS happens to use, mixing scripts mid-card.
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
}

/**
 * Confidence is the probability of the outcome we actually picked.
 *
 * Note this is bounded below by ~33%: a fixture with three equally likely
 * results still reports 33%, which is the honest reading -- we are not confident
 * at all. Treating anything under ~40% as a coin toss is the right instinct.
 */
export function confidence(p: { prob_home_win: number; prob_draw: number; prob_away_win: number }): number {
  return Math.round(Math.max(p.prob_home_win, p.prob_draw, p.prob_away_win) * 100);
}

/**
 * Below this, naming a winner claims more than the model knows.
 *
 * Measured over 20,013 walk-forward predictions: in the 18% of matches where
 * no outcome reaches 40%, the named pick came in only 40% of the time, and the
 * *draw* was the single most common actual result in 29% of them.
 */
export const NO_FAVOURITE_BELOW = 0.4;

/**
 * How to describe a prediction, given that picking the largest of three
 * probabilities is a lossy summary.
 *
 * "Home win" aggregates 1-0, 2-0, 2-1, 3-1 and so on, while a draw only gets
 * 0-0, 1-1, 2-2 — so a draw is almost never the *largest* of the three even
 * when it is very likely. Across those same 20,013 predictions the label said
 * home 67% / away 33% / draw 0.2%, against actual results of 44 / 31 / 25. The
 * probabilities themselves are well calibrated (draw predicted 25.5%, occurred
 * 25.1%); only this one-word summary of them is skewed.
 *
 * So when nothing clears `NO_FAVOURITE_BELOW`, say so rather than naming a
 * winner the model is not backing.
 */
export function verdict(p: {
  prob_home_win: number;
  prob_draw: number;
  prob_away_win: number;
  predicted_outcome: string;
  home_team: string;
  away_team: string;
}): { label: string; color: string; decisive: boolean } {
  const top = Math.max(p.prob_home_win, p.prob_draw, p.prob_away_win);
  if (top < NO_FAVOURITE_BELOW) {
    return { label: 'Too close to call', color: '#94a3b8', decisive: false };
  }
  if (p.predicted_outcome === 'H') {
    return { label: `${p.home_team} Win`, color: '#00e676', decisive: true };
  }
  if (p.predicted_outcome === 'A') {
    return { label: `${p.away_team} Win`, color: '#3b82f6', decisive: true };
  }
  return { label: 'Draw', color: '#ffea00', decisive: true };
}

/**
 * Band colours for a 0-99 rating, and readable text over them.
 *
 * Moved here from the mock-data module, which the site stopped using once
 * every tab ran on the API but which survived 868 lines longer than it needed
 * to because these two functions lived inside it.
 *
 * The bands are the EA FC convention the Player Ratings page is styled on:
 * green for elite, through lime and amber, to red. Text flips to white only on
 * the darkest band, where black would be unreadable.
 */
export function getRatingBg(r: number): string {
  if (r >= 85) return '#00e676';
  if (r >= 70) return '#76ff03';
  if (r >= 60) return '#ffea00';
  if (r >= 50) return '#ff9100';
  return '#f44336';
}

export function getRatingTextColor(r: number): string {
  return r >= 50 ? '#000' : '#fff';
}
