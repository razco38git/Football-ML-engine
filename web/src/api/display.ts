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
