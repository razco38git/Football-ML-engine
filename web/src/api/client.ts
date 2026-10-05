/**
 * Client for the Football ML API.
 *
 * Types here mirror the Pydantic schemas in `src/footballml/api/schemas.py`.
 * They are hand-written for now; once the contract settles, generate them from
 * the OpenAPI document (`openapi-typescript http://localhost:8000/openapi.json`)
 * so the two cannot drift apart silently.
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

export interface TeamForm {
  team: string;
  /** W/D/L, oldest first */
  results: string[];
  xg_for: number[];
  xg_against: number[];
  shots: number[];
  shots_on_target: number[];
  goals_for: number[];
  goals_against: number[];
  opponents: string[];
  venues: string[];
}

export interface Driver {
  feature: string;
  label: string;
  contribution: number;
}

export interface LikelyScore {
  home: number;
  away: number;
  probability: number;
}

export interface Prediction {
  league: string;
  date: string;
  home_team: string;
  away_team: string;

  /**
   * True when the two clubs play in different divisions and the cross-league
   * correction was applied. Always false for a real fixture — the model has
   * never trained on a match between two leagues, so it cannot judge the gap
   * itself and the correction is fitted from actual European results.
   */
  league_adjusted: boolean;
  /** Each club's own division. `league` is the home side's for a hypothetical. */
  home_division: string | null;
  away_division: string | null;
  /**
   * `"domestic"`, or a UEFA code like `"UCL"`. Note `league` is the *home
   * side's division* even for a European tie — the model has no league code for
   * a competition, and a missing one is a value it never trained on.
   */
  competition: string | null;

  expected_goals_home: number;
  expected_goals_away: number;
  prob_home_win: number;
  prob_draw: number;
  prob_away_win: number;
  predicted_outcome: 'H' | 'D' | 'A';
  /** Probability of that exact scoreline — typically ~10%, unlike the 1X2 split. */
  prob_modal_score: number | null;
  modal_score_home: number;
  modal_score_away: number;
  /**
   * The likeliest few scorelines, likeliest first; the first is the modal one
   * above. Shown as a group because one score on its own reads as the forecast
   * and appears to contradict the outcome — an outcome sums a whole triangle of
   * scorelines while a draw's mass sits on the diagonal, so the leader is a
   * draw in 63% of matches. Empty for predictions stored before this existed.
   */
  likely_scores: LikelyScore[];
  /**
   * The whole scoreline distribution, `score_grid[homeGoals][awayGoals]`, 0–5
   * each way. The top scorelines answer "which exact score" and the 1X2 split
   * answers "who wins", and a reader given both still has to take on trust
   * that 1-1 at 11% and a 57% home win are consistent. This is the proof: the
   * diagonal is every draw, the triangle below it every home win.
   */
  score_grid: number[][];
  prob_over_2_5: number;
  prob_btts: number;

  market_prob_home: number | null;
  market_prob_draw: number | null;
  market_prob_away: number | null;

  /** Last season's squad ratings, as the model saw them. Null for a promoted side. */
  strength_home_overall: number | null;
  strength_home_attack: number | null;
  strength_home_defence: number | null;
  strength_away_overall: number | null;
  strength_away_attack: number | null;
  strength_away_defence: number | null;

  actual_home_goals: number | null;
  actual_away_goals: number | null;
  actual_result: 'H' | 'D' | 'A' | null;

  drivers_home: Driver[];
  drivers_away: Driver[];
  form_home: TeamForm | null;
  form_away: TeamForm | null;
}

export interface ModelInfo {
  version: string;
  trained_at: string;
  trained_through: string;
  n_train: number;
  n_features: number;
  leagues: string[];
  rho: number;
  metrics: Record<string, number>;
}

export interface Health {
  status: string;
  model_version: string;
  trained_through: string;
  n_train: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    // FastAPI puts the useful message in `detail`; fall back to the status text
    // when the body is not JSON (a proxy error page, say).
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => get<Health>('/health'),

  leagues: () => get<Record<string, string>>('/leagues'),

  models: () => get<ModelInfo[]>('/models'),

  /** Fixtures that have not been played. May legitimately be empty mid-week. */
  upcoming: (league?: string, explain = true) => {
    const params = new URLSearchParams();
    if (league && league !== 'All') params.set('league', league);
    if (explain) params.set('explain', 'true');
    return get<Prediction[]>(`/fixtures/upcoming?${params}`);
  },

  /** Recent played matches, scored retrospectively by the current model. */
  matches: (league?: string, limit = 50) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (league && league !== 'All') params.set('league', league);
    return get<Prediction[]>(`/matches?${params}`);
  },

  /** The rated player database, filtered and sorted server-side. */
  players: (query: PlayerQuery = {}) => fetchPlayers(query),

  /** Every rated season for one player, newest first. */
  playerHistory: (name: string) => fetchPlayerHistory(name),

  /** Players who most resemble this one. See `SimilarPlayers` for the two axes. */
  similar: (name: string, options: SimilarQuery = {}) => fetchSimilar(name, options),

  /** Team ratings built from player ratings. */
  teams: (league?: string, limit = 100, season?: string) => fetchTeams(league, limit, season),

  /** The players behind a team's rating. */
  squad: (team: string, season?: string) => fetchSquad(team, season),

  /** How the five leagues compare on average squad rating. */
  leagueStrength: (season?: string) => fetchLeagueStrength(season),

  /** Settled matches: prediction beside the real score. */
  history: (source: 'live' | 'backtest', league?: string, limit = 50) =>
    fetchHistory(source, league, limit),

  /** How a team's Elo has moved, match by match. */
  teamElo: (team: string, seasons = 3) =>
    get<EloHistory>(`/teams/${encodeURIComponent(team)}/elo?seasons=${seasons}`),

  /** Projected final table for one league. */
  projection: (league: string) => fetchProjection(league),

  /** Any pairing, scored live using each side's form as of today. */
  predict: async (home_team: string, away_team: string, explain = true) => {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team, away_team, explain }),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        detail = (await response.json()).detail ?? detail;
      } catch {
        /* keep statusText */
      }
      throw new ApiError(detail, response.status);
    }
    return (await response.json()) as Prediction;
  },
};

/**
 * UEFA competitions the site shows fixtures for.
 *
 * Deliberately separate from `LEAGUE_NAMES`. Five components use that map to
 * build dropdowns for things that only exist per division — squad lists,
 * projected tables, team ratings — and a "Champions League" option there is
 * meaningless or an outright 404.
 */
export const COMPETITION_NAMES: Record<string, string> = {
  UCL: 'Champions League',
};

/** Everything that can label a prediction. Display lookups only, never a dropdown. */
export const competitionLabel = (code: string): string =>
  LEAGUE_NAMES[code] ?? COMPETITION_NAMES[code] ?? code;

/** Division codes to the names the design already uses. */
export const LEAGUE_NAMES: Record<string, string> = {
  E0: 'Premier League',
  SP1: 'LaLiga',
  D1: 'Bundesliga',
  I1: 'Serie A',
  F1: 'Ligue 1',
};

export interface PlayerRating {
  player: string;
  team: string;
  league: string;
  season: string;
  /** GK, D, M or F */
  position: string;
  minutes: number;
  rating: number | null;
  rated: boolean;
  unrated_reason: string | null;

  /**
   * The two halves behind `rating`, blended 50/50. They disagree often — a
   * player can hold a high EA overall on reputation while this season's output
   * says otherwise. `fifa_overall` is null where EA has no entry — 3.9% across all seasons, 1.6% in the current one.
   */
  fifa_overall: number | null;
  performance_rating: number | null;
  /**
   * EA's overall put on *our* scale, which is what the blend uses.
   * The printed EA number is a global scale and ours is a rank within
   * position and season, so they are mapped onto one scale before
   * blending — which is why 63 and 78 can produce 64.
   */
  fifa_on_our_scale: number | null;
  /** 0-1: how much of the role the season's data supported. Below 1, the blend leans on EA. */
  measured_share: number | null;
  /** Where the EA half of the blend landed for this row. 0.5 almost everywhere. */
  fifa_weight_used: number | null;

  sub_finishing: number | null;
  sub_creation: number | null;
  sub_involvement: number | null;
  sub_volume: number | null;
  sub_defending: number | null;
  sub_shot_stopping: number | null;
  sub_reliability: number | null;
  sub_workload: number | null;
  sub_penalties: number | null;

  goals: number | null;
  assists: number | null;
  np_xg: number | null;
  xa: number | null;
  key_passes_per90: number | null;
  interceptions_per90: number | null;
  tackles_won_per90: number | null;
  save_pct: number | null;
  goals_against_per90: number | null;
}

/** A season label like `"2526"` shown as `2025/26`. */
export const seasonLabel = (code: string): string =>
  code.length === 4 ? `20${code.slice(0, 2)}/${code.slice(2)}` : code;

export interface PlayerPage {
  total: number;
  players: PlayerRating[];
  /** Every season with ratings, newest first. Drives the selector. */
  seasons: string[];
  /**
   * The season these rows are from. Early in a campaign the default is the
   * *previous* season — too few players have the minutes to be rated yet — so
   * the page has to say which one it is showing, or last season's clubs read
   * as stale data.
   */
  season: string | null;
}

export interface PlayerQuery {
  league?: string;
  position?: string;
  season?: string;
  search?: string;
  min_rating?: number;
  sort?: string;
  descending?: boolean;
  limit?: number;
  offset?: number;
}

/** The rated player database, filtered and sorted server-side. */
export function fetchPlayers(query: PlayerQuery = {}): Promise<PlayerPage> {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== '') {
      params.set(key, String(value));
    }
  }
  return get<PlayerPage>(`/players?${params}`);
}

/** Every rated season for one player, newest first. */
export function fetchPlayerHistory(name: string): Promise<PlayerRating[]> {
  return get<PlayerRating[]>(`/players/${encodeURIComponent(name)}`);
}

export interface SimilarQuery {
  season?: string;
  limit?: number;
  /** False widens the search to other positions, dropping the percentile axis. */
  same_role?: boolean;
}

export interface SimilarPlayer {
  player: string;
  team: string;
  league: string;
  season: string;
  position: string;
  rating: number | null;
  minutes: number;

  /**
   * Both scores are 0-100, and 50 means "no more alike than two random players
   * in this position" -- the scale is pinned to the median distance in the pool,
   * not to the theoretical range.
   *
   * Either can be null, and null is not zero. `fifa_similarity` is null when a
   * player has no EA entry (1.6% of the current season); `percentile_similarity` is null
   * whenever the comparison crosses positions, because our sub-ratings are
   * ranks within a position and a centre-back's 80 is not a winger's 80.
   */
  fifa_similarity: number | null;
  percentile_similarity: number | null;
  combined: number | null;

  attributes: Record<string, number>;
  sub_ratings: Record<string, number>;
}

export interface SimilarPlayers {
  player: PlayerRating;
  /** The six outfield attributes, or the keeper's five. */
  attribute_names: string[];
  player_attributes: Record<string, number>;
  player_sub_ratings: Record<string, number>;
  same_role: boolean;
  results: SimilarPlayer[];
}

/** Players who most resemble this one, on EA's attributes and on our own. */
export function fetchSimilar(name: string, options: SimilarQuery = {}): Promise<SimilarPlayers> {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(options)) {
    if (value !== undefined && value !== null && value !== '') {
      params.set(key, String(value));
    }
  }
  return get<SimilarPlayers>(`/players/${encodeURIComponent(name)}/similar?${params}`);
}

export interface TeamStrength {
  team: string;
  league: string;
  season: string;
  /** eleven | best_n | squad */
  method: string;
  n_players: number;
  strength_overall: number;
  strength_goalkeeper: number | null;
  strength_defence: number | null;
  strength_midfield: number | null;
  strength_attack: number | null;
}

export interface TeamStrengthPage {
  teams: TeamStrength[];
  /** Every season with team ratings, newest first. */
  seasons: string[];
  /** The season these rows are from. */
  season: string | null;
}

/** Team ratings built from player ratings, strongest first. */
export function fetchTeams(
  league?: string,
  limit = 100,
  season?: string,
): Promise<TeamStrengthPage> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (league && league !== 'All') params.set('league', league);
  if (season) params.set('season', season);
  return get<TeamStrengthPage>(`/teams?${params}`);
}

export interface LeagueStrength {
  league: string;
  season: string;
  n_teams: number;
  /** Mean squad rating across the league's clubs — the headline number. */
  mean_strength: number;
  median_strength: number;
  /**
   * Spread between clubs. A high mean with a high spread is a league carried
   * by a few sides rather than strong throughout.
   */
  spread: number;
  strongest_team: string;
  strongest_strength: number;
  weakest_team: string;
  weakest_strength: number;
}

/**
 * How the five leagues compare, strongest first.
 *
 * Comparable because a rating is not league-relative: EA's overall is a global
 * scale and the performance percentiles pool all five leagues. The caveat is
 * that per-90 output is easier to accumulate against weaker opponents, which
 * flatters the weaker leagues — so the real gaps are, if anything, wider.
 */
export function fetchLeagueStrength(season?: string): Promise<LeagueStrength[]> {
  const params = new URLSearchParams();
  if (season) params.set('season', season);
  return get<LeagueStrength[]>(`/leagues/strength?${params}`);
}

/** The players a team's rating was built from, highest minutes first. */
export function fetchSquad(team: string, season?: string): Promise<PlayerRating[]> {
  const params = season ? `?season=${encodeURIComponent(season)}` : '';
  return get<PlayerRating[]>(`/teams/${encodeURIComponent(team)}/squad${params}`);
}

export interface MatchResult {
  /** live = stored before kickoff; backtest = walk-forward, model never saw it */
  source: 'live' | 'backtest';
  league: string;
  date: string;
  home_team: string;
  away_team: string;

  predicted_outcome: 'H' | 'D' | 'A';
  prob_home_win: number;
  prob_draw: number;
  prob_away_win: number;
  expected_goals_home: number | null;
  expected_goals_away: number | null;

  actual_home_goals: number;
  actual_away_goals: number;
  actual_result: 'H' | 'D' | 'A';
  correct: boolean;
  /** Probability assigned to the outcome we picked, 0-100. */
  confidence: number;
}

export interface CompetitionAccuracy {
  n: number;
  accuracy: number;
  rps: number;
  rps_base_rate: number;
}

export interface EloPoint {
  date: string;
  season: string;
  elo: number;
  opponent: string;
  venue: string;
}

/**
 * A team's rating over time.
 *
 * Elo is the only rating here that moves *within* a season — player ratings and
 * squad strength are whole-season numbers, fixed until August — so it is the
 * one series where a trend line means anything.
 */
export interface EloHistory {
  team: string;
  current: number;
  points: EloPoint[];
}

export interface Accuracy {
  n: number;
  accuracy: number;
  rps: number;
  log_loss: number;
  brier: number;
  rps_base_rate: number;
  rps_market: number | null;
  /** Share won by the most common outcome — always backing the home side. */
  accuracy_base_rate: number;
  /** How often the bookmakers' shortest price won, where odds exist. */
  accuracy_market: number | null;
  /**
   * Share of settled matches whose single likeliest scoreline was exactly
   * right. Expect roughly one in eight: the leader in a scoreline distribution
   * typically carries only 10-13%, so this is a far harder bar than the
   * outcome and is published to keep that distinction visible.
   */
  exact_score: number | null;
  /** Accuracy, RPS and base rate per league — the same block, never pooled. */
  by_league: Record<string, CompetitionAccuracy>;
  /**
   * Per competition, kept apart rather than pooled. Measured skill over a base
   * rate is 12.8% on domestic matches against 6.2% on cross-league ones, so one
   * blended figure would overstate the European ties.
   */
  by_competition: Record<string, CompetitionAccuracy>;
  calibration: {
    bin_lower: number;
    bin_upper: number;
    n: number;
    mean_predicted: number;
    observed_rate: number;
  }[];
}

export interface MatchResultPage {
  source: string;
  total: number;
  summary: Accuracy | null;
  matches: MatchResult[];
}

/** Settled matches with the prediction beside the real score, newest first. */
export function fetchHistory(
  source: 'live' | 'backtest',
  league?: string,
  limit = 50,
): Promise<MatchResultPage> {
  const params = new URLSearchParams({ source, limit: String(limit) });
  if (league && league !== 'All') params.set('league', league);
  return get<MatchResultPage>(`/accuracy/history?${params}`);
}

export interface ProjectedTeam {
  team: string;
  played: number;
  points: number;
  goal_difference: number;

  projected_points: number;
  /** 10th and 90th percentile across simulated seasons — the honest spread. */
  points_low: number;
  points_high: number;
  projected_position: number;

  title_pct: number;
  top_four_pct: number;
  relegation_pct: number;
}

export interface SeasonProjection {
  league: string;
  season: string;
  /** Matches still to play — the basis for reading the table sceptically. */
  remaining: number;
  played: number;
  teams: ProjectedTeam[];
}

/** Projected final table, from simulating every remaining fixture. */
export function fetchProjection(league: string): Promise<SeasonProjection> {
  return get<SeasonProjection>(`/projections?league=${encodeURIComponent(league)}`);
}
