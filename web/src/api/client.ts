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

export interface Prediction {
  league: string;
  date: string;
  home_team: string;
  away_team: string;

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
  teams: (league?: string, limit = 100) => fetchTeams(league, limit),

  /** The players behind a team's rating. */
  squad: (team: string) => fetchSquad(team),

  /** Settled matches: prediction beside the real score. */
  history: (source: 'live' | 'backtest', league?: string, limit = 50) =>
    fetchHistory(source, league, limit),

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

export interface PlayerPage {
  total: number;
  players: PlayerRating[];
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
   * player has no EA entry (8.6% of them); `percentile_similarity` is null
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

/** Team ratings built from player ratings, strongest first. */
export function fetchTeams(league?: string, limit = 100): Promise<TeamStrength[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (league && league !== 'All') params.set('league', league);
  return get<TeamStrength[]>(`/teams?${params}`);
}

/** The players a team's rating was built from, highest minutes first. */
export function fetchSquad(team: string): Promise<PlayerRating[]> {
  return get<PlayerRating[]>(`/teams/${encodeURIComponent(team)}/squad`);
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
  by_league_accuracy: Record<string, number>;
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
