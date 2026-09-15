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
  modal_score_home: number;
  modal_score_away: number;
  prob_over_2_5: number;
  prob_btts: number;

  market_prob_home: number | null;
  market_prob_draw: number | null;
  market_prob_away: number | null;

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
