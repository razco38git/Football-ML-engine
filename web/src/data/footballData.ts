export type FormResult = 'W' | 'D' | 'L';

export interface Player {
  id: number;
  name: string;
  firstName: string;
  lastName: string;
  position: string;
  altPosition: string;
  team: string;
  league: string;
  nationality: string;
  flag: string;
  age: number;
  height: number;
  overall: number;
  potential: number;
  preferredFoot: 'R' | 'L';
  skillMoves: number;
  weakFoot: number;
  // Main 6 attributes
  pac: number;
  sho: number;
  pas: number;
  dri: number;
  def: number;
  phy: number;
  // Attacking sub
  crossing: number;
  finishing: number;
  headingAccuracy: number;
  shortPassing: number;
  volleys: number;
  // Skill sub
  dribbling: number;
  curve: number;
  fkAccuracy: number;
  longPassing: number;
  ballControl: number;
  // Movement sub
  acceleration: number;
  sprintSpeed: number;
  agility: number;
  reactions: number;
  balance: number;
  // Power sub
  shotPower: number;
  jumping: number;
  stamina: number;
  strength: number;
  longShots: number;
  // Mentality sub
  aggression: number;
  interceptions: number;
  positioning: number;
  vision: number;
  penalties: number;
  composure: number;
  // Defending sub
  defensiveAwareness: number;
  standingTackle: number;
  slidingTackle: number;
  // GK
  gkDiving: number;
  gkHandling: number;
  gkKicking: number;
  gkPositioning: number;
  gkReflexes: number;
  // FM / Scouting percentile
  fmRating: number; // out of 100 (FM out of 20, converted)
  scoutPercentile: number; // 0-100
  // Photo placeholder
  photoColor: string;
}

export interface TeamMatchData {
  name: string;
  shortName: string;
  logo: string;
  league: string;
  strengthRating: number;
  last5: FormResult[];
  last5xGFor: number[];
  last5xGAgainst: number[];
  last5Shots: number[];
  last5ShotsOnTarget: number[];
  color: string;
}

export interface UpcomingMatch {
  id: number;
  date: string;
  kickoff: string;
  competition: string;
  home: TeamMatchData;
  away: TeamMatchData;
}

export interface PastPrediction {
  id: number;
  date: string;
  competition: string;
  homeTeam: string;
  awayTeam: string;
  homeLogo: string;
  awayLogo: string;
  predictedHomeGoals: number;
  predictedAwayGoals: number;
  predictedHomeXG: number;
  predictedAwayXG: number;
  actualHomeGoals: number;
  actualAwayGoals: number;
  actualHomeXG: number;
  actualAwayXG: number;
  correct: boolean; // outcome correct
  confidence: number; // 0-100
}

// ─── PLAYERS ─────────────────────────────────────────────────────────────────

export const players: Player[] = [
  {
    id: 1, name: 'Kylian Mbappé', firstName: 'Kylian', lastName: 'Mbappé',
    position: 'ST', altPosition: 'LW', team: 'Real Madrid', league: 'LaLiga EA SPORTS',
    nationality: 'France', flag: '🇫🇷', age: 27, height: 178,
    overall: 91, potential: 94, preferredFoot: 'R', skillMoves: 5, weakFoot: 4,
    pac: 96, sho: 91, pas: 80, dri: 92, def: 29, phy: 76,
    crossing: 74, finishing: 95, headingAccuracy: 78, shortPassing: 87, volleys: 87,
    dribbling: 92, curve: 80, fkAccuracy: 70, longPassing: 74, ballControl: 93,
    acceleration: 97, sprintSpeed: 96, agility: 93, reactions: 92, balance: 81,
    shotPower: 91, jumping: 90, stamina: 83, strength: 77, longShots: 86,
    aggression: 61, interceptions: 19, positioning: 91, vision: 81, penalties: 85, composure: 88,
    defensiveAwareness: 15, standingTackle: 34, slidingTackle: 24,
    gkDiving: 13, gkHandling: 5, gkKicking: 7, gkPositioning: 11, gkReflexes: 8,
    fmRating: 91, scoutPercentile: 99, photoColor: '#1a3a6b',
  },
  {
    id: 2, name: 'Erling Haaland', firstName: 'Erling', lastName: 'Haaland',
    position: 'ST', altPosition: 'CF', team: 'Manchester City', league: 'Premier League',
    nationality: 'Norway', flag: '🇳🇴', age: 25, height: 195,
    overall: 91, potential: 95, preferredFoot: 'L', skillMoves: 3, weakFoot: 3,
    pac: 87, sho: 93, pas: 71, dri: 80, def: 47, phy: 89,
    crossing: 45, finishing: 98, headingAccuracy: 90, shortPassing: 72, volleys: 82,
    dribbling: 78, curve: 58, fkAccuracy: 55, longPassing: 62, ballControl: 80,
    acceleration: 88, sprintSpeed: 87, agility: 72, reactions: 90, balance: 65,
    shotPower: 95, jumping: 93, stamina: 84, strength: 90, longShots: 78,
    aggression: 79, interceptions: 25, positioning: 97, vision: 68, penalties: 89, composure: 87,
    defensiveAwareness: 25, standingTackle: 42, slidingTackle: 28,
    gkDiving: 8, gkHandling: 6, gkKicking: 5, gkPositioning: 7, gkReflexes: 9,
    fmRating: 90, scoutPercentile: 98, photoColor: '#1a3c5e',
  },
  {
    id: 3, name: 'Vinicius Jr.', firstName: 'Vinicius', lastName: 'Jr.',
    position: 'LW', altPosition: 'ST', team: 'Real Madrid', league: 'LaLiga EA SPORTS',
    nationality: 'Brazil', flag: '🇧🇷', age: 25, height: 176,
    overall: 90, potential: 93, preferredFoot: 'R', skillMoves: 5, weakFoot: 4,
    pac: 95, sho: 82, pas: 77, dri: 94, def: 24, phy: 65,
    crossing: 73, finishing: 82, headingAccuracy: 60, shortPassing: 79, volleys: 75,
    dribbling: 95, curve: 82, fkAccuracy: 72, longPassing: 71, ballControl: 93,
    acceleration: 96, sprintSpeed: 94, agility: 97, reactions: 88, balance: 94,
    shotPower: 80, jumping: 68, stamina: 80, strength: 66, longShots: 72,
    aggression: 63, interceptions: 18, positioning: 81, vision: 78, penalties: 74, composure: 79,
    defensiveAwareness: 18, standingTackle: 24, slidingTackle: 20,
    gkDiving: 6, gkHandling: 5, gkKicking: 4, gkPositioning: 5, gkReflexes: 7,
    fmRating: 89, scoutPercentile: 97, photoColor: '#3a1a1a',
  },
  {
    id: 4, name: 'Rodri', firstName: 'Rodrigo', lastName: 'Hernández',
    position: 'CDM', altPosition: 'CM', team: 'Manchester City', league: 'Premier League',
    nationality: 'Spain', flag: '🇪🇸', age: 29, height: 191,
    overall: 90, potential: 91, preferredFoot: 'R', skillMoves: 3, weakFoot: 3,
    pac: 62, sho: 72, pas: 89, dri: 82, def: 85, phy: 84,
    crossing: 63, finishing: 66, headingAccuracy: 75, shortPassing: 91, volleys: 64,
    dribbling: 80, curve: 73, fkAccuracy: 70, longPassing: 89, ballControl: 84,
    acceleration: 60, sprintSpeed: 63, agility: 74, reactions: 87, balance: 72,
    shotPower: 79, jumping: 78, stamina: 90, strength: 82, longShots: 75,
    aggression: 80, interceptions: 89, positioning: 72, vision: 88, penalties: 69, composure: 90,
    defensiveAwareness: 88, standingTackle: 87, slidingTackle: 83,
    gkDiving: 7, gkHandling: 6, gkKicking: 8, gkPositioning: 7, gkReflexes: 6,
    fmRating: 90, scoutPercentile: 97, photoColor: '#1a3a2a',
  },
  {
    id: 5, name: 'Jude Bellingham', firstName: 'Jude', lastName: 'Bellingham',
    position: 'CAM', altPosition: 'CM', team: 'Real Madrid', league: 'LaLiga EA SPORTS',
    nationality: 'England', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', age: 22, height: 186,
    overall: 89, potential: 96, preferredFoot: 'R', skillMoves: 4, weakFoot: 4,
    pac: 76, sho: 83, pas: 85, dri: 88, def: 68, phy: 80,
    crossing: 72, finishing: 81, headingAccuracy: 74, shortPassing: 87, volleys: 79,
    dribbling: 87, curve: 78, fkAccuracy: 71, longPassing: 82, ballControl: 88,
    acceleration: 75, sprintSpeed: 77, agility: 82, reactions: 88, balance: 79,
    shotPower: 84, jumping: 82, stamina: 85, strength: 80, longShots: 80,
    aggression: 78, interceptions: 70, positioning: 84, vision: 86, penalties: 75, composure: 87,
    defensiveAwareness: 67, standingTackle: 69, slidingTackle: 64,
    gkDiving: 9, gkHandling: 7, gkKicking: 8, gkPositioning: 8, gkReflexes: 9,
    fmRating: 88, scoutPercentile: 96, photoColor: '#1a2a4a',
  },
  {
    id: 6, name: 'Mohamed Salah', firstName: 'Mohamed', lastName: 'Salah',
    position: 'RW', altPosition: 'ST', team: 'Liverpool', league: 'Premier League',
    nationality: 'Egypt', flag: '🇪🇬', age: 33, height: 175,
    overall: 88, potential: 88, preferredFoot: 'L', skillMoves: 4, weakFoot: 3,
    pac: 90, sho: 88, pas: 79, dri: 88, def: 35, phy: 70,
    crossing: 78, finishing: 91, headingAccuracy: 65, shortPassing: 81, volleys: 79,
    dribbling: 89, curve: 84, fkAccuracy: 73, longPassing: 73, ballControl: 87,
    acceleration: 91, sprintSpeed: 90, agility: 88, reactions: 88, balance: 83,
    shotPower: 84, jumping: 68, stamina: 81, strength: 72, longShots: 82,
    aggression: 62, interceptions: 34, positioning: 88, vision: 80, penalties: 78, composure: 86,
    defensiveAwareness: 33, standingTackle: 36, slidingTackle: 31,
    gkDiving: 7, gkHandling: 6, gkKicking: 5, gkPositioning: 6, gkReflexes: 8,
    fmRating: 87, scoutPercentile: 94, photoColor: '#2a1a1a',
  },
  {
    id: 7, name: 'Bukayo Saka', firstName: 'Bukayo', lastName: 'Saka',
    position: 'RW', altPosition: 'LW', team: 'Arsenal', league: 'Premier League',
    nationality: 'England', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', age: 23, height: 178,
    overall: 87, potential: 91, preferredFoot: 'L', skillMoves: 4, weakFoot: 4,
    pac: 87, sho: 83, pas: 84, dri: 87, def: 52, phy: 67,
    crossing: 82, finishing: 82, headingAccuracy: 65, shortPassing: 86, volleys: 74,
    dribbling: 87, curve: 81, fkAccuracy: 79, longPassing: 78, ballControl: 87,
    acceleration: 88, sprintSpeed: 86, agility: 87, reactions: 85, balance: 88,
    shotPower: 82, jumping: 64, stamina: 86, strength: 66, longShots: 80,
    aggression: 61, interceptions: 51, positioning: 83, vision: 82, penalties: 86, composure: 85,
    defensiveAwareness: 50, standingTackle: 52, slidingTackle: 47,
    gkDiving: 6, gkHandling: 5, gkKicking: 6, gkPositioning: 5, gkReflexes: 7,
    fmRating: 86, scoutPercentile: 93, photoColor: '#1a3a1a',
  },
  {
    id: 8, name: 'Harry Kane', firstName: 'Harry', lastName: 'Kane',
    position: 'ST', altPosition: 'CF', team: 'Bayern München', league: 'Bundesliga',
    nationality: 'England', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', age: 32, height: 188,
    overall: 90, potential: 90, preferredFoot: 'R', skillMoves: 3, weakFoot: 4,
    pac: 62, sho: 94, pas: 83, dri: 82, def: 49, phy: 83,
    crossing: 65, finishing: 95, headingAccuracy: 88, shortPassing: 84, volleys: 82,
    dribbling: 80, curve: 76, fkAccuracy: 68, longPassing: 81, ballControl: 82,
    acceleration: 60, sprintSpeed: 63, agility: 72, reactions: 93, balance: 67,
    shotPower: 91, jumping: 81, stamina: 82, strength: 84, longShots: 85,
    aggression: 71, interceptions: 39, positioning: 96, vision: 85, penalties: 90, composure: 91,
    defensiveAwareness: 43, standingTackle: 48, slidingTackle: 39,
    gkDiving: 8, gkHandling: 6, gkKicking: 7, gkPositioning: 7, gkReflexes: 8,
    fmRating: 89, scoutPercentile: 96, photoColor: '#2a1a3a',
  },
  {
    id: 9, name: 'Lamine Yamal', firstName: 'Lamine', lastName: 'Yamal',
    position: 'RW', altPosition: 'LW', team: 'FC Barcelona', league: 'LaLiga EA SPORTS',
    nationality: 'Spain', flag: '🇪🇸', age: 18, height: 180,
    overall: 87, potential: 96, preferredFoot: 'R', skillMoves: 5, weakFoot: 4,
    pac: 91, sho: 82, pas: 83, dri: 90, def: 30, phy: 59,
    crossing: 81, finishing: 81, headingAccuracy: 58, shortPassing: 85, volleys: 72,
    dribbling: 91, curve: 86, fkAccuracy: 80, longPassing: 77, ballControl: 89,
    acceleration: 92, sprintSpeed: 90, agility: 93, reactions: 87, balance: 92,
    shotPower: 78, jumping: 60, stamina: 77, strength: 58, longShots: 79,
    aggression: 55, interceptions: 24, positioning: 79, vision: 84, penalties: 74, composure: 83,
    defensiveAwareness: 26, standingTackle: 30, slidingTackle: 24,
    gkDiving: 5, gkHandling: 4, gkKicking: 4, gkPositioning: 5, gkReflexes: 6,
    fmRating: 85, scoutPercentile: 96, photoColor: '#3a2a1a',
  },
  {
    id: 10, name: 'Phil Foden', firstName: 'Phil', lastName: 'Foden',
    position: 'CAM', altPosition: 'LW', team: 'Manchester City', league: 'Premier League',
    nationality: 'England', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', age: 25, height: 171,
    overall: 88, potential: 91, preferredFoot: 'L', skillMoves: 4, weakFoot: 3,
    pac: 82, sho: 82, pas: 85, dri: 88, def: 45, phy: 63,
    crossing: 74, finishing: 83, headingAccuracy: 62, shortPassing: 87, volleys: 78,
    dribbling: 88, curve: 83, fkAccuracy: 78, longPassing: 81, ballControl: 89,
    acceleration: 83, sprintSpeed: 81, agility: 90, reactions: 87, balance: 91,
    shotPower: 79, jumping: 65, stamina: 78, strength: 60, longShots: 78,
    aggression: 63, interceptions: 43, positioning: 83, vision: 87, penalties: 74, composure: 86,
    defensiveAwareness: 43, standingTackle: 46, slidingTackle: 40,
    gkDiving: 7, gkHandling: 5, gkKicking: 6, gkPositioning: 6, gkReflexes: 7,
    fmRating: 87, scoutPercentile: 93, photoColor: '#1a3c5e',
  },
  {
    id: 11, name: 'Virgil van Dijk', firstName: 'Virgil', lastName: 'van Dijk',
    position: 'CB', altPosition: 'CB', team: 'Liverpool', league: 'Premier League',
    nationality: 'Netherlands', flag: '🇳🇱', age: 34, height: 193,
    overall: 87, potential: 87, preferredFoot: 'R', skillMoves: 2, weakFoot: 3,
    pac: 74, sho: 58, pas: 72, dri: 70, def: 90, phy: 88,
    crossing: 52, finishing: 51, headingAccuracy: 87, shortPassing: 74, volleys: 44,
    dribbling: 69, curve: 54, fkAccuracy: 48, longPassing: 76, ballControl: 71,
    acceleration: 72, sprintSpeed: 76, agility: 62, reactions: 84, balance: 63,
    shotPower: 64, jumping: 91, stamina: 84, strength: 90, longShots: 55,
    aggression: 72, interceptions: 87, positioning: 60, vision: 72, penalties: 52, composure: 83,
    defensiveAwareness: 91, standingTackle: 91, slidingTackle: 87,
    gkDiving: 8, gkHandling: 6, gkKicking: 9, gkPositioning: 7, gkReflexes: 7,
    fmRating: 87, scoutPercentile: 93, photoColor: '#1a2a1a',
  },
  {
    id: 12, name: 'Thibaut Courtois', firstName: 'Thibaut', lastName: 'Courtois',
    position: 'GK', altPosition: 'GK', team: 'Real Madrid', league: 'LaLiga EA SPORTS',
    nationality: 'Belgium', flag: '🇧🇪', age: 33, height: 199,
    overall: 90, potential: 90, preferredFoot: 'L', skillMoves: 1, weakFoot: 3,
    pac: 87, sho: 24, pas: 78, dri: 36, def: 90, phy: 84,
    crossing: 15, finishing: 12, headingAccuracy: 22, shortPassing: 78, volleys: 10,
    dribbling: 32, curve: 14, fkAccuracy: 14, longPassing: 68, ballControl: 38,
    acceleration: 58, sprintSpeed: 53, agility: 72, reactions: 92, balance: 55,
    shotPower: 52, jumping: 78, stamina: 48, strength: 80, longShots: 16,
    aggression: 39, interceptions: 22, positioning: 18, vision: 62, penalties: 34, composure: 74,
    defensiveAwareness: 30, standingTackle: 25, slidingTackle: 18,
    gkDiving: 90, gkHandling: 89, gkKicking: 72, gkPositioning: 90, gkReflexes: 92,
    fmRating: 90, scoutPercentile: 97, photoColor: '#1a1a3a',
  },
  {
    id: 13, name: 'Kevin De Bruyne', firstName: 'Kevin', lastName: 'De Bruyne',
    position: 'CM', altPosition: 'CAM', team: 'Manchester City', league: 'Premier League',
    nationality: 'Belgium', flag: '🇧🇪', age: 34, height: 181,
    overall: 87, potential: 87, preferredFoot: 'R', skillMoves: 4, weakFoot: 5,
    pac: 73, sho: 82, pas: 93, dri: 87, def: 60, phy: 76,
    crossing: 93, finishing: 78, headingAccuracy: 60, shortPassing: 92, volleys: 76,
    dribbling: 86, curve: 82, fkAccuracy: 82, longPassing: 93, ballControl: 87,
    acceleration: 72, sprintSpeed: 74, agility: 77, reactions: 90, balance: 72,
    shotPower: 87, jumping: 66, stamina: 82, strength: 74, longShots: 86,
    aggression: 67, interceptions: 58, positioning: 82, vision: 95, penalties: 72, composure: 88,
    defensiveAwareness: 60, standingTackle: 61, slidingTackle: 52,
    gkDiving: 8, gkHandling: 6, gkKicking: 7, gkPositioning: 7, gkReflexes: 8,
    fmRating: 87, scoutPercentile: 93, photoColor: '#1a3c5e',
  },
  {
    id: 14, name: 'Martin Ødegaard', firstName: 'Martin', lastName: 'Ødegaard',
    position: 'CAM', altPosition: 'CM', team: 'Arsenal', league: 'Premier League',
    nationality: 'Norway', flag: '🇳🇴', age: 27, height: 178,
    overall: 86, potential: 88, preferredFoot: 'R', skillMoves: 4, weakFoot: 3,
    pac: 72, sho: 80, pas: 89, dri: 86, def: 52, phy: 63,
    crossing: 78, finishing: 79, headingAccuracy: 60, shortPassing: 91, volleys: 74,
    dribbling: 86, curve: 84, fkAccuracy: 79, longPassing: 84, ballControl: 87,
    acceleration: 72, sprintSpeed: 72, agility: 84, reactions: 86, balance: 84,
    shotPower: 79, jumping: 63, stamina: 79, strength: 60, longShots: 80,
    aggression: 62, interceptions: 50, positioning: 81, vision: 90, penalties: 72, composure: 86,
    defensiveAwareness: 51, standingTackle: 53, slidingTackle: 47,
    gkDiving: 6, gkHandling: 5, gkKicking: 6, gkPositioning: 5, gkReflexes: 6,
    fmRating: 85, scoutPercentile: 91, photoColor: '#1a3a1a',
  },
  {
    id: 15, name: 'Pedri', firstName: 'Pedri', lastName: 'González',
    position: 'CM', altPosition: 'CAM', team: 'FC Barcelona', league: 'LaLiga EA SPORTS',
    nationality: 'Spain', flag: '🇪🇸', age: 23, height: 174,
    overall: 86, potential: 93, preferredFoot: 'L', skillMoves: 4, weakFoot: 3,
    pac: 69, sho: 76, pas: 88, dri: 88, def: 60, phy: 62,
    crossing: 72, finishing: 74, headingAccuracy: 60, shortPassing: 91, volleys: 70,
    dribbling: 89, curve: 80, fkAccuracy: 74, longPassing: 83, ballControl: 90,
    acceleration: 68, sprintSpeed: 70, agility: 90, reactions: 88, balance: 92,
    shotPower: 76, jumping: 65, stamina: 79, strength: 60, longShots: 73,
    aggression: 68, interceptions: 58, positioning: 76, vision: 89, penalties: 69, composure: 86,
    defensiveAwareness: 59, standingTackle: 61, slidingTackle: 57,
    gkDiving: 6, gkHandling: 5, gkKicking: 6, gkPositioning: 5, gkReflexes: 6,
    fmRating: 85, scoutPercentile: 90, photoColor: '#3a2a1a',
  },
  {
    id: 16, name: 'Declan Rice', firstName: 'Declan', lastName: 'Rice',
    position: 'CDM', altPosition: 'CM', team: 'Arsenal', league: 'Premier League',
    nationality: 'England', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', age: 26, height: 185,
    overall: 85, potential: 87, preferredFoot: 'R', skillMoves: 3, weakFoot: 3,
    pac: 71, sho: 69, pas: 81, dri: 78, def: 83, phy: 82,
    crossing: 59, finishing: 63, headingAccuracy: 72, shortPassing: 83, volleys: 60,
    dribbling: 77, curve: 65, fkAccuracy: 66, longPassing: 81, ballControl: 79,
    acceleration: 70, sprintSpeed: 73, agility: 74, reactions: 83, balance: 71,
    shotPower: 79, jumping: 79, stamina: 91, strength: 82, longShots: 72,
    aggression: 83, interceptions: 85, positioning: 69, vision: 80, penalties: 66, composure: 83,
    defensiveAwareness: 85, standingTackle: 84, slidingTackle: 80,
    gkDiving: 7, gkHandling: 6, gkKicking: 7, gkPositioning: 6, gkReflexes: 7,
    fmRating: 84, scoutPercentile: 88, photoColor: '#1a3a1a',
  },
  {
    id: 17, name: 'Federico Valverde', firstName: 'Federico', lastName: 'Valverde',
    position: 'CM', altPosition: 'CDM', team: 'Real Madrid', league: 'LaLiga EA SPORTS',
    nationality: 'Uruguay', flag: '🇺🇾', age: 27, height: 182,
    overall: 87, potential: 89, preferredFoot: 'R', skillMoves: 3, weakFoot: 3,
    pac: 80, sho: 77, pas: 81, dri: 82, def: 75, phy: 83,
    crossing: 68, finishing: 73, headingAccuracy: 72, shortPassing: 83, volleys: 72,
    dribbling: 82, curve: 70, fkAccuracy: 67, longPassing: 79, ballControl: 83,
    acceleration: 80, sprintSpeed: 81, agility: 80, reactions: 85, balance: 77,
    shotPower: 82, jumping: 80, stamina: 93, strength: 82, longShots: 77,
    aggression: 80, interceptions: 76, positioning: 78, vision: 79, penalties: 70, composure: 84,
    defensiveAwareness: 76, standingTackle: 77, slidingTackle: 73,
    gkDiving: 7, gkHandling: 6, gkKicking: 7, gkPositioning: 6, gkReflexes: 7,
    fmRating: 86, scoutPercentile: 89, photoColor: '#1a3a6b',
  },
  {
    id: 18, name: 'Ruben Dias', firstName: 'Ruben', lastName: 'Dias',
    position: 'CB', altPosition: 'CB', team: 'Manchester City', league: 'Premier League',
    nationality: 'Portugal', flag: '🇵🇹', age: 28, height: 187,
    overall: 87, potential: 89, preferredFoot: 'R', skillMoves: 2, weakFoot: 3,
    pac: 73, sho: 47, pas: 71, dri: 68, def: 90, phy: 86,
    crossing: 48, finishing: 40, headingAccuracy: 84, shortPassing: 73, volleys: 38,
    dribbling: 66, curve: 48, fkAccuracy: 44, longPassing: 74, ballControl: 70,
    acceleration: 71, sprintSpeed: 75, agility: 63, reactions: 84, balance: 62,
    shotPower: 60, jumping: 89, stamina: 86, strength: 89, longShots: 48,
    aggression: 76, interceptions: 89, positioning: 57, vision: 68, penalties: 45, composure: 82,
    defensiveAwareness: 92, standingTackle: 90, slidingTackle: 87,
    gkDiving: 7, gkHandling: 6, gkKicking: 8, gkPositioning: 7, gkReflexes: 7,
    fmRating: 86, scoutPercentile: 90, photoColor: '#1a3c5e',
  },
  {
    id: 19, name: 'Bernardo Silva', firstName: 'Bernardo', lastName: 'Silva',
    position: 'CM', altPosition: 'RW', team: 'Manchester City', league: 'Premier League',
    nationality: 'Portugal', flag: '🇵🇹', age: 31, height: 173,
    overall: 86, potential: 87, preferredFoot: 'R', skillMoves: 4, weakFoot: 4,
    pac: 76, sho: 79, pas: 87, dri: 88, def: 61, phy: 66,
    crossing: 78, finishing: 77, headingAccuracy: 62, shortPassing: 89, volleys: 72,
    dribbling: 89, curve: 79, fkAccuracy: 73, longPassing: 83, ballControl: 89,
    acceleration: 77, sprintSpeed: 76, agility: 90, reactions: 87, balance: 88,
    shotPower: 78, jumping: 65, stamina: 85, strength: 64, longShots: 78,
    aggression: 65, interceptions: 59, positioning: 79, vision: 86, penalties: 71, composure: 87,
    defensiveAwareness: 60, standingTackle: 62, slidingTackle: 57,
    gkDiving: 7, gkHandling: 5, gkKicking: 6, gkPositioning: 6, gkReflexes: 7,
    fmRating: 85, scoutPercentile: 89, photoColor: '#1a3c5e',
  },
  {
    id: 20, name: 'Jamal Musiala', firstName: 'Jamal', lastName: 'Musiala',
    position: 'CAM', altPosition: 'LW', team: 'Bayern München', league: 'Bundesliga',
    nationality: 'Germany', flag: '🇩🇪', age: 22, height: 183,
    overall: 86, potential: 95, preferredFoot: 'R', skillMoves: 4, weakFoot: 3,
    pac: 82, sho: 80, pas: 83, dri: 90, def: 43, phy: 68,
    crossing: 69, finishing: 80, headingAccuracy: 68, shortPassing: 85, volleys: 77,
    dribbling: 91, curve: 79, fkAccuracy: 71, longPassing: 79, ballControl: 91,
    acceleration: 84, sprintSpeed: 82, agility: 93, reactions: 88, balance: 91,
    shotPower: 78, jumping: 73, stamina: 81, strength: 65, longShots: 79,
    aggression: 65, interceptions: 41, positioning: 81, vision: 86, penalties: 73, composure: 85,
    defensiveAwareness: 41, standingTackle: 44, slidingTackle: 39,
    gkDiving: 6, gkHandling: 5, gkKicking: 5, gkPositioning: 5, gkReflexes: 7,
    fmRating: 85, scoutPercentile: 92, photoColor: '#2a1a1a',
  },
  {
    id: 21, name: 'Trent Alexander-Arnold', firstName: 'Trent', lastName: 'Alexander-Arnold',
    position: 'RB', altPosition: 'CM', team: 'Real Madrid', league: 'LaLiga EA SPORTS',
    nationality: 'England', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', age: 27, height: 175,
    overall: 86, potential: 88, preferredFoot: 'R', skillMoves: 3, weakFoot: 3,
    pac: 78, sho: 72, pas: 91, dri: 80, def: 72, phy: 65,
    crossing: 95, finishing: 66, headingAccuracy: 62, shortPassing: 89, volleys: 62,
    dribbling: 80, curve: 85, fkAccuracy: 83, longPassing: 90, ballControl: 80,
    acceleration: 80, sprintSpeed: 78, agility: 77, reactions: 84, balance: 71,
    shotPower: 77, jumping: 69, stamina: 80, strength: 63, longShots: 77,
    aggression: 58, interceptions: 70, positioning: 66, vision: 89, penalties: 72, composure: 80,
    defensiveAwareness: 73, standingTackle: 73, slidingTackle: 69,
    gkDiving: 7, gkHandling: 5, gkKicking: 7, gkPositioning: 6, gkReflexes: 7,
    fmRating: 85, scoutPercentile: 88, photoColor: '#1a3a6b',
  },
  {
    id: 22, name: 'Robert Lewandowski', firstName: 'Robert', lastName: 'Lewandowski',
    position: 'ST', altPosition: 'CF', team: 'FC Barcelona', league: 'LaLiga EA SPORTS',
    nationality: 'Poland', flag: '🇵🇱', age: 37, height: 185,
    overall: 86, potential: 86, preferredFoot: 'R', skillMoves: 4, weakFoot: 4,
    pac: 68, sho: 93, pas: 78, dri: 81, def: 42, phy: 80,
    crossing: 55, finishing: 95, headingAccuracy: 85, shortPassing: 80, volleys: 83,
    dribbling: 80, curve: 71, fkAccuracy: 64, longPassing: 73, ballControl: 83,
    acceleration: 65, sprintSpeed: 71, agility: 76, reactions: 92, balance: 73,
    shotPower: 90, jumping: 84, stamina: 78, strength: 81, longShots: 82,
    aggression: 69, interceptions: 31, positioning: 96, vision: 76, penalties: 92, composure: 90,
    defensiveAwareness: 37, standingTackle: 42, slidingTackle: 34,
    gkDiving: 8, gkHandling: 5, gkKicking: 7, gkPositioning: 7, gkReflexes: 8,
    fmRating: 85, scoutPercentile: 92, photoColor: '#3a2a1a',
  },
  {
    id: 23, name: 'Victor Osimhen', firstName: 'Victor', lastName: 'Osimhen',
    position: 'ST', altPosition: 'CF', team: 'Galatasaray', league: 'Süper Lig',
    nationality: 'Nigeria', flag: '🇳🇬', age: 26, height: 185,
    overall: 85, potential: 89, preferredFoot: 'R', skillMoves: 3, weakFoot: 4,
    pac: 91, sho: 88, pas: 65, dri: 80, def: 35, phy: 84,
    crossing: 48, finishing: 90, headingAccuracy: 84, shortPassing: 67, volleys: 76,
    dribbling: 78, curve: 63, fkAccuracy: 55, longPassing: 58, ballControl: 80,
    acceleration: 93, sprintSpeed: 91, agility: 79, reactions: 86, balance: 72,
    shotPower: 87, jumping: 84, stamina: 83, strength: 84, longShots: 76,
    aggression: 72, interceptions: 25, positioning: 89, vision: 63, penalties: 74, composure: 79,
    defensiveAwareness: 28, standingTackle: 35, slidingTackle: 28,
    gkDiving: 7, gkHandling: 5, gkKicking: 6, gkPositioning: 6, gkReflexes: 7,
    fmRating: 83, scoutPercentile: 88, photoColor: '#2a2a1a',
  },
  {
    id: 24, name: 'Antoine Griezmann', firstName: 'Antoine', lastName: 'Griezmann',
    position: 'CAM', altPosition: 'ST', team: 'Atlético Madrid', league: 'LaLiga EA SPORTS',
    nationality: 'France', flag: '🇫🇷', age: 34, height: 176,
    overall: 85, potential: 85, preferredFoot: 'L', skillMoves: 4, weakFoot: 3,
    pac: 76, sho: 87, pas: 82, dri: 84, def: 50, phy: 72,
    crossing: 72, finishing: 88, headingAccuracy: 75, shortPassing: 84, volleys: 81,
    dribbling: 83, curve: 80, fkAccuracy: 75, longPassing: 77, ballControl: 85,
    acceleration: 78, sprintSpeed: 75, agility: 84, reactions: 87, balance: 82,
    shotPower: 84, jumping: 74, stamina: 82, strength: 72, longShots: 82,
    aggression: 65, interceptions: 47, positioning: 86, vision: 83, penalties: 83, composure: 87,
    defensiveAwareness: 47, standingTackle: 50, slidingTackle: 46,
    gkDiving: 7, gkHandling: 5, gkKicking: 6, gkPositioning: 6, gkReflexes: 7,
    fmRating: 83, scoutPercentile: 87, photoColor: '#2a1a1a',
  },
  {
    id: 25, name: 'Alisson Becker', firstName: 'Alisson', lastName: 'Becker',
    position: 'GK', altPosition: 'GK', team: 'Liverpool', league: 'Premier League',
    nationality: 'Brazil', flag: '🇧🇷', age: 33, height: 191,
    overall: 88, potential: 88, preferredFoot: 'R', skillMoves: 1, weakFoot: 3,
    pac: 81, sho: 18, pas: 80, dri: 36, def: 88, phy: 77,
    crossing: 14, finishing: 10, headingAccuracy: 19, shortPassing: 80, volleys: 8,
    dribbling: 32, curve: 14, fkAccuracy: 12, longPassing: 72, ballControl: 36,
    acceleration: 52, sprintSpeed: 48, agility: 70, reactions: 89, balance: 52,
    shotPower: 48, jumping: 75, stamina: 50, strength: 76, longShots: 14,
    aggression: 36, interceptions: 19, positioning: 15, vision: 60, penalties: 29, composure: 71,
    defensiveAwareness: 26, standingTackle: 23, slidingTackle: 17,
    gkDiving: 88, gkHandling: 86, gkKicking: 76, gkPositioning: 88, gkReflexes: 91,
    fmRating: 87, scoutPercentile: 94, photoColor: '#2a1a3a',
  },
];

// ─── TEAM MATCH DATA ──────────────────────────────────────────────────────────

export const upcomingMatches: UpcomingMatch[] = [
  {
    id: 1,
    date: '2026-09-20',
    kickoff: '20:45',
    competition: 'LaLiga EA SPORTS',
    home: {
      name: 'Real Madrid', shortName: 'RMA', logo: '⚽', league: 'LaLiga',
      strengthRating: 88.4, color: '#f5c518',
      last5: ['W', 'W', 'W', 'D', 'W'],
      last5xGFor: [2.4, 3.1, 1.8, 1.2, 2.6],
      last5xGAgainst: [0.6, 0.9, 1.4, 1.1, 0.7],
      last5Shots: [17, 22, 14, 12, 19],
      last5ShotsOnTarget: [8, 10, 6, 5, 9],
    },
    away: {
      name: 'FC Barcelona', shortName: 'BAR', logo: '⚽', league: 'LaLiga',
      strengthRating: 86.2, color: '#a50044',
      last5: ['W', 'D', 'W', 'W', 'L'],
      last5xGFor: [2.1, 1.3, 2.4, 1.9, 0.8],
      last5xGAgainst: [0.9, 0.7, 1.1, 0.8, 1.6],
      last5Shots: [16, 12, 18, 15, 10],
      last5ShotsOnTarget: [7, 5, 8, 7, 4],
    },
  },
  {
    id: 2,
    date: '2026-09-21',
    kickoff: '16:30',
    competition: 'Premier League',
    home: {
      name: 'Manchester City', shortName: 'MCI', logo: '⚽', league: 'PL',
      strengthRating: 87.9, color: '#6cabdd',
      last5: ['W', 'W', 'D', 'W', 'W'],
      last5xGFor: [2.8, 2.2, 1.4, 3.0, 2.1],
      last5xGAgainst: [0.5, 0.8, 1.2, 0.6, 0.9],
      last5Shots: [19, 16, 13, 23, 17],
      last5ShotsOnTarget: [9, 7, 5, 11, 8],
    },
    away: {
      name: 'Liverpool', shortName: 'LIV', logo: '⚽', league: 'PL',
      strengthRating: 85.8, color: '#c8102e',
      last5: ['W', 'L', 'W', 'W', 'D'],
      last5xGFor: [2.2, 0.9, 1.9, 2.4, 1.5],
      last5xGAgainst: [0.8, 2.1, 0.7, 0.9, 1.0],
      last5Shots: [15, 10, 14, 18, 13],
      last5ShotsOnTarget: [7, 4, 6, 8, 6],
    },
  },
  {
    id: 3,
    date: '2026-09-21',
    kickoff: '14:00',
    competition: 'Premier League',
    home: {
      name: 'Arsenal', shortName: 'ARS', logo: '⚽', league: 'PL',
      strengthRating: 84.6, color: '#ef0107',
      last5: ['W', 'W', 'W', 'D', 'W'],
      last5xGFor: [2.0, 1.8, 2.4, 1.1, 2.2],
      last5xGAgainst: [0.7, 0.5, 0.8, 1.4, 0.6],
      last5Shots: [16, 14, 18, 11, 17],
      last5ShotsOnTarget: [7, 6, 8, 5, 8],
    },
    away: {
      name: 'Chelsea', shortName: 'CHE', logo: '⚽', league: 'PL',
      strengthRating: 82.1, color: '#034694',
      last5: ['D', 'W', 'L', 'W', 'D'],
      last5xGFor: [1.3, 1.9, 0.9, 2.1, 1.2],
      last5xGAgainst: [1.1, 0.8, 1.9, 0.7, 1.3],
      last5Shots: [13, 14, 9, 15, 12],
      last5ShotsOnTarget: [5, 6, 4, 7, 5],
    },
  },
  {
    id: 4,
    date: '2026-09-22',
    kickoff: '18:30',
    competition: 'Bundesliga',
    home: {
      name: 'Bayern München', shortName: 'BAY', logo: '⚽', league: 'BL',
      strengthRating: 86.8, color: '#dc052d',
      last5: ['W', 'W', 'W', 'W', 'D'],
      last5xGFor: [3.2, 2.6, 2.9, 3.4, 1.8],
      last5xGAgainst: [0.4, 0.7, 0.6, 0.5, 0.9],
      last5Shots: [21, 18, 20, 24, 15],
      last5ShotsOnTarget: [10, 8, 10, 12, 7],
    },
    away: {
      name: 'Bor. Dortmund', shortName: 'BVB', logo: '⚽', league: 'BL',
      strengthRating: 81.4, color: '#fde100',
      last5: ['W', 'D', 'L', 'W', 'W'],
      last5xGFor: [1.9, 1.2, 0.8, 2.1, 1.7],
      last5xGAgainst: [1.0, 1.3, 2.2, 0.9, 1.1],
      last5Shots: [14, 11, 8, 15, 13],
      last5ShotsOnTarget: [6, 4, 3, 7, 6],
    },
  },
  {
    id: 5,
    date: '2026-09-23',
    kickoff: '21:00',
    competition: 'UEFA Champions League',
    home: {
      name: 'Inter Milan', shortName: 'INT', logo: '⚽', league: 'UCL',
      strengthRating: 83.7, color: '#003082',
      last5: ['W', 'D', 'W', 'W', 'L'],
      last5xGFor: [1.7, 1.0, 2.1, 1.8, 0.7],
      last5xGAgainst: [0.8, 1.2, 0.9, 0.6, 1.8],
      last5Shots: [13, 10, 16, 14, 9],
      last5ShotsOnTarget: [5, 4, 7, 6, 3],
    },
    away: {
      name: 'AC Milan', shortName: 'MIL', logo: '⚽', league: 'UCL',
      strengthRating: 82.4, color: '#fb090b',
      last5: ['L', 'W', 'D', 'W', 'W'],
      last5xGFor: [0.9, 1.8, 1.3, 2.0, 1.6],
      last5xGAgainst: [1.6, 0.7, 0.9, 0.8, 0.7],
      last5Shots: [10, 14, 11, 15, 13],
      last5ShotsOnTarget: [4, 6, 5, 7, 6],
    },
  },
  {
    id: 6,
    date: '2026-09-24',
    kickoff: '20:45',
    competition: 'UEFA Champions League',
    home: {
      name: 'PSG', shortName: 'PSG', logo: '⚽', league: 'UCL',
      strengthRating: 85.2, color: '#004170',
      last5: ['W', 'W', 'D', 'W', 'W'],
      last5xGFor: [2.4, 2.0, 1.5, 2.8, 2.2],
      last5xGAgainst: [0.7, 0.5, 1.1, 0.4, 0.6],
      last5Shots: [18, 16, 13, 21, 17],
      last5ShotsOnTarget: [8, 7, 6, 10, 8],
    },
    away: {
      name: 'Atlético Madrid', shortName: 'ATM', logo: '⚽', league: 'UCL',
      strengthRating: 83.0, color: '#cb3524',
      last5: ['D', 'W', 'D', 'L', 'W'],
      last5xGFor: [1.1, 1.6, 1.0, 0.8, 1.4],
      last5xGAgainst: [0.9, 0.6, 0.8, 1.4, 0.7],
      last5Shots: [11, 13, 10, 9, 12],
      last5ShotsOnTarget: [4, 5, 4, 3, 5],
    },
  },
];

// ─── PAST PREDICTIONS ────────────────────────────────────────────────────────

export const pastPredictions: PastPrediction[] = [
  {
    id: 1, date: '2026-09-14', competition: 'Premier League',
    homeTeam: 'Manchester City', awayTeam: 'Arsenal',
    homeLogo: '🔵', awayLogo: '🔴',
    predictedHomeGoals: 2, predictedAwayGoals: 1,
    predictedHomeXG: 2.1, predictedAwayXG: 1.2,
    actualHomeGoals: 2, actualAwayGoals: 0,
    actualHomeXG: 2.4, actualAwayXG: 0.7,
    correct: true, confidence: 72,
  },
  {
    id: 2, date: '2026-09-13', competition: 'LaLiga EA SPORTS',
    homeTeam: 'Real Madrid', awayTeam: 'Valencia',
    homeLogo: '⚪', awayLogo: '🟠',
    predictedHomeGoals: 3, predictedAwayGoals: 1,
    predictedHomeXG: 2.8, predictedAwayXG: 0.9,
    actualHomeGoals: 2, actualAwayGoals: 1,
    actualHomeXG: 2.2, actualAwayXG: 0.8,
    correct: true, confidence: 81,
  },
  {
    id: 3, date: '2026-09-13', competition: 'Premier League',
    homeTeam: 'Liverpool', awayTeam: 'Wolves',
    homeLogo: '🔴', awayLogo: '🟡',
    predictedHomeGoals: 2, predictedAwayGoals: 0,
    predictedHomeXG: 2.3, predictedAwayXG: 0.6,
    actualHomeGoals: 3, actualAwayGoals: 0,
    actualHomeXG: 2.9, actualAwayXG: 0.5,
    correct: true, confidence: 77,
  },
  {
    id: 4, date: '2026-09-07', competition: 'LaLiga EA SPORTS',
    homeTeam: 'FC Barcelona', awayTeam: 'Atlético Madrid',
    homeLogo: '🔵', awayLogo: '🔴',
    predictedHomeGoals: 1, predictedAwayGoals: 1,
    predictedHomeXG: 1.6, predictedAwayXG: 1.2,
    actualHomeGoals: 0, actualAwayGoals: 1,
    actualHomeXG: 0.9, actualAwayXG: 1.4,
    correct: false, confidence: 48,
  },
  {
    id: 5, date: '2026-09-06', competition: 'Bundesliga',
    homeTeam: 'Bayern München', awayTeam: 'Augsburg',
    homeLogo: '🔴', awayLogo: '🟢',
    predictedHomeGoals: 4, predictedAwayGoals: 0,
    predictedHomeXG: 3.4, predictedAwayXG: 0.5,
    actualHomeGoals: 3, actualAwayGoals: 1,
    actualHomeXG: 2.9, actualAwayXG: 0.8,
    correct: true, confidence: 89,
  },
  {
    id: 6, date: '2026-09-06', competition: 'Premier League',
    homeTeam: 'Chelsea', awayTeam: 'Tottenham',
    homeLogo: '🔵', awayLogo: '⚪',
    predictedHomeGoals: 2, predictedAwayGoals: 1,
    predictedHomeXG: 1.8, predictedAwayXG: 1.4,
    actualHomeGoals: 1, actualAwayGoals: 3,
    actualHomeXG: 1.2, actualAwayXG: 2.1,
    correct: false, confidence: 54,
  },
  {
    id: 7, date: '2026-08-31', competition: 'LaLiga EA SPORTS',
    homeTeam: 'Sevilla', awayTeam: 'Real Madrid',
    homeLogo: '🔴', awayLogo: '⚪',
    predictedHomeGoals: 0, predictedAwayGoals: 2,
    predictedHomeXG: 0.7, predictedAwayXG: 2.4,
    actualHomeGoals: 0, actualAwayGoals: 3,
    actualHomeXG: 0.6, actualAwayXG: 3.1,
    correct: true, confidence: 76,
  },
  {
    id: 8, date: '2026-08-30', competition: 'Premier League',
    homeTeam: 'Manchester United', awayTeam: 'Brighton',
    homeLogo: '🔴', awayLogo: '🔵',
    predictedHomeGoals: 2, predictedAwayGoals: 1,
    predictedHomeXG: 1.9, predictedAwayXG: 1.3,
    actualHomeGoals: 1, actualAwayGoals: 1,
    actualHomeXG: 1.4, actualAwayXG: 1.8,
    correct: false, confidence: 59,
  },
];

// ─── HELPERS ─────────────────────────────────────────────────────────────────

export function getRatingClass(r: number): string {
  if (r >= 85) return 'rating-elite';
  if (r >= 70) return 'rating-great';
  if (r >= 60) return 'rating-good';
  if (r >= 50) return 'rating-avg';
  return 'rating-poor';
}

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

export function avg(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export function formScore(form: FormResult[]): number {
  return form.reduce((sum, r) => sum + (r === 'W' ? 3 : r === 'D' ? 1 : 0), 0);
}

export interface MatchPrediction {
  homeXG: number;
  awayXG: number;
  homeGoals: number;
  awayGoals: number;
  outcome: 'home' | 'draw' | 'away';
  confidence: number;
  keyFactors: { label: string; homeValue: string; awayValue: string; winner: 'home' | 'away' | 'even' }[];
}

export function predictMatch(home: TeamMatchData, away: TeamMatchData): MatchPrediction {
  const homeAvgXG = avg(home.last5xGFor);
  const awayAvgXG = avg(away.last5xGFor);
  const homeAvgXGA = avg(home.last5xGAgainst);
  const awayAvgXGA = avg(away.last5xGAgainst);
  const homeForm = formScore(home.last5) / 15;
  const awayForm = formScore(away.last5) / 15;
  const strengthDiff = (home.strengthRating - away.strengthRating) / 100;

  const homeXG = Math.max(
    0.3,
    homeAvgXG * 0.45 + (1 - awayAvgXGA / 3) * 0.3 + homeForm * 0.5 + strengthDiff * 0.8 + 0.35
  );
  const awayXG = Math.max(
    0.1,
    awayAvgXG * 0.45 + (1 - homeAvgXGA / 3) * 0.3 + awayForm * 0.5 - strengthDiff * 0.8
  );

  const homeGoals = Math.round(homeXG);
  const awayGoals = Math.round(awayXG);

  const margin = homeXG - awayXG;
  const outcome: 'home' | 'draw' | 'away' =
    margin > 0.4 ? 'home' : margin < -0.4 ? 'away' : 'draw';

  const confidence = Math.min(
    92,
    Math.max(45, Math.round(55 + Math.abs(margin) * 18 + Math.abs(homeForm - awayForm) * 12))
  );

  const homeShots = avg(home.last5Shots);
  const awayShots = avg(away.last5Shots);
  const homeSOT = avg(home.last5ShotsOnTarget);
  const awaySOT = avg(away.last5ShotsOnTarget);

  const keyFactors = [
    {
      label: 'Team Strength',
      homeValue: home.strengthRating.toFixed(1),
      awayValue: away.strengthRating.toFixed(1),
      winner: home.strengthRating > away.strengthRating ? ('home' as const) : ('away' as const),
    },
    {
      label: 'Form (last 5)',
      homeValue: `${formScore(home.last5)}/15`,
      awayValue: `${formScore(away.last5)}/15`,
      winner: homeForm > awayForm ? ('home' as const) : homeForm < awayForm ? ('away' as const) : ('even' as const),
    },
    {
      label: 'Avg xG For',
      homeValue: homeAvgXG.toFixed(2),
      awayValue: awayAvgXG.toFixed(2),
      winner: homeAvgXG > awayAvgXG ? ('home' as const) : ('away' as const),
    },
    {
      label: 'Avg xG Against',
      homeValue: homeAvgXGA.toFixed(2),
      awayValue: awayAvgXGA.toFixed(2),
      winner: homeAvgXGA < awayAvgXGA ? ('home' as const) : ('away' as const),
    },
    {
      label: 'Avg Shots',
      homeValue: homeShots.toFixed(1),
      awayValue: awayShots.toFixed(1),
      winner: homeShots > awayShots ? ('home' as const) : ('away' as const),
    },
    {
      label: 'Shots on Target',
      homeValue: homeSOT.toFixed(1),
      awayValue: awaySOT.toFixed(1),
      winner: homeSOT > awaySOT ? ('home' as const) : ('away' as const),
    },
    {
      label: 'Home Advantage',
      homeValue: '+0.35 xG',
      awayValue: 'neutral',
      winner: 'home' as const,
    },
  ];

  return { homeXG, awayXG, homeGoals, awayGoals, outcome, confidence, keyFactors };
}

export function computeSimilarity(a: Player, b: Player): number {
  const vec = (p: Player) => [p.pac, p.sho, p.pas, p.dri, p.def, p.phy];
  const va = vec(a);
  const vb = vec(b);
  const dot = va.reduce((s, ai, i) => s + ai * vb[i], 0);
  const magA = Math.sqrt(va.reduce((s, ai) => s + ai * ai, 0));
  const magB = Math.sqrt(vb.reduce((s, bi) => s + bi * bi, 0));
  const cos = dot / (magA * magB);
  return Math.round(cos * 100);
}

export const leagues = ['All', 'Premier League', 'LaLiga EA SPORTS', 'Bundesliga', 'Süper Lig'];
export const positions = ['All', 'GK', 'CB', 'RB', 'LB', 'CDM', 'CM', 'CAM', 'RW', 'LW', 'ST', 'CF'];
