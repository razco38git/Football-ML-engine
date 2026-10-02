import { useEffect, useState } from 'react';
import { api, LEAGUE_NAMES, type Prediction, type TeamStrength } from '../api/client';
import { LeagueGap } from './LeagueStrength';
import { MatchCard } from './MatchPredictor';

const ACCENT = '#00e676';

/** The one place the model runs live rather than serving a stored row. */
export default function WhatIf() {
  const [teams, setTeams] = useState<TeamStrength[]>([]);
  const [home, setHome] = useState('');
  const [away, setAway] = useState('');

  /** The prediction *and* the pairing that produced it. Keeping the labels
   *  beside the result stops a stale card being relabelled when the selection
   *  changes -- which briefly showed Real Madrid as a Premier League side. */
  const [result, setResult] = useState<
    { p: Prediction; leagues: { home: string; away: string } } | null
  >(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Every rated team, across all five leagues -- the point of the tab is the
  // pairing that never happens, and most of those cross a border. The endpoint
  // scores an unlabelled fixture in the home side's league.
  useEffect(() => {
    let live = true;
    api
      .teams(undefined, 200)
      .then(rows => {
        if (!live) return;
        const sorted = [...rows].sort(
          (a, b) => a.league.localeCompare(b.league) || a.team.localeCompare(b.team),
        );
        setTeams(sorted);
        setHome(sorted.find(t => t.league === 'E0')?.team ?? sorted[0]?.team ?? '');
        setAway(sorted.find(t => t.league === 'SP1')?.team ?? sorted[1]?.team ?? '');
      })
      .catch(e => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, []);

  /** Teams grouped by league, so a 96-entry list stays navigable. */
  const byLeague = Object.entries(LEAGUE_NAMES)
    .map(([code, name]) => [name, teams.filter(t => t.league === code)] as const)
    .filter(([, rows]) => rows.length > 0);

  const options = byLeague.map(([name, rows]) => (
    <optgroup key={name} label={name}>
      {rows.map(t => <option key={`${t.league}-${t.team}`} value={t.team}>{t.team}</option>)}
    </optgroup>
  ));

  const sameTeam = home !== '' && home === away;
  const leagueOf = (team: string) => teams.find(t => t.team === team)?.league;

  async function run() {
    if (sameTeam || !home || !away) return;
    setPending(true);
    setError(null);
    try {
      const p = await api.predict(home, away);
      setResult({
        p,
        leagues: {
          home: leagueOf(home) ?? p.league,
          away: leagueOf(away) ?? p.league,
        },
      });
    } catch (e) {
      setResult(null);
      // ApiError carries the endpoint's own message, e.g. "Unknown team 'x'".
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }

  function swap() {
    setHome(away);
    setAway(home);
    setResult(null);
    setError(null);
  }

  /** A card for a pairing nobody is looking at any more is just confusing. */
  function choose(side: 'home' | 'away', team: string) {
    (side === 'home' ? setHome : setAway)(team);
    setResult(null);
    setError(null);
  }

  const selectStyle = {
    background: 'var(--card)',
    border: '1px solid var(--border)',
    color: 'var(--foreground)',
  };

  return (
    <div>
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          What If?
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          A sandbox, not a forecast. Pick any two sides and the model scores the match
          live — probabilities, expected goals, the likeliest scorelines and the features
          that drove them, exactly as the Match Predictor shows them.
          <strong className="block mt-2" style={{ color: 'var(--foreground)' }}>
            These pairings are not scheduled, so nothing here can ever be checked against a
            result.
          </strong>
          Every other tab describes matches that did or will happen, and the Prediction
          Accuracy page is built from those. This one answers a question football never
          asks back.
        </p>
      </div>

      <div
        className="rounded-xl border p-4 mb-6"
        style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
      >
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 flex-1 min-w-[180px]">
            <span className="text-xs uppercase tracking-widest" style={{ color: 'var(--muted-foreground)' }}>
              Home
            </span>
            <select
              value={home}
              onChange={e => choose('home', e.target.value)}
              className="px-3 py-2 rounded-lg text-sm w-full"
              style={selectStyle}
            >
              {options}
            </select>
          </label>

          <button
            onClick={swap}
            title="Swap home and away"
            className="px-3 py-2 rounded-lg text-sm font-display font-bold"
            style={{ background: 'transparent', border: '1px solid var(--border)', color: 'var(--muted-foreground)' }}
          >
            ⇄
          </button>

          <label className="flex flex-col gap-1 flex-1 min-w-[180px]">
            <span className="text-xs uppercase tracking-widest" style={{ color: 'var(--muted-foreground)' }}>
              Away
            </span>
            <select
              value={away}
              onChange={e => choose('away', e.target.value)}
              className="px-3 py-2 rounded-lg text-sm w-full"
              style={selectStyle}
            >
              {options}
            </select>
          </label>

          <button
            onClick={run}
            disabled={pending || sameTeam || !home || !away}
            className="px-5 py-2 rounded-lg text-sm font-display font-bold transition-opacity"
            style={{
              background: ACCENT,
              color: '#000',
              opacity: pending || sameTeam || !home || !away ? 0.4 : 1,
              cursor: pending || sameTeam ? 'not-allowed' : 'pointer',
            }}
          >
            {pending ? 'Predicting…' : 'Predict'}
          </button>
        </div>

        {sameTeam && (
          <p className="mt-3 text-xs" style={{ color: '#ffa726' }}>
            A team cannot play itself.
          </p>
        )}

        {/* The honest caveat: form is today's, whenever these two last met. */}
        <p className="mt-3 text-xs leading-relaxed" style={{ color: 'var(--muted-foreground)' }}>
          Both sides are scored on their form <strong>as of today</strong>, so this answers
          “what if they played this week?” — not what would have happened at some other point
          in the season. Squad strength is last season's, as the model always uses it.
          A cross-league pairing is scored in the home side's league, and both sides bring
          their own squad rating. The model has never seen those two meet, though, so read it
          as a comparison of current form rather than a fixture forecast.
        </p>
      </div>

      {error && (
        <div
          className="rounded-xl border p-4 mb-6 text-sm"
          style={{ background: 'rgba(244,67,54,0.08)', borderColor: 'rgba(244,67,54,0.35)', color: '#f44336' }}
        >
          {error}
        </div>
      )}

      {/* Context for a pairing the fixture list never contains. */}
      {result && <LeagueGap home={result.leagues.home} away={result.leagues.away} />}

      {result?.p.league_adjusted && (
        <div
          className="rounded-xl border p-4 mb-4 text-xs leading-relaxed"
          style={{ background: 'rgba(0,176,255,0.07)', borderColor: 'rgba(0,176,255,0.3)', color: '#7fd3ff' }}
        >
          <strong>Corrected for the gap between the two leagues.</strong> The model
          only ever trains on domestic matches, so it has never seen a tie between two
          divisions and cannot judge one — left alone it predicts much the same whichever
          league each side comes from. The shift applied here is measured from 822 real
          European ties, and improves these predictions out of sample
          (RPS 0.2162 → 0.2140, fitted without the season being scored).
        </div>
      )}

      {result && <MatchCard p={result.p} leagues={result.leagues} />}

      {!result && !error && !pending && (
        <div className="text-center py-16 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Choose two teams and hit Predict.
        </div>
      )}
    </div>
  );
}
