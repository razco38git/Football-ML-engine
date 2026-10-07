import { useState } from 'react';
import {
  api,
  COMPETITION_NAMES,
  LEAGUE_NAMES,
  competitionLabel,
  type Prediction,
  type TeamForm,
} from '../api/client';
import { confidence, formatDate, mean, pct, shortName, teamColor, verdict } from '../api/display';
import { useAsync } from '../api/hooks';
import { CallVersusResult, FinalScore, OutcomeBar, OutcomeSplit, ResultBadge } from './MatchOutcome';

function FormBadge({ result }: { result: string }) {
  const cls = result === 'W' ? 'form-w' : result === 'D' ? 'form-d' : 'form-l';
  return <span className={`form-pill ${cls}`}>{result}</span>;
}

function XGBar({ values, color }: { values: number[]; color: string }) {
  const max = 4;
  return (
    <div className="flex gap-1 items-end h-12">
      {values.map((v, i) => (
        <div key={i} className="flex flex-col items-center gap-0.5" style={{ width: 20 }}>
          <span className="text-[9px] font-data" style={{ color: '#94a3b8' }}>{v.toFixed(1)}</span>
          <div
            className="rounded-sm w-4 transition-all"
            style={{ height: `${(v / max) * 36}px`, background: color, opacity: 0.85, minHeight: 4 }}
          />
        </div>
      ))}
    </div>
  );
}

function TeamSide({ team, form, league, align }: {
  team: string;
  form: TeamForm | null;
  league: string;
  align: 'left' | 'right';
}) {
  const right = align === 'right';
  const color = teamColor(team);
  return (
    <div className={`flex flex-col gap-2 ${right ? 'sm:items-end' : ''}`}>
      <div className={`flex items-center gap-2 ${right ? 'sm:flex-row-reverse' : ''}`}>
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center text-sm font-bold font-display flex-shrink-0"
          style={{ background: color + '22', border: `1px solid ${color}44`, color }}
        >
          {shortName(team)}
        </div>
        <div className={right ? 'sm:text-right' : ''}>
          <div className="font-display font-bold text-lg leading-tight" style={{ color: 'var(--foreground)' }}>
            {team}
          </div>
          <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
            {competitionLabel(league)}
          </div>
        </div>
      </div>

      <div className={`flex gap-1 ${right ? 'sm:flex-row-reverse' : ''}`}>
        {form?.results.length
          ? form.results.map((r, i) => <FormBadge key={i} result={r} />)
          : <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>no prior matches</span>}
      </div>

      {/*
        Both halves of the form, because one alone misleads in opposite
        directions. Says "last N" for the same reason: this sits beside the
        fixture's own xG forecast and was read as one. Barcelona came into
        Sevilla averaging 4.32 xG over five matches against four bottom-eight
        defences and the forecast was 2.10, which looked like an error rather
        than the regression it is.

        Labelled xG and xGA, not colour alone: green-for-created and
        red-for-conceded would be invisible to a red-green colourblind reader,
        and these two numbers are meaningless the wrong way round. The colours
        stay as reinforcement. Each label is kept with its number by
        `whitespace-nowrap`, or a phone wraps "xGA" onto one line and 1.71 onto
        the next, which reads as the wrong number against the wrong word. The
        count comes from the data, not a hardcoded 5, so it stays true if the
        form window ever changes.
      */}
      <div className={`flex flex-wrap gap-3 text-xs ${right ? 'sm:flex-row-reverse' : ''}`} style={{ color: 'var(--muted-foreground)' }}>
        {!!form?.xg_for.length && (
          <span title={`Average xG created and conceded over the last ${form.xg_for.length} matches`}>
            <span className="whitespace-nowrap">
              xG <span className="font-data font-bold" style={{ color: '#00e676' }}>
                {mean(form.xg_for).toFixed(2)}
              </span>
            </span>
            <span style={{ opacity: 0.5 }}> · </span>
            <span className="whitespace-nowrap">
              xGA <span className="font-data font-bold" style={{ color: '#f44336' }}>
                {mean(form.xg_against).toFixed(2)}
              </span>
              <span style={{ opacity: 0.6 }}> (last {form.xg_for.length})</span>
            </span>
          </span>
        )}
        {!!form?.shots_on_target.length && (
          <span title={`Average shots on target over the last ${form.shots_on_target.length} matches`}>
            SOT: <span className="font-data font-bold text-white">{mean(form.shots_on_target).toFixed(1)}</span>
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Squad strength for both sides, shown as its own block.
 *
 * Deliberately separate from the SHAP drivers rather than forced into them:
 * that list is the model's real top-five reasoning, and editing it to always
 * include strength would misrepresent it. Strength reaches the top five in only
 * a minority of fixtures because recent form usually dominates — but a reader
 * still wants to see how the squads compare, so it gets a permanent row here.
 *
 * These are last season's ratings, which is what the model was given: current
 * ratings do not exist until players have enough minutes to be rated.
 */
/**
 * The scoreline distribution, as a grid.
 *
 * It exists to settle one apparent contradiction that no amount of text
 * settles: the single likeliest score is a draw in 67% of matches, including
 * matches with a clear favourite, so a card can read "1-1" and "Villarreal win
 * 59%" at once and look like it is arguing with itself.
 *
 * Both are true, and the grid shows why in one glance. Every cell is one exact
 * score. The diagonal is every draw; the triangle below it is every home win;
 * above it, every away win. The brightest single *cell* can sit on the diagonal
 * while the triangle around it holds far more total probability, because a draw
 * has six ways to happen and a home win has fifteen.
 */
/**
 * Why the likeliest exact score and the likeliest outcome can disagree.
 *
 * They answer different questions. The exact score is one cell of the
 * distribution; the outcome sums a whole region of it. A draw's mass sits on
 * the diagonal -- six scorelines inside 0-5 -- while a home win is spread over
 * fifteen, so the single brightest cell is a draw in 67% of matches, including
 * matches with a clear favourite. That share tracks the model's fitted Dixon-Coles
 * rho (-0.057 as measured); see `top_scores` in dixon_coles.py.
 */
function ScoreVersusOutcome({ p }: { p: Prediction }) {
  const modalIsDraw = p.modal_score_home === p.modal_score_away;
  const best = Math.max(p.prob_home_win, p.prob_draw, p.prob_away_win);
  const outcomeIsDraw = best === p.prob_draw;
  // Nothing to reconcile when the two already agree.
  if (modalIsDraw === outcomeIsDraw) return null;

  const winner = p.prob_home_win > p.prob_away_win ? p.home_team : p.away_team;
  return (
    <div
      className="text-xs text-center leading-snug px-2"
      style={{ color: 'var(--muted-foreground)', maxWidth: 230 }}
    >
      {modalIsDraw ? (
        <>
          a draw lands on one of {DRAW_SCORELINES} scores, a win on{' '}
          {WIN_SCORELINES} — so {winner} is likelier overall at {pct(best)}
        </>
      ) : (
        <>
          no single score is likely, but the draws together reach {pct(p.prob_draw)}
        </>
      )}
    </div>
  );
}

//: Scorelines of each kind inside the 0-5 grid the card reasons about: six
//: draws (0-0 up to 5-5) against fifteen of each win.
const DRAW_SCORELINES = 6;
const WIN_SCORELINES = 15;

function ScoreGrid({ p }: { p: Prediction }) {
  const grid = p.score_grid;
  if (!grid?.length) return null;

  const size = grid.length;
  const peak = Math.max(...grid.flat());
  let home = 0, draw = 0, away = 0, homeCells = 0, drawCells = 0, awayCells = 0;
  for (let h = 0; h < size; h++) {
    for (let a = 0; a < size; a++) {
      if (h > a) { home += grid[h][a]; homeCells++; }
      else if (h === a) { draw += grid[h][a]; drawCells++; }
      else { away += grid[h][a]; awayCells++; }
    }
  }
  const covered = home + draw + away;
  const region = (h: number, a: number) => (h > a ? '#00e676' : h === a ? '#94a3b8' : '#3b82f6');
  const modal = { h: p.modal_score_home, a: p.modal_score_away };
  const outcomeWord = p.prob_home_win >= p.prob_draw && p.prob_home_win >= p.prob_away_win
    ? `a ${p.home_team} win`
    : p.prob_away_win >= p.prob_draw
      ? `a ${p.away_team} win`
      : 'a draw';
  const modalWord = modal.h > modal.a ? `a ${p.home_team} win`
    : modal.h === modal.a ? 'a draw' : `a ${p.away_team} win`;

  return (
    <div className="mt-5">
      <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: 'var(--muted-foreground)' }}>
        Every scoreline, and how the outcome adds up
      </div>

      <div className="flex flex-wrap gap-5 items-start">
        <div>
          <div className="flex">
            <div style={{ width: 26 }} />
            <div className="text-xs text-center" style={{ color: 'var(--muted-foreground)', width: size * 30, fontSize: 10 }}>
              {p.away_team} goals →
            </div>
          </div>
          <div className="flex">
            <div
              className="text-xs flex items-center"
              style={{ color: 'var(--muted-foreground)', fontSize: 10, width: 26, writingMode: 'vertical-rl', transform: 'rotate(180deg)', justifyContent: 'center' }}
            >
              {p.home_team} goals →
            </div>
            <div>
              <div className="flex">
                <div style={{ width: 22 }} />
                {Array.from({ length: size }, (_, a) => (
                  <div key={a} className="text-center font-data" style={{ width: 30, fontSize: 10, color: 'var(--muted-foreground)' }}>{a}</div>
                ))}
              </div>
              {grid.map((row, h) => (
                <div key={h} className="flex items-center">
                  <div className="text-center font-data" style={{ width: 22, fontSize: 10, color: 'var(--muted-foreground)' }}>{h}</div>
                  {row.map((value, a) => {
                    const isModal = h === modal.h && a === modal.a;
                    return (
                      <div
                        key={a}
                        title={`${h}-${a} · ${(value * 100).toFixed(1)}%`}
                        className="flex items-center justify-center font-data"
                        style={{
                          width: 28, height: 24, margin: 1, borderRadius: 3, fontSize: 9,
                          background: region(h, a) + Math.round(18 + (value / peak) * 220).toString(16).padStart(2, '0'),
                          color: value / peak > 0.45 ? '#000' : 'var(--muted-foreground)',
                          outline: isModal ? '2px solid var(--foreground)' : undefined,
                          outlineOffset: -2,
                        }}
                      >
                        {value >= 0.01 ? (value * 100).toFixed(0) : ''}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="text-xs" style={{ color: 'var(--muted-foreground)', maxWidth: 300 }}>
          {([['#00e676', `${p.home_team} win`, home, homeCells],
             ['#94a3b8', 'Draw', draw, drawCells],
             ['#3b82f6', `${p.away_team} win`, away, awayCells]] as const).map(
            ([colour, name, total, cells]) => (
              <div key={name} className="flex items-center gap-2 mb-1.5">
                <span style={{ width: 10, height: 10, borderRadius: 2, background: colour, flexShrink: 0 }} />
                <span style={{ flex: 1 }}>{name}</span>
                <span className="font-data" style={{ color: 'var(--foreground)' }}>{(total * 100).toFixed(0)}%</span>
                <span style={{ fontSize: 10, opacity: 0.7, width: 70, textAlign: 'right' }}>
                  over {cells} scores
                </span>
              </div>
            ),
          )}
          <p className="mt-3 leading-relaxed">
            The outlined cell is the single likeliest exact score,{' '}
            <strong style={{ color: 'var(--foreground)' }}>{modal.h}–{modal.a}</strong>
            {p.prob_modal_score != null && <> at {pct(p.prob_modal_score)}</>} — {modalWord}.
            The most likely <em>outcome</em> is {outcomeWord}, because it adds up{' '}
            {modal.h > modal.a ? homeCells : modal.h === modal.a ? drawCells : awayCells}{' '}
            different scorelines against the one cell.
          </p>
          <p className="mt-2 leading-relaxed" style={{ opacity: 0.75 }}>
            These {size * size} scorelines cover {pct(covered)} of outcomes; the rest is
            higher-scoring tail the model still counts.
          </p>
        </div>
      </div>
    </div>
  );
}


function SquadStrength({ p }: { p: Prediction }) {
  const rows = [
    ['Overall', p.strength_home_overall, p.strength_away_overall],
    ['Attack', p.strength_home_attack, p.strength_away_attack],
    ['Defence', p.strength_home_defence, p.strength_away_defence],
  ] as const;

  // A promoted side has no rating in its new league. That is a real gap in the
  // data, not a failure, and the model sees the same gap.
  const missing = [
    p.strength_home_overall == null ? p.home_team : null,
    p.strength_away_overall == null ? p.away_team : null,
  ].filter(Boolean);

  // Neither side rated: say so rather than rendering nothing. A silently absent
  // section reads as a bug, and leaves the reader wondering whether squad
  // quality was used at all.
  if (missing.length === 2) {
    return (
      <div className="mt-5">
        <div
          className="text-xs font-display font-bold uppercase tracking-wider mb-2"
          style={{ color: 'var(--muted-foreground)' }}
        >
          Squad strength · last season
        </div>
        <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
          Neither {p.home_team} nor {p.away_team} has a squad rating in this
          league from last season, so this prediction rests on form alone.
        </div>
      </div>
    );
  }

  return (
    <div className="mt-5">
      <div
        className="text-xs font-display font-bold uppercase tracking-wider mb-3"
        style={{ color: 'var(--muted-foreground)' }}
      >
        Squad strength · last season
      </div>

      {rows.map(([label, home, away]) => {
        const both = home != null && away != null;
        return (
          <div key={label} className="flex items-center gap-3 mb-2">
            <span
              className="font-data font-bold text-sm text-right"
              style={{ width: 40, color: both && home > away ? '#00e676' : 'var(--foreground)' }}
            >
              {home?.toFixed(1) ?? '—'}
            </span>
            <div className="flex h-1.5 rounded-full overflow-hidden flex-1" style={{ background: 'var(--secondary)' }}>
              {both && (
                <>
                  <div style={{ width: `${(home / (home + away)) * 100}%`, background: '#00e676' }} />
                  <div style={{ width: `${(away / (home + away)) * 100}%`, background: '#3b82f6' }} />
                </>
              )}
            </div>
            <span
              className="font-data font-bold text-sm"
              style={{ width: 40, color: both && away > home ? '#3b82f6' : 'var(--foreground)' }}
            >
              {away?.toFixed(1) ?? '—'}
            </span>
            <span className="text-xs" style={{ color: 'var(--muted-foreground)', width: 56 }}>
              {label}
            </span>
          </div>
        );
      })}

      {missing.length > 0 && (
        <div className="text-xs mt-2" style={{ color: 'var(--muted-foreground)' }}>
          No rating for {missing.join(' or ')} — newly promoted, so there is no
          squad rating from last season.
        </div>
      )}
    </div>
  );
}

/** One prediction, rendered. Exported so the What-If tab shows the same card
 *  rather than a second, subtly different one.
 *
 *  `leagues` overrides the per-side league label. A real fixture has one league
 *  and needs no override, but a hypothetical pairing can cross borders, and the
 *  prediction carries only the league it was *scored* in -- which would label
 *  Man City "LaLiga" in a Real Madrid tie. */
/**
 * What the model said would happen: likeliest exact score, the scores either
 * side of it, a verdict and a confidence.
 *
 * Its own component because a settled match shows it somewhere else. Before
 * kick-off this is the headline; afterwards the headline is the result, and
 * this moves into the analysis panel beside the score grid that explains it.
 * A card that leads with its own forecast and footnotes the final score is
 * answering a question nobody has once the match has been played.
 */
function ForecastSummary({ p }: { p: Prediction }) {
  const { label: outcomeLabel, color: outcomeColor, decisive } = verdict(p);
  const conf = confidence(p);
  return (
    <div className="flex flex-col items-center gap-2">
      {/*
        The call comes first and biggest, because it is what the model is
        actually for. The exact score used to hold this spot in 52px badges,
        which made a ~10% event look like the forecast while the thing the
        model is confident about -- a one-in-three to four-in-five call on the
        outcome -- sat underneath it in 12px.
      */}
      <div className="text-center">
        <div
          className="font-display font-black leading-none tracking-wide"
          style={{ color: outcomeColor, fontSize: 24 }}
        >
          {outcomeLabel}
        </div>
        {/*
          "38% confidence" reads as confidence in a named winner. Where there
          is no favourite there is no winner to be confident about, so show
          what the model actually thinks: all three numbers.
        */}
        {decisive ? (
          <div
            className="font-display font-bold mt-1"
            style={{
              fontSize: 15,
              color: conf >= 60 ? '#00e676' : conf >= 45 ? '#ffea00' : '#ff9100',
            }}
          >
            {conf}% confidence
          </div>
        ) : (
          <div className="font-data mt-1" style={{ fontSize: 13, color: '#94a3b8' }}>
            {pct(p.prob_home_win)} / {pct(p.prob_draw)} / {pct(p.prob_away_win)}
          </div>
        )}
      </div>

      {/*
        Secondary, and sized to say so. Still coloured by the scoreline itself
        rather than by `predicted_outcome`: they answer different questions and
        often disagree -- Roma v Inter is a 50% Roma win whose single most
        likely exact score is 1-1 -- and painting that 1-1 with the home-win
        colour made the card look self-contradictory.
      */}
      <div className="flex flex-col items-center gap-1 pt-1">
        <div className="text-xs" style={{ color: 'var(--muted-foreground)', fontSize: 10 }}>
          likeliest exact score
          {p.prob_modal_score != null && ` · ${pct(p.prob_modal_score)}`}
        </div>
        <div className="flex items-center gap-1">
          <span
            className="overall-badge"
            style={{
              background: p.modal_score_home > p.modal_score_away ? '#00e676' : 'var(--secondary)',
              color: p.modal_score_home > p.modal_score_away ? '#000' : 'var(--foreground)',
              width: 30, height: 30, fontSize: 14, borderRadius: 7,
            }}
          >
            {p.modal_score_home}
          </span>
          <span className="font-display font-bold text-xs" style={{ color: 'var(--muted-foreground)' }}>-</span>
          <span
            className="overall-badge"
            style={{
              background: p.modal_score_away > p.modal_score_home ? '#3b82f6' : 'var(--secondary)',
              color: p.modal_score_away > p.modal_score_home ? '#000' : 'var(--foreground)',
              width: 30, height: 30, fontSize: 14, borderRadius: 7,
            }}
          >
            {p.modal_score_away}
          </span>
        </div>
      </div>

      {/*
        One line to settle the apparent contradiction without making the
        reader open the panel. "1-1" under "Villarreal win" still looks like
        the card arguing with itself, and the answer is not subtle once said
        out loud: a draw has to land on one of six exact scores, a home win can
        arrive by any of fifteen. The grid under "Show analysis" is the same
        statement as a picture.
      */}
      {p.prob_modal_score != null && <ScoreVersusOutcome p={p} />}

      {/*
        The runners-up, which are the point. The leader clears them by about a
        percentage point, so showing one score alone invites the reader to
        treat a ~10% event as the forecast -- and since a draw's mass sits on
        the diagonal while a win is spread across many scorelines, that leader
        is a draw in 67% of matches even when a side is a clear favourite.
      */}
      {p.likely_scores?.length > 1 && (
        <div className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--muted-foreground)' }}>
          {p.likely_scores.slice(1).map(s => (
            <span
              key={`${s.home}-${s.away}`}
              className="px-1.5 py-0.5 rounded font-data"
              style={{ background: 'var(--secondary)', border: '1px solid var(--border)' }}
              title={`${s.home}-${s.away} in ${pct(s.probability)} of simulations`}
            >
              {s.home}-{s.away} <span style={{ opacity: 0.7 }}>{pct(s.probability)}</span>
            </span>
          ))}
        </div>
      )}

      <div className="text-xs text-center" style={{ color: 'var(--muted-foreground)' }}>
        xG: {p.expected_goals_home.toFixed(2)} – {p.expected_goals_away.toFixed(2)}
      </div>
    </div>
  );
}

export function MatchCard({ p, leagues }: {
  p: Prediction;
  leagues?: { home: string; away: string };
}) {
  const [expanded, setExpanded] = useState(false);

  // `verdict` and `confidence` moved with the forecast block into
  // `ForecastSummary`; nothing in the card itself needs them now.
  const settled = p.actual_result !== null;
  const correct = settled && p.actual_result === p.predicted_outcome;

  return (
    <div
      className="rounded-xl overflow-hidden border cursor-pointer transition-all"
      style={{ background: 'var(--card)', borderColor: 'var(--border)' }}
      onClick={() => setExpanded(e => !e)}
    >
      <div
        className="px-4 py-2 flex items-center justify-between"
        style={{ background: 'rgba(255,255,255,0.03)', borderBottom: '1px solid var(--border)' }}
      >
        <span className="text-xs font-medium tracking-widest uppercase" style={{ color: 'var(--muted-foreground)' }}>
          {competitionLabel(p.competition && p.competition !== 'domestic' ? p.competition : p.league)}
        </span>
        <div className="flex items-center gap-2">
          {/* Just the tick. The score itself is now the headline below. */}
          {settled && <ResultBadge correct={correct} />}
          <span className="text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
            {formatDate(p.date)}
          </span>
        </div>
      </div>

      <div className="p-5">
        {/*
          Three columns on a desktop, stacked on a phone. At 375px the fixed
          `1fr auto 1fr` pushed the away side off the right edge of the card --
          its badge and form pills were simply unreachable.
        */}
        <div className="grid items-center gap-4 grid-cols-1 sm:grid-cols-[1fr_auto_1fr]">
          <TeamSide team={p.home_team} form={p.form_home} league={leagues?.home ?? p.league} align="left" />

          {/*
            Before kick-off, the forecast. Afterwards, what actually happened
            -- with the forecast kept as the small line underneath, because
            this page's whole claim is that the two can be compared.
          */}
          {settled ? (
            <div className="flex flex-col items-center flex-shrink-0" style={{ minWidth: 104 }}>
              <FinalScore
                homeGoals={p.actual_home_goals!}
                awayGoals={p.actual_away_goals!}
                expectedHome={p.expected_goals_home}
                expectedAway={p.expected_goals_away}
              />
            </div>
          ) : (
            <ForecastSummary p={p} />
          )}

          <TeamSide team={p.away_team} form={p.form_away} league={leagues?.away ?? p.league} align="right" />
        </div>

        <div className="mt-4">
          <OutcomeBar home={p.prob_home_win} draw={p.prob_draw} away={p.prob_away_win} />
          {/*
            Named, not "Home / Draw / Away". The reader had to map two of the
            three onto clubs himself, and on a settled card nothing said which
            one had happened -- which is the only question he arrived with.
          */}
          <OutcomeSplit
            home={p.home_team}
            away={p.away_team}
            probHome={p.prob_home_win}
            probDraw={p.prob_draw}
            probAway={p.prob_away_win}
            actual={p.actual_result}
            pct={pct}
          />
          {settled && (
            <CallVersusResult
              predicted={p.predicted_outcome}
              actual={p.actual_result!}
              home={p.home_team}
              away={p.away_team}
              correct={correct}
            />
          )}
        </div>

        {p.market_prob_home !== null && (
          <div className="flex justify-between mt-2 text-xs" style={{ color: 'var(--muted-foreground)' }}>
            <span className="tracking-wide uppercase" style={{ fontSize: 10 }}>Bookmaker</span>
            <span className="font-data">
              {pct(p.market_prob_home)} / {pct(p.market_prob_draw!)} / {pct(p.market_prob_away!)}
            </span>
          </div>
        )}
      </div>

      <div
        className="flex items-center justify-center py-2 text-xs gap-1"
        style={{ color: 'var(--muted-foreground)', borderTop: '1px solid var(--border)' }}
      >
        <span>{expanded ? 'Hide' : 'Show'} analysis</span>
        <span style={{ transform: expanded ? 'rotate(180deg)' : undefined, display: 'inline-block', transition: 'transform 0.2s' }}>▼</span>
      </div>

      {expanded && (
        <div className="px-5 pb-5" style={{ borderTop: '1px solid var(--border)' }}>
          {/*
            What we had said, for a match that has been played. It is the first
            thing in the panel because it is what the rest of the panel is
            evidence for, and it sits directly above the score grid, which is
            the same statement as a picture.
          */}
          {settled && (
            <div
              className="pt-4 flex flex-col items-center"
              style={{ borderBottom: '1px solid var(--border)', paddingBottom: 16 }}
            >
              <div
                className="uppercase tracking-wider mb-2"
                style={{ color: 'var(--muted-foreground)', fontSize: 9 }}
              >
                What we forecast
              </div>
              <ForecastSummary p={p} />
            </div>
          )}
          <div className="pt-4 grid gap-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div>
              <div className="text-xs font-display font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--muted-foreground)' }}>
                {p.home_team} · Last 5 xG
              </div>
              <XGBar values={p.form_home?.xg_for ?? []} color="#00e676" />
            </div>
            <div>
              <div className="text-xs font-display font-bold uppercase tracking-wider mb-2 text-right" style={{ color: 'var(--muted-foreground)' }}>
                {p.away_team} · Last 5 xG
              </div>
              <div className="flex justify-end">
                <XGBar values={p.form_away?.xg_for ?? []} color="#3b82f6" />
              </div>
            </div>
          </div>

          <SquadStrength p={p} />

          <ScoreGrid p={p} />

          {/* SHAP drivers: the model's actual reasoning, not a summary of it. */}
          {!!p.drivers_home.length && (
            <div className="mt-5">
              <div className="text-xs font-display font-bold uppercase tracking-wider mb-3" style={{ color: 'var(--muted-foreground)' }}>
                Why the model predicts this
              </div>
              <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
                {([['home', p.home_team, p.drivers_home, '#00e676'],
                   ['away', p.away_team, p.drivers_away, '#3b82f6']] as const).map(
                  ([key, team, drivers, color]) => (
                    <div key={key}>
                      <div className="text-xs mb-2" style={{ color }}>{team} expected goals</div>
                      {drivers.map((d, i) => (
                        <div key={i} className="flex items-center gap-2 mb-1.5">
                          <span
                            className="font-data text-xs font-bold flex-shrink-0"
                            style={{ color: d.contribution > 0 ? '#00e676' : '#f44336', width: 48 }}
                          >
                            {d.contribution > 0 ? '+' : ''}{d.contribution.toFixed(3)}
                          </span>
                          <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>{d.label}</span>
                        </div>
                      ))}
                    </div>
                  ),
                )}
              </div>
            </div>
          )}

          <div className="mt-4 grid gap-3" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div>
              <div className="text-xs mb-1" style={{ color: 'var(--muted-foreground)' }}>Shots last 5</div>
              <div className="flex gap-1">
                {(p.form_home?.shots ?? []).map((s, i) => (
                  <div key={i} className="text-center">
                    <div className="text-xs font-data font-bold text-white">{s}</div>
                    <div className="w-4 rounded-sm mt-0.5" style={{ height: `${(s / 25) * 24}px`, background: '#00e67644', border: '1px solid #00e676' }} />
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div className="text-xs mb-1 text-right" style={{ color: 'var(--muted-foreground)' }}>Shots last 5</div>
              <div className="flex gap-1 justify-end">
                {(p.form_away?.shots ?? []).map((s, i) => (
                  <div key={i} className="text-center">
                    <div className="text-xs font-data font-bold text-white">{s}</div>
                    <div className="w-4 rounded-sm mt-0.5" style={{ height: `${(s / 25) * 24}px`, background: '#3b82f644', border: '1px solid #3b82f6' }} />
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 flex gap-4 text-xs" style={{ color: 'var(--muted-foreground)' }}>
            <span>Over 2.5 goals: <span className="font-data font-bold text-white">{pct(p.prob_over_2_5)}</span></span>
            <span>Both teams score: <span className="font-data font-bold text-white">{pct(p.prob_btts)}</span></span>
          </div>
        </div>
      )}
    </div>
  );
}

type Mode = 'upcoming' | 'recent';

export default function MatchPredictor() {
  const [mode, setMode] = useState<Mode>('upcoming');
  const [league, setLeague] = useState('All');
  const isCompetition = league in COMPETITION_NAMES;

  const { data, loading, error, reload } = useAsync(
    () => (mode === 'upcoming' ? api.upcoming(league) : api.matches(league, 20)),
    [mode, league],
  );

  return (
    <div>
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Match Predictor
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Predictions from team form, expected goals, shot quality and pressing, across 29,000 matches.
          Click any match to see the model's reasoning.
        </p>
      </div>

      <div className="flex flex-wrap gap-3 mb-6 items-center">
        <div className="flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          {(['upcoming', 'recent'] as Mode[]).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className="px-4 py-2 text-sm font-display font-bold transition-colors"
              style={{
                background: mode === m ? 'rgba(0,230,118,0.15)' : 'transparent',
                color: mode === m ? '#00e676' : 'var(--muted-foreground)',
              }}
            >
              {m === 'upcoming' ? 'Upcoming' : 'Recent results'}
            </button>
          ))}
        </div>

        <select
          value={league}
          onChange={e => setLeague(e.target.value)}
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)' }}
        >
          <option value="All">All competitions</option>
          {Object.entries(LEAGUE_NAMES).map(([code, name]) => (
            <option key={code} value={code}>{name}</option>
          ))}
          {Object.entries(COMPETITION_NAMES).map(([code, name]) => (
            <option key={code} value={code}>{name}</option>
          ))}
        </select>
      </div>

      {/* A Champions League view that shows 7 of 18 matches has to say so. */}
      {isCompetition && (
        <div
          className="rounded-xl border p-4 mb-6 text-xs leading-relaxed"
          style={{ background: 'rgba(0,176,255,0.07)', borderColor: 'rgba(0,176,255,0.3)', color: '#7fd3ff' }}
        >
          <strong>Only ties between clubs from the big five.</strong> We can rate a
          club only if it plays in England, Spain, Germany, Italy or France, so a match
          against Benfica, Ajax or Galatasaray has no form, no expected goals and no
          squad rating to predict from and is not shown. That covers about{' '}
          <strong>33%</strong> of this season's Champions League — 48 of its 144 matches.
          These predictions are also weaker than domestic ones: measured over 822 past
          European ties, the model keeps about half its usual edge over a simple base
          rate. The Prediction Accuracy tab reports them separately for that reason.
        </div>
      )}

      {loading && (
        <div className="text-center py-16 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Loading predictions…
        </div>
      )}

      {error && (
        <div className="rounded-xl p-6 text-center" style={{ background: 'rgba(244,67,54,0.08)', border: '1px solid rgba(244,67,54,0.25)' }}>
          <div className="text-sm mb-3" style={{ color: '#f44336' }}>{error}</div>
          <button
            onClick={reload}
            className="px-4 py-2 rounded-lg text-sm font-display font-bold"
            style={{ background: 'rgba(244,67,54,0.15)', color: '#f44336' }}
          >
            Retry
          </button>
        </div>
      )}

      {/* An empty upcoming list is normal mid-week, not a failure -- but the
          message has to match the mode, or "Recent results" offers to show you
          recent results. */}
      {!loading && !error && data?.length === 0 && (
        <div className="rounded-xl p-8 text-center" style={{ background: 'var(--card)', border: '1px solid var(--border)' }}>
          <div className="text-sm mb-2" style={{ color: 'var(--foreground)' }}>
            {mode === 'upcoming' ? 'No fixtures published right now' : 'No results to show'}
          </div>
          <div className="text-xs mb-4" style={{ color: 'var(--muted-foreground)' }}>
            {mode === 'upcoming'
              ? isCompetition
                ? 'European rounds are a fortnight apart, so this is empty between them.'
                : 'The fixture feed covers the next few days. Mid-week it is often empty.'
              : 'Nothing has been played and scored here yet.'}
          </div>
          {mode === 'upcoming' && (
          <button
            onClick={() => setMode('recent')}
            className="px-4 py-2 rounded-lg text-sm font-display font-bold"
            style={{ background: 'rgba(0,230,118,0.15)', color: '#00e676' }}
          >
            Show recent results instead
          </button>
          )}
        </div>
      )}

      <div className="grid gap-4">
        {data?.map(p => <MatchCard key={`${p.league}-${p.date}-${p.home_team}`} p={p} />)}
      </div>
    </div>
  );
}
