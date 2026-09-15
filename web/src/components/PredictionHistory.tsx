import { pastPredictions, getRatingBg, type PastPrediction } from '../data/footballData';
import DemoBanner from './DemoBanner';

function ScoreBlock({ home, away, label }: { home: number; away: number; label: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="text-xs font-display uppercase tracking-wider" style={{ color: 'var(--muted-foreground)' }}>{label}</div>
      <div className="flex items-center gap-1">
        <span className="font-display font-black text-xl" style={{ color: 'var(--foreground)' }}>{home}</span>
        <span className="font-display font-bold text-base" style={{ color: 'var(--muted-foreground)' }}>-</span>
        <span className="font-display font-black text-xl" style={{ color: 'var(--foreground)' }}>{away}</span>
      </div>
    </div>
  );
}

function XGDiff({ predicted, actual }: { predicted: number; actual: number }) {
  const diff = Math.abs(predicted - actual);
  const color = diff < 0.3 ? '#00e676' : diff < 0.7 ? '#ffea00' : '#f44336';
  return (
    <span className="text-xs font-data" style={{ color }}>
      {diff < 0.01 ? '✓' : diff.toFixed(2)}
    </span>
  );
}

function PredictionRow({ p, index }: { p: PastPrediction; index: number }) {
  const predictedOutcome = p.predictedHomeGoals > p.predictedAwayGoals ? 'home' : p.predictedHomeGoals < p.predictedAwayGoals ? 'away' : 'draw';
  const actualOutcome = p.actualHomeGoals > p.actualAwayGoals ? 'home' : p.actualHomeGoals < p.actualAwayGoals ? 'away' : 'draw';

  const xgHomeDiff = Math.abs(p.predictedHomeXG - p.actualHomeXG);
  const xgAwayDiff = Math.abs(p.predictedAwayXG - p.actualAwayXG);
  const avgXGError = ((xgHomeDiff + xgAwayDiff) / 2).toFixed(2);

  return (
    <div
      className="rounded-xl p-4 transition-all"
      style={{
        background: 'var(--card)',
        border: `1px solid ${p.correct ? 'rgba(0,230,118,0.25)' : 'rgba(244,67,54,0.25)'}`,
      }}
    >
      <div className="flex items-center gap-1 mb-3 justify-between">
        <div className="flex items-center gap-2">
          <span
            className="text-xs px-2 py-0.5 rounded-full font-display font-bold"
            style={{
              background: p.correct ? 'rgba(0,230,118,0.15)' : 'rgba(244,67,54,0.15)',
              color: p.correct ? '#00e676' : '#f44336',
            }}
          >
            {p.correct ? '✓ Correct' : '✗ Wrong'}
          </span>
          <span
            className="text-xs px-2 py-0.5 rounded-full"
            style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}
          >
            {p.confidence}% confidence
          </span>
        </div>
        <span className="text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
          {p.date} · {p.competition}
        </span>
      </div>

      <div className="grid items-center gap-4" style={{ gridTemplateColumns: '1fr auto auto auto 1fr' }}>
        {/* Home */}
        <div className="font-display font-bold text-base" style={{ color: 'var(--foreground)' }}>
          {p.homeLogo} {p.homeTeam}
        </div>

        {/* Predicted score */}
        <ScoreBlock home={p.predictedHomeGoals} away={p.predictedAwayGoals} label="Predicted" />

        {/* vs */}
        <div className="flex flex-col items-center gap-1">
          <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--muted-foreground)' }}>vs</span>
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center"
            style={{
              background: p.correct ? 'rgba(0,230,118,0.1)' : 'rgba(244,67,54,0.1)',
              border: `1px solid ${p.correct ? 'rgba(0,230,118,0.3)' : 'rgba(244,67,54,0.3)'}`,
            }}
          >
            {p.correct ? (
              <span style={{ color: '#00e676', fontSize: 14 }}>✓</span>
            ) : (
              <span style={{ color: '#f44336', fontSize: 14 }}>✗</span>
            )}
          </div>
        </div>

        {/* Actual score */}
        <ScoreBlock home={p.actualHomeGoals} away={p.actualAwayGoals} label="Actual" />

        {/* Away */}
        <div className="font-display font-bold text-base text-right" style={{ color: 'var(--foreground)' }}>
          {p.awayTeam} {p.awayLogo}
        </div>
      </div>

      {/* xG comparison */}
      <div className="mt-3 flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Predicted xG:</span>
          <span className="text-xs font-data font-bold" style={{ color: '#3b82f6' }}>
            {p.predictedHomeXG.toFixed(2)} – {p.predictedAwayXG.toFixed(2)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Actual xG:</span>
          <span className="text-xs font-data font-bold text-white">
            {p.actualHomeXG.toFixed(2)} – {p.actualAwayXG.toFixed(2)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Avg xG error:</span>
          <span className="text-xs font-data font-bold" style={{ color: parseFloat(avgXGError) < 0.3 ? '#00e676' : parseFloat(avgXGError) < 0.7 ? '#ffea00' : '#f44336' }}>
            ±{avgXGError}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Outcome:</span>
          <span className="text-xs font-data font-bold" style={{ color: predictedOutcome === actualOutcome ? '#00e676' : '#f44336' }}>
            {predictedOutcome === actualOutcome ? 'Correct' : `Predicted ${predictedOutcome}, got ${actualOutcome}`}
          </span>
        </div>
      </div>
    </div>
  );
}

export default function PredictionHistory() {
  const correct = pastPredictions.filter(p => p.correct).length;
  const accuracy = Math.round((correct / pastPredictions.length) * 100);

  const allXGErrors = pastPredictions.map(p =>
    (Math.abs(p.predictedHomeXG - p.actualHomeXG) + Math.abs(p.predictedAwayXG - p.actualAwayXG)) / 2
  );
  const avgXGError = allXGErrors.reduce((a, b) => a + b, 0) / allXGErrors.length;

  const avgConfidence = Math.round(
    pastPredictions.reduce((s, p) => s + p.confidence, 0) / pastPredictions.length
  );

  const correctWhen70Plus = pastPredictions.filter(p => p.confidence >= 70 && p.correct).length;
  const totalWhen70Plus = pastPredictions.filter(p => p.confidence >= 70).length;

  return (
    <div>
      <DemoBanner reason="Sample history. Real predictions started being recorded on 15 Sep 2026 and appear here once those matches finish." />
      <div className="mb-6">
        <h2 className="font-display font-bold text-3xl" style={{ color: 'var(--foreground)' }}>
          Prediction Accuracy
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--muted-foreground)' }}>
          Track record of past predictions vs actual results. Used to validate and improve the model.
        </p>
      </div>

      {/* Accuracy summary */}
      <div className="grid gap-4 mb-6" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        {[
          { label: 'Overall Accuracy', value: `${accuracy}%`, sub: `${correct}/${pastPredictions.length} correct`, color: accuracy >= 70 ? '#00e676' : accuracy >= 55 ? '#ffea00' : '#f44336' },
          { label: 'Avg xG Error', value: `±${avgXGError.toFixed(2)}`, sub: 'Goals per team per match', color: avgXGError < 0.4 ? '#00e676' : avgXGError < 0.7 ? '#ffea00' : '#f44336' },
          { label: 'Avg Confidence', value: `${avgConfidence}%`, sub: 'Model certainty', color: '#3b82f6' },
          { label: 'High Confidence', value: `${totalWhen70Plus > 0 ? Math.round((correctWhen70Plus / totalWhen70Plus) * 100) : 0}%`, sub: `When ≥70% confident (${totalWhen70Plus} bets)`, color: '#a78bfa' },
        ].map(stat => (
          <div
            key={stat.label}
            className="rounded-xl p-5"
            style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
          >
            <div className="text-xs font-display uppercase tracking-wider mb-2" style={{ color: 'var(--muted-foreground)' }}>
              {stat.label}
            </div>
            <div className="font-display font-black text-4xl" style={{ color: stat.color }}>
              {stat.value}
            </div>
            <div className="text-xs mt-1" style={{ color: 'var(--muted-foreground)' }}>{stat.sub}</div>
          </div>
        ))}
      </div>

      {/* Accuracy bar over time */}
      <div
        className="rounded-xl p-5 mb-6"
        style={{ background: 'var(--card)', border: '1px solid var(--border)' }}
      >
        <div className="text-xs font-display font-bold uppercase tracking-wider mb-4" style={{ color: 'var(--muted-foreground)' }}>
          Prediction Outcomes (most recent first)
        </div>
        <div className="flex items-end gap-2 h-20">
          {[...pastPredictions].reverse().map((p, i) => (
            <div key={p.id} className="flex flex-col items-center gap-1 flex-1">
              <div
                className="w-full rounded-t-sm transition-all"
                style={{
                  height: `${p.confidence * 0.6}px`,
                  background: p.correct ? '#00e676' : '#f44336',
                  opacity: 0.8,
                }}
                title={`${p.homeTeam} vs ${p.awayTeam} – ${p.confidence}% confidence – ${p.correct ? 'Correct' : 'Wrong'}`}
              />
              <span className="text-xs font-data" style={{ color: 'var(--muted-foreground)' }}>
                {p.confidence}%
              </span>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-3 mt-3">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ background: '#00e676' }} />
            <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Correct</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ background: '#f44336' }} />
            <span className="text-xs" style={{ color: 'var(--muted-foreground)' }}>Incorrect</span>
          </div>
          <span className="text-xs ml-auto" style={{ color: 'var(--muted-foreground)' }}>
            Height = confidence level
          </span>
        </div>
      </div>

      {/* Individual predictions */}
      <div className="flex flex-col gap-3">
        {pastPredictions.map((p, i) => (
          <PredictionRow key={p.id} p={p} index={i} />
        ))}
      </div>

      {/* Model learning note */}
      <div
        className="rounded-xl p-4 mt-6"
        style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.2)' }}
      >
        <div className="text-xs font-display font-bold uppercase tracking-wider mb-2" style={{ color: '#a78bfa' }}>
          Model Learning
        </div>
        <div className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
          Each completed prediction is fed back into the model. Wrong predictions with high confidence are prioritized for weight adjustment.
          xG error data helps calibrate the xG contribution factor. The model currently achieves <strong className="text-white">{accuracy}% outcome accuracy</strong> with
          an average xG error of <strong className="text-white">±{avgXGError.toFixed(2)}</strong> goals per team — improving as more data accumulates.
        </div>
      </div>
    </div>
  );
}
