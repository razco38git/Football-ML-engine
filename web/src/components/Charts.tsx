/**
 * The two charts this site needs, drawn as inline SVG.
 *
 * There is no charting library here and adding one for two shapes would be a
 * poor trade: `XGBar` and `ScoreGrid` in `MatchPredictor.tsx` are already drawn
 * by hand, and these follow the same approach so the whole site renders the
 * same way.
 */

/**
 * A reliability diagram: predicted probability against how often it happened.
 *
 * The single most informative picture of a probabilistic model. Points on the
 * diagonal mean the numbers can be taken at face value — of the outcomes given
 * 30%, 30% happened. Points below it mean over-confidence, above it means
 * under-confidence.
 *
 * Every bin is plotted, including the small ones. The previous version of this
 * dropped bins under 100 matches, which quietly hid the low-probability end —
 * exactly where a model is most often wrong, and where a reader most needs to
 * know. Sample size is shown as the radius instead, so a thin bin looks thin
 * rather than disappearing.
 */
export function ReliabilityDiagram({
  bins,
  size = 260,
}: {
  bins: { mean_predicted: number; observed_rate: number; n: number }[];
  size?: number;
}) {
  const pad = 30;
  const span = size - pad * 2;
  const x = (v: number) => pad + v * span;
  const y = (v: number) => size - pad - v * span;
  const biggest = Math.max(...bins.map(b => b.n), 1);

  return (
    <svg width={size} height={size} role="img" aria-label="Calibration: predicted against observed">
      {[0, 0.25, 0.5, 0.75, 1].map(t => (
        <g key={t}>
          <line x1={x(t)} y1={y(0)} x2={x(t)} y2={y(1)} stroke="var(--border)" strokeWidth={1} />
          <line x1={x(0)} y1={y(t)} x2={x(1)} y2={y(t)} stroke="var(--border)" strokeWidth={1} />
          <text x={x(t)} y={size - pad + 14} textAnchor="middle" fontSize={9} fill="var(--muted-foreground)">
            {Math.round(t * 100)}%
          </text>
          <text x={pad - 6} y={y(t) + 3} textAnchor="end" fontSize={9} fill="var(--muted-foreground)">
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}

      {/* Perfect calibration. Every point should sit on this line. */}
      <line
        x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)}
        stroke="var(--muted-foreground)" strokeWidth={1} strokeDasharray="4 3" opacity={0.6}
      />

      <polyline
        points={bins.map(b => `${x(b.mean_predicted)},${y(b.observed_rate)}`).join(' ')}
        fill="none" stroke="#00e676" strokeWidth={1.5} opacity={0.5}
      />
      {bins.map(b => (
        <circle
          key={b.mean_predicted}
          cx={x(b.mean_predicted)}
          cy={y(b.observed_rate)}
          r={3 + 4 * Math.sqrt(b.n / biggest)}
          fill="#00e676"
          fillOpacity={0.75}
          stroke="#00e676"
        >
          <title>
            {`We said ${Math.round(b.mean_predicted * 100)}% — it happened ${Math.round(
              b.observed_rate * 100,
            )}% of the time, over ${b.n.toLocaleString()} outcomes`}
          </title>
        </circle>
      ))}

      <text x={size / 2} y={size - 4} textAnchor="middle" fontSize={9} fill="var(--muted-foreground)">
        we said
      </text>
      <text
        x={10} y={size / 2} fontSize={9} fill="var(--muted-foreground)"
        transform={`rotate(-90 10 ${size / 2})`} textAnchor="middle"
      >
        it happened
      </text>
    </svg>
  );
}

/**
 * A plain line, scaled to its own range.
 *
 * Deliberately unlabelled: it sits inside a table row, where the shape is the
 * message and the number beside it is the value. The tooltip carries the rest.
 */
export function Sparkline({
  values,
  width = 90,
  height = 24,
  color = '#00e676',
  title,
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  title?: string;
}) {
  if (values.length < 2) return null;
  const low = Math.min(...values);
  const high = Math.max(...values);
  // A flat series would divide by zero and, worse, read as a dramatic line at
  // whatever height the rounding landed on. Pin it to the middle instead.
  const range = high - low || 1;
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * (width - 2) + 1;
    const y = height - 1 - ((v - low) / range) * (height - 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const rising = values[values.length - 1] >= values[0];

  return (
    <svg width={width} height={height} role="img" aria-label={title ?? 'trend'}>
      {title && <title>{title}</title>}
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke={rising ? color : '#f44336'}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle
        cx={(width - 2) + 1}
        cy={height - 1 - ((values[values.length - 1] - low) / range) * (height - 2)}
        r={2}
        fill={rising ? color : '#f44336'}
      />
    </svg>
  );
}
