/**
 * Marks a view that is still running on mock data.
 *
 * Player ratings, similarity and team strength all depend on the FBref player
 * pipeline, which does not exist yet. Until it does, these screens show the
 * placeholder data the design shipped with. Labelling that plainly matters more
 * than it might seem: a site whose whole premise is "here is how accurate our
 * predictions are" cannot quietly present invented numbers as real ones.
 */
export default function DemoBanner({ reason }: { reason: string }) {
  return (
    <div
      className="rounded-xl px-4 py-3 mb-6 flex items-start gap-3"
      style={{ background: 'rgba(255,145,0,0.08)', border: '1px solid rgba(255,145,0,0.3)' }}
    >
      <span className="text-base leading-none mt-0.5">⚠</span>
      <div>
        <div className="text-sm font-display font-bold" style={{ color: '#ff9100' }}>
          Placeholder data
        </div>
        <div className="text-xs mt-0.5" style={{ color: 'var(--muted-foreground)' }}>
          {reason}
        </div>
      </div>
    </div>
  );
}
