const BANDS = {
  safe: { label: 'Safe', hint: 'This reads as friendly and appropriate.' },
  sensitive: { label: 'Sensitive', hint: 'This may come across as harsh or touchy. Consider rewording.' },
  offensive: { label: 'Offensive', hint: 'This is likely to offend. Think twice before sending.' },
}

export default function SensitivityMeter({ result, loading }) {
  if (!result) return null
  const band = BANDS[result.band] || BANDS.safe
  const score = Math.max(0, Math.min(100, result.score))

  return (
    <div className={`sens sens--${result.band}`} aria-live="polite">
      <div className="sens-row">
        <span className="sens-title">Tone check</span>
        <span className="sens-badge">{band.label}</span>
        <span className="sens-score">{Math.round(score)}/100</span>
        {loading && <span className="sens-updating">Updating…</span>}
      </div>
      <div
        className="sens-track"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(score)}
        aria-label="Sensitivity score"
      >
        <div className="sens-fill" style={{ width: `${Math.max(score, 3)}%` }} />
      </div>
      {result.band !== 'safe' && <p className="sens-hint">{band.hint}</p>}
    </div>
  )
}
