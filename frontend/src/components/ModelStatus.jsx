const LABELS = {
  connecting: { dot: 'status-dot--connecting', text: 'Connecting' },
  available: { dot: 'status-dot--available', text: 'Available' },
  unavailable: { dot: 'status-dot--unavailable', text: 'Unavailable' },
}

export default function ModelStatus({ label, state }) {
  const { dot, text } = LABELS[state] || LABELS.connecting

  return (
    <div className="model-status">
      {label && <span className="model-status-label">{label}:</span>}
      <span className={`status-dot ${dot}`} aria-hidden="true" />
      <span className="model-status-text">{text}</span>
    </div>
  )
}
