const LABELS = {
  connecting: { dot: 'status-dot--connecting', text: 'Connecting' },
  available: { dot: 'status-dot--available', text: 'Model available' },
  unavailable: { dot: 'status-dot--unavailable', text: 'Model unavailable' },
}

export default function ModelStatus({ state }) {
  const { dot, text } = LABELS[state] || LABELS.connecting

  return (
    <div className="model-status">
      <span className={`status-dot ${dot}`} aria-hidden="true" />
      <span className="model-status-text">{text}</span>
    </div>
  )
}
