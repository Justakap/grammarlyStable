import { useState } from 'react'

export default function OutputCard({ original, corrected }) {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(corrected)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      // Clipboard API unavailable; silently ignore.
    }
  }

  return (
    <section className="output-card">
      <div className="output-block output-block--original">
        <h2 className="output-label">Original</h2>
        <p className="output-text output-text--original">{original}</p>
      </div>

      <div className="output-block output-block--corrected">
        <h2 className="output-label">Corrected</h2>
        <p className="output-text output-text--corrected">{corrected}</p>
        <button type="button" className="btn btn-copy" onClick={handleCopy}>
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
    </section>
  )
}
