import { useEffect, useRef, useState } from 'react'
import { predictNextWords } from '../api/correction'

export default function Editor({ value, onChange, onCorrect, onClear, loading }) {
  const charCount = value.length
  const wordCount = value.trim() === '' ? 0 : value.trim().split(/\s+/).length

  const [suggestions, setSuggestions] = useState([])
  const textareaRef = useRef(null)
  const requestIdRef = useRef(0)

  async function fetchSuggestions(text) {
    const requestId = ++requestIdRef.current
    const predictions = await predictNextWords(text)
    if (requestIdRef.current !== requestId) return
    setSuggestions(predictions)
  }

  function handleChange(newValue) {
    // A single space just typed at the end is the trigger for suggestions,
    // matching how Grammarly-style predictive text behaves.
    const spaceJustTyped =
      newValue.length === value.length + 1 && newValue.endsWith(' ') && newValue.trim().length > 0

    onChange(newValue)

    if (spaceJustTyped) {
      fetchSuggestions(newValue)
    } else {
      requestIdRef.current++ // invalidate any in-flight fetch
      setSuggestions([])
    }
  }

  function handleSelectSuggestion(word) {
    const textarea = textareaRef.current
    const cursor = textarea ? textarea.selectionStart : value.length
    const before = value.slice(0, cursor)
    const after = value.slice(cursor)
    const needsSpaceBefore = before.length > 0 && !before.endsWith(' ')
    const insertion = `${needsSpaceBefore ? ' ' : ''}${word} `
    const newValue = `${before}${insertion}${after}`

    onChange(newValue)
    setSuggestions([])

    const cursorPos = before.length + insertion.length
    requestAnimationFrame(() => {
      if (textarea) {
        textarea.focus()
        textarea.setSelectionRange(cursorPos, cursorPos)
      }
    })

    fetchSuggestions(newValue)
  }

  useEffect(() => {
    if (loading || value === '') setSuggestions([])
  }, [loading, value])

  return (
    <section className="editor-card">
      <textarea
        ref={textareaRef}
        className="editor-textarea"
        placeholder="Type your Hinglish sentence here..."
        value={value}
        onChange={(e) => handleChange(e.target.value)}
        disabled={loading}
        rows={7}
      />

      {suggestions.length > 0 && !loading && (
        <div className="predict-suggestions" role="listbox" aria-label="Next word suggestions">
          {suggestions.map((word, i) => (
            <button
              key={`${word}-${i}`}
              type="button"
              className="predict-chip"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => handleSelectSuggestion(word)}
            >
              {word}
            </button>
          ))}
        </div>
      )}

      <div className="editor-footer">
        <div className="editor-counts">
          <span>{charCount} characters</span>
          <span className="editor-counts-sep">&middot;</span>
          <span>{wordCount} words</span>
        </div>
        <div className="editor-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onClear}
            disabled={loading || value.length === 0}
          >
            Clear
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onCorrect}
            disabled={loading || value.trim().length === 0}
          >
            {loading ? 'Correcting…' : 'Correct Sentence'}
          </button>
        </div>
      </div>
    </section>
  )
}
