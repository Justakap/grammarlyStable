export default function Editor({ value, onChange, onCorrect, onClear, loading }) {
  const charCount = value.length
  const wordCount = value.trim() === '' ? 0 : value.trim().split(/\s+/).length

  return (
    <section className="editor-card">
      <textarea
        className="editor-textarea"
        placeholder="Type your Hinglish sentence here..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={loading}
        rows={7}
      />
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
