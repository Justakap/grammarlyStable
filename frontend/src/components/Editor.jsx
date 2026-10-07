import { useCallback, useLayoutEffect, useRef, useState } from 'react'
import { correctText, predictNextWords } from '../api/correction'
import { diffWords } from '../utils/wordDiff'
import { getCaretOffset, setCaretOffset } from '../utils/caret'

const CHECK_DEBOUNCE_MS = 900

export default function Editor({ value, onChange, onClear }) {
  const charCount = value.length
  const wordCount = value.trim() === '' ? 0 : value.trim().split(/\s+/).length

  const [suggestions, setSuggestions] = useState([])
  const [issues, setIssues] = useState([])
  const [checking, setChecking] = useState(false)
  const [hoveredIssue, setHoveredIssue] = useState(null)

  const editableRef = useRef(null)
  const pendingCaretRef = useRef(null)
  const predictRequestIdRef = useRef(0)
  const checkRequestIdRef = useRef(0)
  const checkTimerRef = useRef(null)

  const runCheck = useCallback((text) => {
    const trimmed = text.trim()
    if (!trimmed) {
      setIssues([])
      return
    }
    const requestId = ++checkRequestIdRef.current
    setChecking(true)
    correctText(trimmed)
      .then((corrected) => {
        if (checkRequestIdRef.current !== requestId) return
        setIssues(diffWords(text, corrected))
      })
      .catch(() => {
        // A flaky/unavailable correction model shouldn't interrupt typing.
      })
      .finally(() => {
        if (checkRequestIdRef.current === requestId) setChecking(false)
      })
  }, [])

  function scheduleCheck(text) {
    clearTimeout(checkTimerRef.current)
    checkTimerRef.current = setTimeout(() => runCheck(text), CHECK_DEBOUNCE_MS)
  }

  async function fetchSuggestions(text) {
    const requestId = ++predictRequestIdRef.current
    const predictions = await predictNextWords(text)
    if (predictRequestIdRef.current !== requestId) return
    setSuggestions(predictions)
  }

  function handleInput(e) {
    const newValue = e.currentTarget.textContent
    pendingCaretRef.current = getCaretOffset(editableRef.current)

    const spaceJustTyped =
      newValue.length === value.length + 1 && newValue.endsWith(' ') && newValue.trim().length > 0

    onChange(newValue)
    checkRequestIdRef.current++ // invalidate any in-flight check against the old text
    setHoveredIssue(null)
    if (issues.length > 0) setIssues([]) // drop stale underlines while editing

    if (spaceJustTyped) {
      fetchSuggestions(newValue)
    } else {
      predictRequestIdRef.current++
      setSuggestions([])
    }

    scheduleCheck(newValue)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault()
      document.execCommand('insertText', false, '\n')
    }
  }

  function handlePaste(e) {
    e.preventDefault()
    document.execCommand('insertText', false, e.clipboardData.getData('text/plain'))
  }

  function applyIssueFix(issue) {
    const newValue = value.slice(0, issue.start) + issue.suggestion + value.slice(issue.end)
    pendingCaretRef.current = issue.start + issue.suggestion.length
    onChange(newValue)
    setHoveredIssue(null)
    clearTimeout(checkTimerRef.current)
    runCheck(newValue)
  }

  function handleSelectSuggestion(word) {
    const cursor = pendingCaretRef.current ?? getCaretOffset(editableRef.current) ?? value.length
    const before = value.slice(0, cursor)
    const after = value.slice(cursor)
    const needsSpaceBefore = before.length > 0 && !before.endsWith(' ')
    const insertion = `${needsSpaceBefore ? ' ' : ''}${word} `
    const newValue = `${before}${insertion}${after}`

    pendingCaretRef.current = before.length + insertion.length
    onChange(newValue)
    setSuggestions([])

    clearTimeout(checkTimerRef.current)
    runCheck(newValue)
    fetchSuggestions(newValue)
  }

  function handleContainerClick(e) {
    const span = e.target.closest('.issue-underline')
    if (!span) return
    const spans = Array.from(editableRef.current.querySelectorAll('.issue-underline'))
    const issueSegments = issues.filter((s) => s.type === 'issue')
    const seg = issueSegments[spans.indexOf(span)]
    if (seg) applyIssueFix(seg)
  }

  function handleContainerMouseOver(e) {
    const span = e.target.closest('.issue-underline')
    if (!span) return
    const spans = Array.from(editableRef.current.querySelectorAll('.issue-underline'))
    const issueSegments = issues.filter((s) => s.type === 'issue')
    const seg = issueSegments[spans.indexOf(span)]
    if (seg) setHoveredIssue({ segment: seg, rect: span.getBoundingClientRect() })
  }

  function handleContainerMouseOut(e) {
    if (e.target.closest('.issue-underline')) setHoveredIssue(null)
  }

  // Imperatively sync the contentEditable DOM from `value`/`issues`, since
  // React's normal reconciliation fights contentEditable's own DOM
  // mutations. The caret is restored from `pendingCaretRef`, set right
  // before whichever state change triggered this sync.
  useLayoutEffect(() => {
    const el = editableRef.current
    if (!el) return

    el.innerHTML = ''
    if (issues.length === 0) {
      if (value.length > 0) el.appendChild(document.createTextNode(value))
    } else {
      issues.forEach((seg) => {
        if (seg.type === 'issue') {
          const span = document.createElement('span')
          span.className = 'issue-underline'
          span.textContent = seg.text
          el.appendChild(span)
        } else if (seg.text.length > 0) {
          el.appendChild(document.createTextNode(seg.text))
        }
      })
    }

    if (document.activeElement === el && pendingCaretRef.current != null) {
      setCaretOffset(el, pendingCaretRef.current)
    }
    pendingCaretRef.current = null
  }, [value, issues])

  return (
    <section className="editor-card">
      <div
        ref={editableRef}
        className="editor-textarea"
        contentEditable
        suppressContentEditableWarning
        data-placeholder="Type your Hinglish sentence here..."
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        onPaste={handlePaste}
        onClick={handleContainerClick}
        onMouseOver={handleContainerMouseOver}
        onMouseOut={handleContainerMouseOut}
      />

      {hoveredIssue && (
        <div
          className="issue-tooltip"
          style={{
            top: hoveredIssue.rect.bottom + window.scrollY + 6,
            left: hoveredIssue.rect.left + window.scrollX,
          }}
        >
          {hoveredIssue.segment.suggestion || 'Remove'}
        </div>
      )}

      {suggestions.length > 0 && (
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
          {checking && <span className="editor-checking">Checking…</span>}
        </div>
        <div className="editor-actions">
          <button type="button" className="btn btn-secondary" onClick={onClear} disabled={value.length === 0}>
            Clear
          </button>
        </div>
      </div>
    </section>
  )
}
