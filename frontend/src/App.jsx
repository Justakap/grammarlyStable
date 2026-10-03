import { useEffect, useRef, useState } from 'react'
import Header from './components/Header'
import Editor from './components/Editor'
import OutputCard from './components/OutputCard'
import ModelStatus from './components/ModelStatus'
import ErrorBanner from './components/ErrorBanner'
import { correctText, getModelStatus } from './api/correction'
import './App.css'

const STATUS_POLL_MS = 20000

export default function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null) // { original, corrected }
  const [error, setError] = useState('')
  const [modelState, setModelState] = useState('connecting')

  useEffect(() => {
    let cancelled = false

    async function pollStatus() {
      const { available } = await getModelStatus()
      if (!cancelled) setModelState(available ? 'available' : 'unavailable')
    }

    pollStatus()
    const id = setInterval(pollStatus, STATUS_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const requestIdRef = useRef(0)

  async function handleCorrect() {
    const trimmed = text.trim()
    if (!trimmed) {
      setError('Please type a sentence before correcting.')
      return
    }

    const requestId = ++requestIdRef.current
    setLoading(true)
    setError('')

    try {
      const corrected = await correctText(trimmed)
      if (requestIdRef.current !== requestId) return
      setResult({ original: trimmed, corrected })
    } catch (err) {
      if (requestIdRef.current !== requestId) return
      setError(err.message || 'Something went wrong. Please try again.')
      setResult(null)
    } finally {
      if (requestIdRef.current === requestId) setLoading(false)
    }
  }

  function handleClear() {
    setText('')
    setResult(null)
    setError('')
  }

  return (
    <div className="app-shell">
      <div className="app-container">
        <Header />

        <Editor
          value={text}
          onChange={setText}
          onCorrect={handleCorrect}
          onClear={handleClear}
          loading={loading}
        />

        <ErrorBanner message={error} />

        {!result && !loading && !error && (
          <section className="empty-state">
            <p>Your corrected Hinglish sentence will appear here.</p>
          </section>
        )}

        {loading && (
          <section className="loading-state">
            <span className="spinner" aria-hidden="true" />
            <p>Correcting your sentence…</p>
          </section>
        )}

        {!loading && result && (
          <OutputCard original={result.original} corrected={result.corrected} />
        )}

        <footer className="app-footer">
          <span className="app-footer-label">Model status:</span>
          <ModelStatus state={modelState} />
        </footer>
      </div>
    </div>
  )
}
