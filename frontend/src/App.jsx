import { useEffect, useState } from 'react'
import Header from './components/Header'
import Editor from './components/Editor'
import ModelStatus from './components/ModelStatus'
import { getModelStatus } from './api/correction'
import './App.css'

const STATUS_POLL_MS = 20000

export default function App() {
  const [text, setText] = useState('')
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

  return (
    <div className="app-shell">
      <div className="app-container">
        <Header />

        <Editor value={text} onChange={setText} onClear={() => setText('')} />

        <footer className="app-footer">
          <span className="app-footer-label">Model status:</span>
          <ModelStatus state={modelState} />
        </footer>
      </div>
    </div>
  )
}
