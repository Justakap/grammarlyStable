import { useEffect, useState } from 'react'
import Header from './components/Header'
import Editor from './components/Editor'
import PersonaSelect from './components/PersonaSelect'
import ModelStatus from './components/ModelStatus'
import { getModelStatus } from './api/correction'
import './App.css'

const STATUS_POLL_MS = 20000
const PERSONA_KEY = 'hinglish-persona'

function loadPersona() {
  try {
    return localStorage.getItem(PERSONA_KEY) || 'none'
  } catch {
    return 'none'
  }
}

export default function App() {
  const [text, setText] = useState('')
  const [correctState, setCorrectState] = useState('connecting')
  const [predictState, setPredictState] = useState('connecting')
  const [sensitivityState, setSensitivityState] = useState('connecting')
  const [persona, setPersona] = useState(loadPersona)

  function handlePersonaChange(next) {
    setPersona(next)
    try {
      localStorage.setItem(PERSONA_KEY, next)
    } catch {
      // Remembering the choice is a convenience only.
    }
  }

  useEffect(() => {
    let cancelled = false

    async function pollStatus() {
      const [correct, predict, sensitivity] = await Promise.all([
        getModelStatus('correct'),
        getModelStatus('predict'),
        getModelStatus('sensitivity'),
      ])
      if (cancelled) return
      setCorrectState(correct.available ? 'available' : 'unavailable')
      setPredictState(predict.available ? 'available' : 'unavailable')
      setSensitivityState(sensitivity.available ? 'available' : 'unavailable')
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

        <PersonaSelect value={persona} onChange={handlePersonaChange} />

        <Editor value={text} onChange={setText} onClear={() => setText('')} persona={persona} />

        <footer className="app-footer">
          <ModelStatus label="Correction" state={correctState} />
          <span className="app-footer-sep">&middot;</span>
          <ModelStatus label="Prediction" state={predictState} />
          <span className="app-footer-sep">&middot;</span>
          <ModelStatus label="Tone" state={sensitivityState} />
        </footer>
      </div>
    </div>
  )
}
