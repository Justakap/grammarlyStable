import { useEffect, useState } from 'react'
import Header from './components/Header'
import Editor from './components/Editor'
import WritingStyleSelect from './components/WritingStyleSelect'
import ModelStatus from './components/ModelStatus'
import { getModelStatus } from './api/correction'
import { DEFAULT_STYLE, isValidStyle, styleToPersona } from './utils/writingStyles'
import './App.css'

const STATUS_POLL_MS = 20000
const STYLE_KEY = 'hinglish-writing-style'

function loadStyle() {
  try {
    const saved = localStorage.getItem(STYLE_KEY)
    return isValidStyle(saved) ? saved : DEFAULT_STYLE
  } catch {
    return DEFAULT_STYLE
  }
}

export default function App() {
  const [text, setText] = useState('')
  const [correctState, setCorrectState] = useState('connecting')
  const [predictState, setPredictState] = useState('connecting')
  const [sensitivityState, setSensitivityState] = useState('connecting')
  const [style, setStyle] = useState(loadStyle)

  function handleStyleChange(next) {
    setStyle(next)
    try {
      localStorage.setItem(STYLE_KEY, next)
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

        <WritingStyleSelect value={style} onChange={handleStyleChange} />

        <Editor value={text} onChange={setText} onClear={() => setText('')} persona={styleToPersona(style)} />

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
