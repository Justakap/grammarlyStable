import { useEffect, useState } from 'react'
import Header from './components/Header'
import Editor from './components/Editor'
import ModelStatus from './components/ModelStatus'
import { getModelStatus } from './api/correction'
import './App.css'

const STATUS_POLL_MS = 20000

export default function App() {
  const [text, setText] = useState('')
  const [correctState, setCorrectState] = useState('connecting')
  const [predictState, setPredictState] = useState('connecting')

  useEffect(() => {
    let cancelled = false

    async function pollStatus() {
      const [correct, predict] = await Promise.all([getModelStatus('correct'), getModelStatus('predict')])
      if (cancelled) return
      setCorrectState(correct.available ? 'available' : 'unavailable')
      setPredictState(predict.available ? 'available' : 'unavailable')
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
          <ModelStatus label="Correction" state={correctState} />
          <span className="app-footer-sep">&middot;</span>
          <ModelStatus label="Prediction" state={predictState} />
        </footer>
      </div>
    </div>
  )
}
