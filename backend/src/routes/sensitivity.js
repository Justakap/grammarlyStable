import { Router } from 'express'
import { getColabUrl } from '../lib/colabStore.js'

const router = Router()

const COLAB_TIMEOUT_MS = Number(process.env.COLAB_REQUEST_TIMEOUT_MS || 25000)
const BANDS = new Set(['safe', 'sensitive', 'offensive'])

// Scores how sensitive/offensive a message is: 0 (safe) to 100 (offensive).
router.post('/sensitivity', async (req, res) => {
  const { text } = req.body || {}

  if (typeof text !== 'string' || text.trim().length === 0) {
    return res.status(400).json({ error: 'Please provide a non-empty "text" field.' })
  }

  if (text.length > 2000) {
    return res.status(400).json({ error: 'Text is too long. Please shorten your sentence.' })
  }

  const colabUrl = getColabUrl('sensitivity')
  if (!colabUrl) {
    return res.status(503).json({ error: 'Sensitivity model is currently unavailable.' })
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), COLAB_TIMEOUT_MS)

  try {
    const headers = { 'Content-Type': 'application/json' }
    if (process.env.COLAB_SENSITIVITY_API_KEY) {
      headers['x-api-key'] = process.env.COLAB_SENSITIVITY_API_KEY
    }

    const upstream = await fetch(`${colabUrl.replace(/\/+$/, '')}/sensitivity`, {
      method: 'POST',
      headers,
      // The Colab server caps text at 1000 chars.
      body: JSON.stringify({ text: text.trim().slice(0, 1000) }),
      signal: controller.signal,
    })

    if (!upstream.ok) {
      console.error(`Colab (sensitivity) responded with status ${upstream.status}`)
      return res.status(502).json({ error: 'Sensitivity model returned an error. Please try again.' })
    }

    const data = await upstream.json().catch(() => null)

    if (!data || typeof data.score !== 'number' || !BANDS.has(data.band)) {
      console.error('Colab (sensitivity) returned an invalid response shape')
      return res.status(502).json({ error: 'Received an invalid response from the sensitivity model.' })
    }

    return res.json({ score: data.score, band: data.band })
  } catch (err) {
    if (err.name === 'AbortError') {
      return res.status(504).json({ error: 'The sensitivity model timed out. Please try again.' })
    }
    console.error('Failed to reach Colab (sensitivity) model:', err.message)
    return res.status(503).json({ error: 'Sensitivity model is currently unavailable.' })
  } finally {
    clearTimeout(timeout)
  }
})

export default router
