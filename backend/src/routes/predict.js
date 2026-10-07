import { Router } from 'express'
import { getColabUrl } from '../lib/colabStore.js'

const router = Router()

const COLAB_TIMEOUT_MS = Number(process.env.COLAB_REQUEST_TIMEOUT_MS || 25000)

router.post('/predict', async (req, res) => {
  const { text } = req.body || {}

  if (typeof text !== 'string' || text.trim().length === 0) {
    return res.status(400).json({ error: 'Please provide a non-empty "text" field.' })
  }

  if (text.length > 2000) {
    return res.status(400).json({ error: 'Text is too long. Please shorten your sentence.' })
  }

  const colabUrl = getColabUrl('predict')
  if (!colabUrl) {
    return res.status(503).json({ error: 'Prediction model is currently unavailable.' })
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), COLAB_TIMEOUT_MS)

  try {
    const headers = { 'Content-Type': 'application/json' }
    if (process.env.COLAB_PREDICT_API_KEY) {
      headers['x-api-key'] = process.env.COLAB_PREDICT_API_KEY
    }

    const upstream = await fetch(`${colabUrl.replace(/\/+$/, '')}/predict`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ text: text.trim() }),
      signal: controller.signal,
    })

    if (!upstream.ok) {
      console.error(`Colab (predict) responded with status ${upstream.status}`)
      return res.status(502).json({ error: 'Prediction model returned an error. Please try again.' })
    }

    const data = await upstream.json().catch(() => null)

    if (!data || !Array.isArray(data.predictions)) {
      console.error('Colab (predict) returned an invalid response shape')
      return res.status(502).json({ error: 'Received an invalid response from the prediction model.' })
    }

    return res.json({ predictions: data.predictions })
  } catch (err) {
    if (err.name === 'AbortError') {
      return res.status(504).json({ error: 'The prediction model timed out. Please try again.' })
    }
    console.error('Failed to reach Colab (predict) model:', err.message)
    return res.status(503).json({ error: 'Prediction model is currently unavailable.' })
  } finally {
    clearTimeout(timeout)
  }
})

export default router
