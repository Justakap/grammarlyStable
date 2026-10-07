import { Router } from 'express'
import { getColabUrl, isValidService } from '../lib/colabStore.js'

const router = Router()

const HEALTH_TIMEOUT_MS = 4000

// Reports whether a Colab endpoint is registered and currently reachable.
// Never exposes the actual URL to the caller.
// ?service=correct (default) or ?service=predict
router.get('/model/status', async (req, res) => {
  const service = req.query.service
  if (service !== undefined && !isValidService(service)) {
    return res.status(400).json({ error: 'If provided, "service" must be "correct" or "predict".' })
  }

  const colabUrl = getColabUrl(service)

  if (!colabUrl) {
    return res.json({ available: false })
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS)

  try {
    const upstream = await fetch(`${colabUrl.replace(/\/+$/, '')}/health`, {
      signal: controller.signal,
    })
    return res.json({ available: upstream.ok })
  } catch {
    return res.json({ available: false })
  } finally {
    clearTimeout(timeout)
  }
})

export default router
