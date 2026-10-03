import { Router } from 'express'
import { getColabUrl } from '../lib/colabStore.js'

const router = Router()

const HEALTH_TIMEOUT_MS = 4000

// Reports whether a Colab endpoint is registered and currently reachable.
// Never exposes the actual URL to the caller.
router.get('/model/status', async (_req, res) => {
  const colabUrl = getColabUrl()

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
