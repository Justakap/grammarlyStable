import { Router } from 'express'
import { setColabUrl, isValidService } from '../lib/colabStore.js'

const router = Router()

function isAuthorized(req) {
  const token = process.env.BACKEND_REGISTRATION_TOKEN
  if (!token) return false

  const header = req.get('authorization') || ''
  const provided = header.startsWith('Bearer ') ? header.slice(7) : null
  return provided === token
}

function isValidHttpUrl(value) {
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

// Only the Colab runtime should ever call this, using the shared registration token.
router.post('/colab/register', (req, res) => {
  if (!isAuthorized(req)) {
    return res.status(401).json({ error: 'Unauthorized.' })
  }

  const { url, service } = req.body || {}

  if (typeof url !== 'string' || !isValidHttpUrl(url)) {
    return res.status(400).json({ error: 'A valid "url" field is required.' })
  }

  if (service !== undefined && !isValidService(service)) {
    return res.status(400).json({ error: 'If provided, "service" must be "correct" or "predict".' })
  }

  setColabUrl(url, service)
  console.log(`Registered new Colab endpoint for "${service || 'correct'}": ${url}`)

  return res.json({ ok: true })
})

export default router
