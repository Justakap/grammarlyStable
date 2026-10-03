import { Router } from 'express'
import { setColabUrl } from '../lib/colabStore.js'

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

  const { url } = req.body || {}

  if (typeof url !== 'string' || !isValidHttpUrl(url)) {
    return res.status(400).json({ error: 'A valid "url" field is required.' })
  }

  setColabUrl(url)
  console.log(`Registered new Colab endpoint: ${url}`)

  return res.json({ ok: true })
})

export default router
