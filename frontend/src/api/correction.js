const API_URL = import.meta.env.VITE_API_URL

if (!API_URL) {
  // eslint-disable-next-line no-console
  console.warn('VITE_API_URL is not set. Falling back to relative /api paths.')
}

const BASE = API_URL || ''

async function parseJsonSafe(response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

function withTimeout(ms) {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), ms)
  return { signal: controller.signal, cancel: () => clearTimeout(id) }
}

/**
 * Sends Hinglish text to our backend for correction.
 * Throws an Error with a user-friendly message on any failure.
 */
export async function correctText(text) {
  const { signal, cancel } = withTimeout(30000)

  let response
  try {
    response = await fetch(`${BASE}/api/correct`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal,
    })
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('The request took too long. Please try again.')
    }
    throw new Error('Could not reach the server. Check your connection and try again.')
  } finally {
    cancel()
  }

  const data = await parseJsonSafe(response)

  if (response.status === 503) {
    throw new Error(
      (data && data.error) || 'The correction model is currently unavailable. Please try again shortly.'
    )
  }

  if (!response.ok) {
    throw new Error((data && data.error) || 'Something went wrong while correcting your sentence.')
  }

  if (!data || typeof data.corrected !== 'string') {
    throw new Error('Received an unexpected response from the server.')
  }

  return data.corrected
}

/**
 * Fetches whether a given model service ("correct" or "predict") is
 * currently reachable. Never throws — resolves to { available: false }
 * on any failure.
 */
export async function getModelStatus(service = 'correct') {
  const { signal, cancel } = withTimeout(8000)

  try {
    const response = await fetch(`${BASE}/api/model/status?service=${encodeURIComponent(service)}`, { signal })
    cancel()
    if (!response.ok) return { available: false }
    const data = await parseJsonSafe(response)
    return { available: Boolean(data && data.available) }
  } catch {
    cancel()
    return { available: false }
  }
}

/**
 * Fetches next-word suggestions for the given (partial) Hinglish text.
 * Never throws — resolves to [] on any failure, so a flaky prediction
 * model never disrupts typing.
 */
export async function predictNextWords(text) {
  const { signal, cancel } = withTimeout(8000)

  try {
    const response = await fetch(`${BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal,
    })
    cancel()
    if (!response.ok) return []
    const data = await parseJsonSafe(response)
    if (!data || !Array.isArray(data.predictions)) return []
    return data.predictions.filter((word) => typeof word === 'string' && word.length > 0)
  } catch {
    cancel()
    return []
  }
}
