// In-memory store for the currently active Colab endpoint(s).
// Each Colab runtime registers its URL under a service name ("correct",
// "predict", ...), replacing any previous URL for that service only.

const VALID_SERVICES = new Set(['correct', 'predict'])
const DEFAULT_SERVICE = 'correct'

const state = {
  correct: { url: process.env.COLAB_API_URL || null, lastRegisteredAt: null },
  predict: { url: process.env.COLAB_PREDICT_API_URL || null, lastRegisteredAt: null },
}

export function isValidService(service) {
  return VALID_SERVICES.has(service)
}

export function normalizeService(service) {
  return service && VALID_SERVICES.has(service) ? service : DEFAULT_SERVICE
}

export function setColabUrl(url, service = DEFAULT_SERVICE) {
  const key = normalizeService(service)
  state[key] = { url, lastRegisteredAt: new Date().toISOString() }
}

export function getColabUrl(service = DEFAULT_SERVICE) {
  const key = normalizeService(service)
  return state[key].url
}

export function getLastRegisteredAt(service = DEFAULT_SERVICE) {
  const key = normalizeService(service)
  return state[key].lastRegisteredAt
}
