// In-memory store for the currently active Colab endpoint.
// A new Colab runtime registers its URL here, replacing any previous one.

let currentUrl = process.env.COLAB_API_URL || null
let lastRegisteredAt = null

export function setColabUrl(url) {
  currentUrl = url
  lastRegisteredAt = new Date().toISOString()
}

export function getColabUrl() {
  return currentUrl
}

export function getLastRegisteredAt() {
  return lastRegisteredAt
}
