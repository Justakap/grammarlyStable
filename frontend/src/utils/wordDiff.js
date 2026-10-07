function tokenize(text) {
  return text.match(/[A-Za-z0-9']+|[^A-Za-z0-9']+/g) || []
}

function isWord(token) {
  return /^[A-Za-z0-9']+$/.test(token)
}

function lcsPairs(wordsA, wordsB) {
  const n = wordsA.length
  const m = wordsB.length
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0))

  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = wordsA[i] === wordsB[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }

  const pairs = []
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (wordsA[i] === wordsB[j]) {
      pairs.push([i, j])
      i++
      j++
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      i++
    } else {
      j++
    }
  }
  return pairs
}

/**
 * Aligns `original` against `corrected` and returns segments covering the
 * full `original` string in order, each `{ type, text, start, end, suggestion? }`.
 * `type` is "same" or "issue"; `start`/`end` are character offsets into
 * `original`, so an issue segment can be replaced in place.
 */
export function diffWords(original, corrected) {
  const a = tokenize(original)
  const b = tokenize(corrected)

  const wordIdxA = []
  const wordsA = []
  a.forEach((t, idx) => {
    if (isWord(t)) {
      wordIdxA.push(idx)
      wordsA.push(t.toLowerCase())
    }
  })

  const wordIdxB = []
  const wordsB = []
  b.forEach((t, idx) => {
    if (isWord(t)) {
      wordIdxB.push(idx)
      wordsB.push(t.toLowerCase())
    }
  })

  const matches = lcsPairs(wordsA, wordsB).map(([p, q]) => [wordIdxA[p], wordIdxB[q]])

  const segments = []
  let offset = 0
  let lastA = 0
  let lastB = 0

  function pushSegment(type, tokens, suggestion) {
    const text = tokens.join('')
    if (text.length === 0) return
    segments.push({ type, text, suggestion, start: offset, end: offset + text.length })
    offset += text.length
  }

  function pushGap(gapTokensA, gapTokensB) {
    const gapAText = gapTokensA.join('')
    const gapBText = gapTokensB.join('')
    if (gapAText.length === 0) return
    if (gapAText === gapBText || gapAText.trim() === '') {
      pushSegment('same', gapTokensA)
    } else {
      pushSegment('issue', gapTokensA, gapBText)
    }
  }

  for (const [ai, bi] of matches) {
    pushGap(a.slice(lastA, ai), b.slice(lastB, bi))

    const matchedToken = a[ai]
    if (matchedToken === b[bi]) {
      pushSegment('same', [matchedToken])
    } else {
      // Same word, different case (e.g. a capitalization fix).
      pushSegment('issue', [matchedToken], b[bi])
    }

    lastA = ai + 1
    lastB = bi + 1
  }

  pushGap(a.slice(lastA), b.slice(lastB))

  return segments
}
