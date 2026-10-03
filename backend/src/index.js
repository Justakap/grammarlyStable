import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import correctRouter from './routes/correct.js'
import colabRouter from './routes/colab.js'
import statusRouter from './routes/status.js'

const app = express()
const PORT = process.env.PORT || 8787

// FRONTEND_ORIGIN may be a single origin or a comma-separated list
// (e.g. local dev + the deployed frontend), so both work without code changes.
const allowedOrigins = (process.env.FRONTEND_ORIGIN || '*')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean)

app.use(
  cors({
    origin: allowedOrigins.includes('*') ? '*' : allowedOrigins,
  })
)
app.use(express.json({ limit: '100kb' }))

app.get('/health', (_req, res) => res.json({ ok: true }))

app.use('/api', correctRouter)
app.use('/api', colabRouter)
app.use('/api', statusRouter)

// eslint-disable-next-line no-unused-vars
app.use((err, _req, res, _next) => {
  console.error('Unhandled error:', err)
  res.status(500).json({ error: 'Internal server error.' })
})

app.listen(PORT, () => {
  console.log(`Hinglish Grammarly backend listening on port ${PORT}`)
})
