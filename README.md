# Hinglish Grammarly

A Grammarly-style writing assistant for Hinglish. Users type a Hinglish sentence,
the app corrects spelling/grammar/punctuation while preserving Hinglish style
(no translation to English or pure Hindi), via a 4B language model running on
Google Colab.

## Architecture

```
React frontend  --POST /api/correct-->  Backend  --POST /correct-->  Colab #1 (grammar correction model)
React frontend  --POST /api/predict-->  Backend  --POST /predict-->  Colab #2 (next-word prediction model)
```

- The frontend only ever talks to the backend (`VITE_API_URL`). It never knows either Colab URL.
- Each feature has its own independent Colab runtime and its own registered URL, tracked by the
  backend under a `service` name: `"correct"` (the default) or `"predict"`. Restarting one
  Colab runtime has no effect on the other.
- A Colab URL changes every time its runtime/tunnel restarts. Each notebook registers its
  current public URL with the backend via `POST /api/colab/register` (authenticated with a
  shared secret token), passing `service: "correct"` or `service: "predict"`.
- The backend stores each service's active Colab URL in memory and forwards requests
  accordingly: `/api/correct` → `<correct_colab_url>/correct`, `/api/predict` → `<predict_colab_url>/predict`.

## Project layout

```
grammarly-hinglish/
  frontend/   Vite + React app (the UI)
  backend/    Express server (stable API, stores/forwards to Colab)
  colab/      FastAPI model server + notebook to run in Google Colab
```

## API contract

### `POST /api/correct` (backend)
Request: `{ "text": "mujhe kal college jana hai" }`
Response: `{ "corrected": "Mujhe kal college jaana hai." }`
`503` if no Colab model is currently registered.

### `POST /api/predict` (backend)
Request: `{ "text": "mujhe kal market" }`
Response: `{ "predictions": ["jana", "jaana", "ja"] }`
`503` if no predict Colab model is currently registered.

### `GET /api/model/status[?service=correct|predict]` (backend)
Response: `{ "available": true }` — never exposes the actual Colab URL. Defaults to `service=correct`.

### `POST /api/colab/register` (backend, called only by Colab)
Header: `Authorization: Bearer <BACKEND_REGISTRATION_TOKEN>`
Request: `{ "url": "https://current-colab-url", "service": "correct" | "predict" }` (`service` defaults to `"correct"`)
Response: `{ "ok": true }` / `401` if unauthorized / `400` if `service` is invalid.

### `POST /correct` (Colab #1 FastAPI)
Request: `{ "text": "mujhe kal college jana hai" }`
Response: `{ "corrected": "Mujhe kal college jaana hai." }`

### `POST /predict` (Colab #2 FastAPI)
Request: `{ "text": "mujhe kal market" }`
Response: `{ "predictions": ["jana", "jaana", "ja"] }`

## Running everything

### 1. Backend — local dev

```bash
cd backend
cp .env.example .env
# edit .env: set BACKEND_REGISTRATION_TOKEN to a long random secret,
# set FRONTEND_ORIGIN to http://localhost:5173 for local dev
npm install
npm run dev
```

Backend runs on `http://localhost:8787` by default.

### 1b. Backend — deploy to Render

The backend is what gets a stable, permanent URL. Colab's URL changes every runtime;
Render's does not, so the frontend always points at Render and never has to change.

1. Push this repo to GitHub (or a repo Render can access).
2. In Render: **New → Web Service**, point it at this repo, with **Root Directory** set to `backend`.
   (A `backend/render.yaml` blueprint is included if you prefer **New → Blueprint**.)
3. Render auto-detects `npm install` / `npm start` (from `backend/package.json`). Health check path: `/health`.
4. Set environment variables in Render's dashboard (**never** commit these):
   - `BACKEND_REGISTRATION_TOKEN` — a long random secret (Colab must send this exact value)
   - `FRONTEND_ORIGIN` — your deployed frontend's URL (comma-separate multiple, e.g. local + prod)
5. Deploy. Render gives you a stable URL like `https://hinglish-grammarly-backend.onrender.com` —
   this is the `BACKEND_URL` the Colab notebook registers against, and the `VITE_API_URL` the frontend calls.

Note: Render's free tier spins down on idle, so the first request after inactivity is slow —
that's expected, not a bug.

### 2. Colab (the model)

1. Open `colab/hinglish_colab.ipynb` in Google Colab (choose a GPU runtime).
2. Upload `colab/colab_server.py` and `colab/register.py` into the Colab filesystem
   (left sidebar → Files → upload), or mount Drive and adjust paths.
3. In Colab's **Secrets** panel (key icon in the left sidebar), add:
   - `BACKEND_URL` — your Render backend URL from step 1b (e.g. `https://hinglish-grammarly-backend.onrender.com`)
   - `BACKEND_REGISTRATION_TOKEN` — must match the value set on Render
   - `NGROK_AUTH_TOKEN` — from https://dashboard.ngrok.com/get-started/your-authtoken
   - `MODEL_NAME` (optional, defaults to `google/gemma-3-4b-it`)
4. Run all cells top to bottom. The last cell keeps the server + tunnel alive.
5. **Every time you start a new Colab runtime, it gets a new public URL automatically** —
   cell 5 (ngrok) generates the new URL and cell 6 (`register_with_backend`) immediately
   sends it to Render via `POST /api/colab/register`. Render overwrites the previously
   stored URL with this new one. If a long-running tunnel ever drops mid-session, just
   re-run cells 5 and 6 to get a fresh URL registered — no frontend or backend changes needed.

### 2b. A second Colab runtime for next-word prediction

The next-word-prediction model runs as its own independent Colab notebook/runtime,
registering under `service: "predict"` instead of the default `"correct"`. It needs its
own `BACKEND_URL` + `BACKEND_REGISTRATION_TOKEN` (same values as the correction notebook)
and its own LoRA checkpoint path. Starting, stopping, or restarting this runtime never
affects the grammar-correction service, since the backend tracks each service's URL
separately.

### 3. Frontend

```bash
cd frontend
cp .env.example .env
# edit .env: set VITE_API_URL to the Render backend URL from step 1b
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` by default. The frontend always calls this one
stable Render URL for `/api/correct` and `/api/model/status` — it reads whatever Colab URL
is currently registered on the backend, so it always gets the latest model endpoint without
ever knowing Colab's address itself.

## Security notes

- `VITE_*` variables are public — never put secrets there.
- `BACKEND_REGISTRATION_TOKEN`, `NGROK_AUTH_TOKEN`, and any Hugging Face token live only
  on the backend/Colab side.
- The backend validates the registration token on every `/api/colab/register` call and
  never returns the Colab URL to the frontend.
