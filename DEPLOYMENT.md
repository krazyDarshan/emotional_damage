# Deploy Ponponchan Emotional AI

This app is one FastAPI service that serves both the API and the frontend.

The important model detail:

- The web app uses Ollama through `OLLAMA_HOST`.
- Local default: `http://127.0.0.1:11434`
- Public deployment needs `OLLAMA_HOST` to point to a reachable Ollama server, usually a GPU VPS or model server.

## Fastest Phone Test

Use this when you only want to open the app from your phone while your computer is on.

1. Start Ollama and make sure the model exists:

```powershell
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

2. Start the app:

```powershell
cd C:\Users\darsh\OneDrive\Desktop\ai\emotional_damage
.\.venv\Scripts\Activate.ps1
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Expose it with a tunnel:

```powershell
ngrok http 8000
```

Or use a Cloudflare quick tunnel:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Open the HTTPS URL on your phone.

## Render Web Service

This makes the UI public, but AI replies work only if `OLLAMA_HOST` points to a public Ollama/model endpoint.

Render settings:

- Build command: `pip install -r requirements.txt`
- Start command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variables:
  - `OLLAMA_HOST`: public Ollama endpoint, for example `https://your-model-server.example.com`
  - `CHAT_MODEL`: `qwen3:8b`
  - `EMBED_MODEL`: `nomic-embed-text`
  - `DATA_DIR`: `/var/data` if you attach a persistent disk

The included `render.yaml` has these defaults ready for a Blueprint deployment.

## Docker/VPS

Build and run the app container:

```powershell
docker build -t ponponchan .
docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 ponponchan
```

For a real VPS, run Ollama on the same server or set `OLLAMA_HOST` to a separate model server.

## Health Check

After deployment, open:

```text
https://your-app-url/api/health
```

It should return `{"ok": true, ...}`.
