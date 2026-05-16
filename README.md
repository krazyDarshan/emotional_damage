# Ponponchan Emotional AI

Local emotional companion app using FastAPI, SQLite, Ollama `qwen3:8b`, and summary-first memory.

## Run Locally

```powershell
cd C:\Users\darsh\OneDrive\Desktop\ai\emotional_damage
.\.venv\Scripts\Activate.ps1
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Local Models

```powershell
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## Environment Variables

- `OLLAMA_HOST`: Ollama server URL. Default: `http://127.0.0.1:11434`
- `CHAT_MODEL`: chat model name. Default: `qwen3:8b`
- `EMBED_MODEL`: embedding model name. Default: `nomic-embed-text`
- `DATA_DIR`: SQLite data folder. Default: `data`

## Features

- Login and register pages
- Multiple chat sessions
- Summary-first long-term context
- Recent raw chat retention only
- Editable pinned memories
- Qwen-based emotion classification with keyword safety fallback
- Character Studio for personality, tone, boundaries, and emotional modes
- Export memory data as JSON

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md).
