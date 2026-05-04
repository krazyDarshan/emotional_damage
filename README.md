# ponponchan Emotional AI

Local emotional companion app using FastAPI, SQLite, Ollama `qwen3:8b`, and `nomic-embed-text`.

## Run

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

## Features

- Semantic conversation recall with `nomic-embed-text`
- Every user and assistant message saved in SQLite
- Archive search across current or all sessions
- Pin/delete conversation messages
- Add, pin, and delete extracted memories
- Export full local history as JSON
- Qwen-based emotion classification with keyword safety fallback
- Character Studio for name, personality, boundaries, and emotional response modes
- Chat sessions for relationship talk, study stress, personal goals, roleplay mode, and general chat

## Local Models

```powershell
ollama pull qwen3:8b
ollama pull nomic-embed-text
```
