import json
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from decision_tree import choose_mode
from emotion_detector import detect_emotion
from memory import get_recent_context, init_db, save_message
from prompt_builder import build_system_prompt


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
CHAT_MODEL = "qwen3:8b"

app = FastAPI(title="Emotional AI Starter")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "local-user"


class ChatResponse(BaseModel):
    reply: str
    emotion: str
    intensity: str
    mode: str


@app.on_event("startup")
def startup() -> None:
    init_db()


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    emotion_result = detect_emotion(user_message)
    mode = choose_mode(
        emotion=emotion_result.emotion,
        intensity=emotion_result.intensity,
        risk=emotion_result.risk
    )
    recent_context = get_recent_context(request.user_id)
    system_prompt = build_system_prompt(
        emotion=emotion_result.emotion,
        intensity=emotion_result.intensity,
        mode=mode,
        recent_context=recent_context
    )

    save_message(
        user_id=request.user_id,
        role="user",
        content=user_message,
        emotion=emotion_result.emotion,
        mode=mode
    )

    reply = call_ollama(system_prompt, user_message)

    save_message(
        user_id=request.user_id,
        role="assistant",
        content=reply,
        emotion=emotion_result.emotion,
        mode=mode
    )

    return ChatResponse(
        reply=reply,
        emotion=emotion_result.emotion,
        intensity=emotion_result.intensity,
        mode=mode
    )


def call_ollama(system_prompt: str, user_message: str) -> str:
    payload = {
        "model": CHAT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "options": {
            "temperature": 0.8,
            "top_p": 0.9
        }
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Make sure Ollama is running and qwen3:8b is installed."
        ) from exc

    return result.get("message", {}).get("content", "").strip()
