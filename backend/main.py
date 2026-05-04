from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from decision_tree import choose_mode
from emotion_detector import detect_emotion_ai
from memory import (
    backfill_message_embeddings,
    create_session,
    delete_memory,
    delete_message,
    ensure_default_sessions,
    export_user_data,
    extract_and_save_memories,
    get_recent_context,
    get_relevant_conversation,
    get_relevant_memories,
    get_session,
    init_db,
    list_conversation,
    list_memories,
    list_sessions,
    pin_memory,
    pin_message,
    save_memory,
    save_message,
    search_conversation,
    update_session
)
from ollama_client import generate_reply
from prompt_builder import build_system_prompt, load_character, save_character


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="Emotional AI Starter")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "local-user"
    session_id: str = "general"


class ChatResponse(BaseModel):
    reply: str
    emotion: str
    intensity: str
    risk: str
    confidence: float = 0.0
    reason: str = ""
    mode: str
    session_id: str
    user_message_id: int
    assistant_message_id: int
    new_memories: list[str] = []
    memories_used: int = 0
    archive_used: int = 0


class MemoryRequest(BaseModel):
    content: str
    user_id: str = "local-user"
    pinned: bool = True


class SessionRequest(BaseModel):
    user_id: str = "local-user"
    title: str
    description: str = ""
    mode_hint: str = ""


class PinRequest(BaseModel):
    user_id: str = "local-user"
    pinned: bool = True


class CharacterRequest(BaseModel):
    character: dict


@app.on_event("startup")
def startup() -> None:
    init_db()
    ensure_default_sessions("local-user")


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

    ensure_default_sessions(request.user_id)
    session = get_session(request.user_id, request.session_id)
    emotion_result = detect_emotion_ai(user_message)
    mode = choose_mode(
        emotion=emotion_result.emotion,
        intensity=emotion_result.intensity,
        risk=emotion_result.risk
    )

    user_message_id = save_message(
        user_id=request.user_id,
        session_id=session["id"],
        role="user",
        content=user_message,
        emotion=emotion_result.emotion,
        mode=mode
    )

    new_memories = extract_and_save_memories(request.user_id, user_message)
    recent_context = get_recent_context(request.user_id, session_id=session["id"])
    long_term_memories = get_relevant_memories(request.user_id, user_message)
    relevant_past_conversation = get_relevant_conversation(
        request.user_id,
        user_message,
        session_id=session["id"]
    )
    system_prompt = build_system_prompt(
        emotion=emotion_result.emotion,
        intensity=emotion_result.intensity,
        mode=mode,
        recent_context=recent_context,
        long_term_memories=long_term_memories,
        relevant_past_conversation=relevant_past_conversation,
        session=session
    )

    try:
        reply = generate_reply(system_prompt, user_message)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Make sure Ollama is running and qwen3:8b is installed."
        ) from exc

    assistant_message_id = save_message(
        user_id=request.user_id,
        session_id=session["id"],
        role="assistant",
        content=reply,
        emotion=emotion_result.emotion,
        mode=mode
    )

    return ChatResponse(
        reply=reply,
        emotion=emotion_result.emotion,
        intensity=emotion_result.intensity,
        risk=emotion_result.risk,
        confidence=emotion_result.confidence,
        reason=emotion_result.reason,
        mode=mode,
        session_id=session["id"],
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        new_memories=new_memories,
        memories_used=count_lines(long_term_memories),
        archive_used=count_lines(relevant_past_conversation)
    )


@app.get("/api/sessions")
def sessions(user_id: str = Query(default="local-user")) -> dict:
    return {"sessions": list_sessions(user_id)}


@app.post("/api/sessions")
def add_session(request: SessionRequest) -> dict:
    if not request.title.strip():
        raise HTTPException(status_code=400, detail="Session title cannot be empty")
    session = create_session(
        user_id=request.user_id,
        title=request.title,
        description=request.description,
        mode_hint=request.mode_hint
    )
    return {"session": session, "sessions": list_sessions(request.user_id)}


@app.put("/api/sessions/{session_id}")
def edit_session(session_id: str, request: SessionRequest) -> dict:
    session = update_session(
        user_id=request.user_id,
        session_id=session_id,
        title=request.title,
        description=request.description,
        mode_hint=request.mode_hint
    )
    return {"session": session, "sessions": list_sessions(request.user_id)}


@app.get("/api/memories")
def memories(user_id: str = Query(default="local-user")) -> dict:
    return {"memories": list_memories(user_id)}


@app.post("/api/memories")
def add_memory(request: MemoryRequest) -> dict:
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory cannot be empty")

    saved = save_memory(
        user_id=request.user_id,
        content=content,
        source_message="manual memory",
        importance=10,
        pinned=request.pinned
    )
    return {"saved": saved, "memories": list_memories(request.user_id)}


@app.patch("/api/memories/{memory_id}/pin")
def set_memory_pin(memory_id: int, request: PinRequest) -> dict:
    updated = pin_memory(request.user_id, memory_id, request.pinned)
    return {"updated": updated, "memories": list_memories(request.user_id)}


@app.delete("/api/memories/{memory_id}")
def remove_memory(memory_id: int, user_id: str = Query(default="local-user")) -> dict:
    deleted = delete_memory(user_id, memory_id)
    return {"deleted": deleted, "memories": list_memories(user_id)}


@app.get("/api/history")
def history(
    user_id: str = Query(default="local-user"),
    session_id: str = Query(default="all"),
    limit: int = Query(default=80, ge=1, le=500),
    offset: int = Query(default=0, ge=0)
) -> dict:
    return list_conversation(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        offset=offset
    )


@app.get("/api/history/search")
def search_history(
    q: str = Query(default=""),
    user_id: str = Query(default="local-user"),
    session_id: str = Query(default="all"),
    limit: int = Query(default=30, ge=1, le=80)
) -> dict:
    query = q.strip()
    if not query:
        return {"results": []}
    return {
        "results": search_conversation(
            user_id=user_id,
            query=query,
            session_id=session_id,
            limit=limit
        )
    }


@app.patch("/api/history/{message_id}/pin")
def set_message_pin(message_id: int, request: PinRequest) -> dict:
    updated = pin_message(request.user_id, message_id, request.pinned)
    return {"updated": updated}


@app.delete("/api/history/{message_id}")
def remove_message(message_id: int, user_id: str = Query(default="local-user")) -> dict:
    deleted = delete_message(user_id, message_id)
    return {"deleted": deleted}


@app.post("/api/embeddings/backfill")
def rebuild_embeddings(
    user_id: str = Query(default="local-user"),
    session_id: str = Query(default="all"),
    limit: int = Query(default=120, ge=1, le=1000)
) -> dict:
    count = backfill_message_embeddings(
        user_id=user_id,
        session_id=None if session_id == "all" else session_id,
        limit=limit
    )
    return {"embedded": count}


@app.get("/api/export")
def export_history(user_id: str = Query(default="local-user")) -> JSONResponse:
    payload = export_user_data(user_id)
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": "attachment; filename=ponponchan-history.json"}
    )


@app.get("/api/character")
def character() -> dict:
    return {"character": load_character()}


@app.put("/api/character")
def update_character(request: CharacterRequest) -> dict:
    return {"character": save_character(request.character)}


def count_lines(text: str) -> int:
    if not text:
        return 0
    return len([line for line in text.splitlines() if line.strip()])
