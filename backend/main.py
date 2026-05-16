from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from decision_tree import choose_mode
from emotion_detector import detect_emotion_ai
from memory import (
    authenticate_user,
    backfill_message_embeddings,
    clear_user_saved_text,
    create_auth_session,
    create_session,
    create_user,
    delete_auth_session,
    delete_memory,
    delete_message,
    ensure_default_sessions,
    export_user_data,
    extract_and_save_memories,
    get_recent_context,
    get_relevant_memories,
    get_session,
    get_session_summary,
    get_unsummarized_messages,
    get_user_by_token,
    init_db,
    list_conversation,
    list_memories,
    list_sessions,
    DB_PATH,
    pin_memory,
    pin_message,
    save_memory,
    save_message,
    save_session_summary,
    search_conversation,
    update_session,
)
from ollama_client import CHAT_MODEL, EMBED_MODEL, OLLAMA_HOST, generate_reply, ollama_chat
from prompt_builder import build_system_prompt, load_character, save_character


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="Ponponchan Emotional AI")


class AuthRequest(BaseModel):
    username: str
    password: str
    display_name: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    emotion: str
    intensity: str
    risk: str
    confidence: float = 0.0
    reason: str = ""
    mode: str
    session_id: str
    new_memories: list[int] = []
    memories_used: int = 0
    summary_used: bool = False
    summary_compacted: bool = False


class MemoryRequest(BaseModel):
    title: str = "important memory"
    content: str
    emotion: str = "neutral"
    importance: int = 4
    pinned: bool = True
    session_id: str | None = None


class SessionRequest(BaseModel):
    title: str
    description: str = ""
    mode_hint: str = "supportive"


class PinRequest(BaseModel):
    pinned: bool = True


class CharacterRequest(BaseModel):
    character: dict[str, Any]


class SummaryRequest(BaseModel):
    session_id: str | None = None


def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def require_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = extract_bearer_token(authorization)
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    ensure_default_sessions(str(user["id"]))
    return user


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


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "ollama_host_configured": bool(OLLAMA_HOST),
        "database_path": str(DB_PATH),
        "auth_mode": "stateless-token",
    }


@app.post("/api/auth/register")
def register(request: AuthRequest) -> dict[str, Any]:
    try:
        user = create_user(request.username, request.password, request.display_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = create_auth_session(int(user["id"]))
    return {"token": token, "user": user, "sessions": list_sessions(str(user["id"]))}


@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict[str, Any]:
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")
    ensure_default_sessions(str(user["id"]))
    token = create_auth_session(int(user["id"]))
    return {"token": token, "user": user, "sessions": list_sessions(str(user["id"]))}


@app.get("/api/auth/me")
def me(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    return {"user": user, "sessions": list_sessions(str(user["id"]))}


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, bool]:
    token = extract_bearer_token(authorization)
    if token:
        delete_auth_session(token)
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user: dict[str, Any] = Depends(require_user)) -> ChatResponse:
    user_id = str(user["id"])
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session = get_session(user_id, request.session_id)
    if not session:
        ensure_default_sessions(user_id)
        session = get_session(user_id, None)
    if not session:
        raise HTTPException(status_code=500, detail="Could not create a chat session.")

    emotion_result = detect_emotion_ai(user_message)
    mode = choose_mode(
        emotion=emotion_result.emotion,
        intensity=emotion_result.intensity,
        risk=emotion_result.risk,
    )

    save_message(
        user_id=user_id,
        session_id=session["id"],
        role="user",
        content=user_message,
        emotion=emotion_result.emotion,
        mode=mode,
    )

    new_memories = extract_and_save_memories(
        user_id=user_id,
        user_message=user_message,
        emotion=emotion_result.emotion,
        session_id=session["id"],
    )
    summary_before = get_session_summary(user_id, session["id"])
    summary = maybe_update_session_summary(user_id, session["id"])
    summary_compacted = bool(
        summary
        and (
            not summary_before
            or summary.get("updated_at") != summary_before.get("updated_at")
        )
    )

    recent_context = get_recent_context(user_id, session_id=session["id"], limit=10)
    long_term_memories = get_relevant_memories(
        user_id=user_id,
        query=user_message,
        limit=6,
        session_id=session["id"],
    )
    system_prompt = build_system_prompt(
        emotion=emotion_result.emotion,
        intensity=emotion_result.intensity,
        mode=mode,
        session_summary=(summary or {}).get("summary", ""),
        recent_context=recent_context,
        long_term_memories=long_term_memories,
        session=session,
    )

    try:
        reply = generate_reply(system_prompt, user_message)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Make sure Ollama is running and qwen3:8b is installed.",
        ) from exc

    save_message(
        user_id=user_id,
        session_id=session["id"],
        role="assistant",
        content=reply,
        emotion=emotion_result.emotion,
        mode=mode,
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
        new_memories=new_memories,
        memories_used=len(long_term_memories),
        summary_used=bool((summary or {}).get("summary")),
        summary_compacted=summary_compacted,
    )


@app.get("/api/sessions")
def sessions(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    return {"sessions": list_sessions(str(user["id"]))}


@app.post("/api/sessions")
def add_session(request: SessionRequest, user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    if not request.title.strip():
        raise HTTPException(status_code=400, detail="Session title cannot be empty")
    user_id = str(user["id"])
    session = create_session(
        user_id=user_id,
        title=request.title,
        description=request.description,
        mode_hint=request.mode_hint,
    )
    return {"session": session, "sessions": list_sessions(user_id)}


@app.put("/api/sessions/{session_id}")
def edit_session(
    session_id: str,
    request: SessionRequest,
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    user_id = str(user["id"])
    session = update_session(
        user_id=user_id,
        session_id=session_id,
        title=request.title,
        description=request.description,
        mode_hint=request.mode_hint,
    )
    return {"session": session, "sessions": list_sessions(user_id)}


@app.get("/api/history")
def history(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=60, ge=1, le=120),
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    return {"messages": list_conversation(str(user["id"]), session_id=session_id, limit=limit)}


@app.get("/api/summary")
def summary(
    session_id: str | None = Query(default=None),
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    user_id = str(user["id"])
    session = get_session(user_id, session_id)
    if not session:
        return empty_summary(user_id, "")
    return get_session_summary(user_id, session["id"]) or empty_summary(user_id, session["id"])


@app.post("/api/summary/rebuild")
def rebuild_summary(
    request: SummaryRequest,
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    user_id = str(user["id"])
    session = get_session(user_id, request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return update_session_summary(user_id, session["id"], force=True)


@app.get("/api/memories")
def memories(
    q: str = Query(default=""),
    session_id: str | None = Query(default=None),
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    user_id = str(user["id"])
    return {"memories": list_memories(user_id, query=q, session_id=session_id)}


@app.post("/api/memories")
def add_memory(request: MemoryRequest, user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    user_id = str(user["id"])
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory cannot be empty")

    saved = save_memory(
        user_id=user_id,
        title=request.title,
        content=content,
        emotion=request.emotion,
        importance=request.importance,
        pinned=request.pinned,
        source="manual",
        session_id=request.session_id,
    )
    return {"saved": saved, "memories": list_memories(user_id)}


@app.patch("/api/memories/{memory_id}/pin")
def set_memory_pin(
    memory_id: int,
    request: PinRequest,
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    user_id = str(user["id"])
    updated = pin_memory(user_id, memory_id, request.pinned)
    return {"updated": updated, "memories": list_memories(user_id)}


@app.delete("/api/memories/{memory_id}")
def remove_memory(memory_id: int, user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    user_id = str(user["id"])
    deleted = delete_memory(user_id, memory_id)
    return {"deleted": deleted, "memories": list_memories(user_id)}


@app.get("/api/history/search")
def search_history(
    q: str = Query(default=""),
    session_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=60),
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    query = q.strip()
    if not query:
        return {"results": []}
    return {
        "results": search_conversation(
            user_id=str(user["id"]),
            query=query,
            session_id=session_id,
            limit=limit,
        ),
        "note": "Search only checks the recent retained chat text. Long-term memory is stored as summaries.",
    }


@app.patch("/api/history/{message_id}/pin")
def set_message_pin(
    message_id: int,
    request: PinRequest,
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    updated = pin_message(str(user["id"]), message_id, request.pinned)
    return {"updated": updated}


@app.delete("/api/history/{message_id}")
def remove_message(message_id: int, user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    deleted = delete_message(str(user["id"]), message_id)
    return {"deleted": deleted}


@app.post("/api/data/clear")
def clear_saved_text(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    user_id = str(user["id"])
    clear_user_saved_text(user_id)
    return {"ok": True, "sessions": list_sessions(user_id)}


@app.post("/api/embeddings/backfill")
def rebuild_embeddings() -> dict[str, Any]:
    return {
        "embedded": backfill_message_embeddings(),
        "note": "Semantic archive embeddings are disabled in summary-first mode.",
    }


@app.get("/api/export")
def export_history(user: dict[str, Any] = Depends(require_user)) -> JSONResponse:
    payload = export_user_data(str(user["id"]))
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": "attachment; filename=ponponchan-memory.json"},
    )


@app.get("/api/character")
def character(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    del user
    return {"character": load_character()}


@app.put("/api/character")
def update_character(
    request: CharacterRequest,
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    del user
    return {"character": save_character(request.character)}


def empty_summary(user_id: str, session_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "summary": "",
        "summarized_until_message_id": 0,
        "updated_at": "",
    }


def maybe_update_session_summary(user_id: str, session_id: str) -> dict[str, Any]:
    current = get_session_summary(user_id, session_id) or empty_summary(user_id, session_id)
    pending = get_unsummarized_messages(user_id=user_id, session_id=session_id, limit=24)
    if len(pending) < 12:
        return current
    return update_session_summary(user_id, session_id)


def update_session_summary(user_id: str, session_id: str, force: bool = False) -> dict[str, Any]:
    current = get_session_summary(user_id, session_id) or empty_summary(user_id, session_id)
    pending = (
        list_conversation(user_id=user_id, session_id=session_id, limit=120)
        if force
        else get_unsummarized_messages(user_id=user_id, session_id=session_id, limit=80)
    )
    if not pending:
        return current

    previous_summary = "" if force else current["summary"]
    transcript = "\n".join(f"{item['role']}: {item['content']}" for item in pending)
    prompt = f"""
Update the rolling memory summary for a local emotional companion.

Previous summary:
{previous_summary if previous_summary else "No previous summary."}

New conversation:
{transcript}

Write a compact durable summary that preserves:
- important facts about the user
- emotional patterns and relationship context
- unresolved problems, goals, and promises
- names, preferences, boundaries, and recurring topics
- what the assistant should remember next time

Keep it under 650 words. Do not invent facts. Write in clear bullet points.
""".strip()

    try:
        new_summary = ollama_chat(
            messages=[
                {
                    "role": "system",
                    "content": "You compress conversation into durable memory for a local emotional AI.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=0.9,
            timeout=180,
            num_predict=700,
        )
    except RuntimeError:
        new_summary = build_fallback_summary(previous_summary, pending)

    last_id = max(int(item["id"]) for item in pending)
    return save_session_summary(user_id, session_id, new_summary, last_id)


def build_fallback_summary(previous_summary: str, messages: list[dict[str, Any]]) -> str:
    recent = "\n".join(
        f"- {item['role']}: {item['content'][:220]}"
        for item in messages[-16:]
    )
    if previous_summary:
        return f"{previous_summary}\n\nRecent compacted context:\n{recent}".strip()
    return f"Recent compacted context:\n{recent}".strip()
