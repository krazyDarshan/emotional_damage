from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
DB_PATH = DATA_DIR / "emotional_ai.sqlite"
RAW_MESSAGE_KEEP_LIMIT = 18

DEFAULT_SESSIONS = [
    ("relationship talk", "Heart matters, attachment, conflict, repair."),
    ("study stress", "Study pressure, focus, anxiety, planning."),
    ("personal goals", "Habits, confidence, identity, future plans."),
    ("roleplay mode", "Character-focused conversations and playful scenes."),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    if column not in _column_names(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                mode_hint TEXT DEFAULT 'supportive',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT DEFAULT 'neutral',
                mode TEXT DEFAULT 'supportive',
                pinned INTEGER DEFAULT 0,
                deleted INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT DEFAULT 'neutral',
                importance INTEGER DEFAULT 3,
                pinned INTEGER DEFAULT 0,
                source TEXT DEFAULT 'extracted',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                summarized_until_message_id INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id)
            );
            """
        )

        for table, column, definition in [
            ("chat_sessions", "archived", "INTEGER DEFAULT 0"),
            ("chat_sessions", "user_id", "TEXT DEFAULT 'local-user'"),
            ("chat_sessions", "title", "TEXT DEFAULT 'conversation'"),
            ("chat_sessions", "mode_hint", "TEXT DEFAULT 'supportive'"),
            ("chat_sessions", "description", "TEXT DEFAULT ''"),
            ("chat_sessions", "created_at", "TEXT"),
            ("chat_sessions", "updated_at", "TEXT"),
            ("messages", "user_id", "TEXT DEFAULT 'local-user'"),
            ("messages", "session_id", "TEXT DEFAULT 'general'"),
            ("messages", "role", "TEXT DEFAULT 'user'"),
            ("messages", "content", "TEXT DEFAULT ''"),
            ("messages", "emotion", "TEXT DEFAULT 'neutral'"),
            ("messages", "mode", "TEXT DEFAULT 'supportive'"),
            ("messages", "pinned", "INTEGER DEFAULT 0"),
            ("messages", "deleted", "INTEGER DEFAULT 0"),
            ("messages", "created_at", "TEXT"),
            ("memories", "title", "TEXT DEFAULT 'memory'"),
            ("memories", "content", "TEXT DEFAULT ''"),
            ("memories", "session_id", "TEXT"),
            ("memories", "emotion", "TEXT DEFAULT 'neutral'"),
            ("memories", "importance", "INTEGER DEFAULT 3"),
            ("memories", "pinned", "INTEGER DEFAULT 0"),
            ("memories", "source", "TEXT DEFAULT 'extracted'"),
            ("memories", "created_at", "TEXT"),
            ("memories", "updated_at", "TEXT"),
            ("session_summaries", "user_id", "TEXT DEFAULT 'local-user'"),
            ("session_summaries", "session_id", "TEXT DEFAULT 'general'"),
            ("session_summaries", "summary", "TEXT DEFAULT ''"),
            ("session_summaries", "summarized_until_message_id", "INTEGER DEFAULT 0"),
            ("session_summaries", "updated_at", "TEXT"),
        ]:
            if _table_exists(conn, table):
                _add_column(conn, table, column, definition)

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id, archived, updated_at);
            CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id, id);
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, pinned, updated_at);
            CREATE INDEX IF NOT EXISTS idx_summary_user_session ON session_summaries(user_id, session_id);
            """
        )


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 180_000)
    return digest.hex(), salt


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_user(username: str, password: str, display_name: str | None = None) -> dict[str, Any]:
    init_db()
    username = username.strip()
    display_name = (display_name or username).strip() or username
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    password_hash, salt = _hash_password(password)
    created_at = now_iso()
    try:
        with _connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, display_name, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, display_name, password_hash, salt, created_at),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username is already registered.") from exc

    ensure_default_sessions(str(user_id))
    return {"id": user_id, "username": username, "display_name": display_name, "created_at": created_at}


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    if not row:
        return None

    candidate, _ = _hash_password(password, row["salt"])
    if not hmac.compare_digest(candidate, row["password_hash"]):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
    }


def create_auth_session(user_id: int, days: int = 30) -> str:
    init_db()
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    created_at = now_iso()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token_hash, user_id, created_at, expires_at),
        )
    return token


def get_user_by_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    init_db()
    token_hash = _hash_token(token)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.username, users.display_name, users.created_at, auth_sessions.expires_at
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if not row:
            return None

        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at < datetime.now(timezone.utc):
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
            return None

    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
    }


def delete_auth_session(token: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_hash_token(token),))


def ensure_default_sessions(user_id: str) -> None:
    init_db()
    existing = list_sessions(user_id)
    if existing:
        return

    for title, description in DEFAULT_SESSIONS:
        create_session(user_id, title, description)


def list_sessions(user_id: str) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT chat_sessions.*,
                   COALESCE(message_counts.message_count, 0) AS message_count,
                   session_summaries.summary AS summary
            FROM chat_sessions
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS message_count
                FROM messages
                WHERE user_id = ? AND deleted = 0
                GROUP BY session_id
            ) AS message_counts ON message_counts.session_id = chat_sessions.id
            LEFT JOIN session_summaries
              ON session_summaries.user_id = chat_sessions.user_id
             AND session_summaries.session_id = chat_sessions.id
            WHERE chat_sessions.user_id = ? AND COALESCE(chat_sessions.archived, 0) = 0
            ORDER BY datetime(chat_sessions.updated_at) DESC
            """,
            (user_id, user_id),
        ).fetchall()
    return [dict(row) for row in rows]


def create_session(user_id: str, title: str, description: str = "", mode_hint: str = "supportive") -> dict[str, Any]:
    init_db()
    session_id = secrets.token_hex(8)
    timestamp = now_iso()
    title = title.strip() or "new conversation"
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (id, user_id, title, description, mode_hint, created_at, updated_at, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (session_id, user_id, title, description.strip(), mode_hint, timestamp, timestamp),
        )
    return get_session(user_id, session_id) or {
        "id": session_id,
        "user_id": user_id,
        "title": title,
        "description": description,
        "mode_hint": mode_hint,
        "created_at": timestamp,
        "updated_at": timestamp,
        "archived": 0,
    }


def update_session(
    user_id: str,
    session_id: str,
    title: str | None = None,
    description: str | None = None,
    mode_hint: str | None = None,
) -> dict[str, Any] | None:
    init_db()
    updates: list[str] = ["updated_at = ?"]
    values: list[Any] = [now_iso()]
    if title is not None:
        updates.append("title = ?")
        values.append(title.strip() or "conversation")
    if description is not None:
        updates.append("description = ?")
        values.append(description.strip())
    if mode_hint is not None:
        updates.append("mode_hint = ?")
        values.append(mode_hint.strip() or "supportive")
    values.extend([user_id, session_id])
    with _connect() as conn:
        conn.execute(
            f"UPDATE chat_sessions SET {', '.join(updates)} WHERE user_id = ? AND id = ?",
            values,
        )
    return get_session(user_id, session_id)


def get_session(user_id: str, session_id: str | None) -> dict[str, Any] | None:
    init_db()
    if not session_id:
        ensure_default_sessions(user_id)
        sessions = list_sessions(user_id)
        return sessions[0] if sessions else None

    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE user_id = ? AND id = ? AND COALESCE(archived, 0) = 0",
            (user_id, session_id),
        ).fetchone()
    return _row_to_dict(row)


def save_message(
    user_id: str,
    role: str,
    content: str,
    emotion: str = "neutral",
    mode: str = "supportive",
    session_id: str | None = None,
    pinned: bool = False,
    embedding: list[float] | None = None,
) -> int:
    del embedding
    init_db()
    session = get_session(user_id, session_id)
    if not session:
        session = create_session(user_id, "general")
    timestamp = now_iso()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (user_id, session_id, role, content, emotion, mode, pinned, deleted, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (user_id, session["id"], role, content, emotion, mode, 1 if pinned else 0, timestamp),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE user_id = ? AND id = ?",
            (timestamp, user_id, session["id"]),
        )
        return int(cursor.lastrowid)


def list_conversation(user_id: str, session_id: str | None = None, limit: int = 80) -> list[dict[str, Any]]:
    init_db()
    session = get_session(user_id, session_id)
    if not session:
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, emotion, mode, pinned, created_at
            FROM messages
            WHERE user_id = ? AND session_id = ? AND deleted = 0
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, session["id"], limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def get_recent_context(user_id: str, session_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    return list_conversation(user_id, session_id, limit)


def get_unsummarized_messages(user_id: str, session_id: str, limit: int = 60) -> list[dict[str, Any]]:
    init_db()
    summary = get_session_summary(user_id, session_id)
    summarized_until = int(summary.get("summarized_until_message_id") or 0) if summary else 0
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, emotion, mode, created_at
            FROM messages
            WHERE user_id = ?
              AND session_id = ?
              AND deleted = 0
              AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (user_id, session_id, summarized_until, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_session_summary(user_id: str, session_id: str | None) -> dict[str, Any] | None:
    init_db()
    session = get_session(user_id, session_id)
    if not session:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT user_id, session_id, summary, summarized_until_message_id, updated_at
            FROM session_summaries
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session["id"]),
        ).fetchone()
    return _row_to_dict(row)


def save_session_summary(user_id: str, session_id: str, summary: str, summarized_until_message_id: int) -> dict[str, Any]:
    init_db()
    timestamp = now_iso()
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE session_summaries
            SET summary = ?, summarized_until_message_id = ?, updated_at = ?
            WHERE user_id = ? AND session_id = ?
            """,
            (summary.strip(), summarized_until_message_id, timestamp, user_id, session_id),
        )
        if cursor.rowcount == 0:
            conn.execute(
                """
                INSERT INTO session_summaries (user_id, session_id, summary, summarized_until_message_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, session_id, summary.strip(), summarized_until_message_id, timestamp),
            )
    prune_summarized_messages(user_id, session_id, summarized_until_message_id)
    return get_session_summary(user_id, session_id) or {}


def prune_summarized_messages(user_id: str, session_id: str, summarized_until_message_id: int, keep_recent: int = RAW_MESSAGE_KEEP_LIMIT) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            DELETE FROM messages
            WHERE user_id = ?
              AND session_id = ?
              AND pinned = 0
              AND id <= ?
              AND id NOT IN (
                  SELECT id
                  FROM messages
                  WHERE user_id = ? AND session_id = ? AND deleted = 0
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (user_id, session_id, summarized_until_message_id, user_id, session_id, keep_recent),
        )


def save_memory(
    user_id: str,
    title: str,
    content: str,
    emotion: str = "neutral",
    importance: int = 3,
    pinned: bool = False,
    source: str = "manual",
    session_id: str | None = None,
    embedding: list[float] | None = None,
) -> int:
    del embedding
    init_db()
    timestamp = now_iso()
    importance = max(1, min(5, int(importance or 3)))
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO memories (user_id, session_id, title, content, emotion, importance, pinned, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_id,
                title.strip()[:120] or "memory",
                content.strip(),
                emotion,
                importance,
                1 if pinned else 0,
                source,
                timestamp,
                timestamp,
            ),
        )
        return int(cursor.lastrowid)


def _memory_matches(memory: sqlite3.Row, query: str) -> bool:
    if not query:
        return True
    haystack = f"{memory['title']} {memory['content']} {memory['emotion']}".lower()
    terms = [term for term in query.lower().split() if len(term) > 2]
    return any(term in haystack for term in terms)


def list_memories(user_id: str, query: str = "", limit: int = 30, session_id: str | None = None) -> list[dict[str, Any]]:
    init_db()
    params: list[Any] = [user_id]
    session_filter = ""
    if session_id:
        session_filter = "AND (session_id = ? OR session_id IS NULL)"
        params.append(session_id)
    params.append(max(1, limit * 3))
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, user_id, session_id, title, content, emotion, importance, pinned, source, created_at, updated_at
            FROM memories
            WHERE user_id = ? {session_filter}
            ORDER BY pinned DESC, importance DESC, datetime(updated_at) DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    memories = [dict(row) for row in rows if _memory_matches(row, query)]
    return memories[:limit]


def get_relevant_memories(
    user_id: str,
    query: str,
    limit: int = 5,
    session_id: str | None = None,
    embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    del embedding
    return list_memories(user_id=user_id, query=query, limit=limit, session_id=session_id)


def pin_memory(user_id: str, memory_id: int, pinned: bool) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE memories SET pinned = ?, updated_at = ? WHERE user_id = ? AND id = ?",
            (1 if pinned else 0, now_iso(), user_id, memory_id),
        )
        row = conn.execute(
            "SELECT * FROM memories WHERE user_id = ? AND id = ?",
            (user_id, memory_id),
        ).fetchone()
    return _row_to_dict(row)


def delete_memory(user_id: str, memory_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM memories WHERE user_id = ? AND id = ?", (user_id, memory_id))
        return cursor.rowcount > 0


def extract_and_save_memories(
    user_id: str,
    user_message: str,
    emotion: str,
    session_id: str | None = None,
    embedding: list[float] | None = None,
) -> list[int]:
    del embedding
    text = user_message.strip()
    lower = text.lower()
    markers = (
        "remember",
        "my name is",
        "i am",
        "i'm",
        "i feel",
        "i like",
        "i love",
        "i hate",
        "i want",
        "i need",
        "my goal",
        "my dream",
        "important",
    )
    if len(text) < 12 or not any(marker in lower for marker in markers):
        return []

    title = "important personal note"
    if "my name is" in lower:
        title = "name"
    elif "my goal" in lower or "my dream" in lower:
        title = "personal goal"
    elif "i feel" in lower or emotion in {"sad", "anxious", "angry", "lonely"}:
        title = "emotional pattern"
    elif "i like" in lower or "i love" in lower or "i hate" in lower:
        title = "preference"

    memory_id = save_memory(
        user_id=user_id,
        title=title,
        content=text[:600],
        emotion=emotion,
        importance=4 if "important" in lower or "remember" in lower else 3,
        pinned=False,
        source="auto",
        session_id=session_id,
    )
    return [memory_id]


def pin_message(user_id: str, message_id: int, pinned: bool) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE messages SET pinned = ? WHERE user_id = ? AND id = ?",
            (1 if pinned else 0, user_id, message_id),
        )
        row = conn.execute(
            "SELECT id, role, content, emotion, mode, pinned, created_at FROM messages WHERE user_id = ? AND id = ?",
            (user_id, message_id),
        ).fetchone()
    return _row_to_dict(row)


def delete_message(user_id: str, message_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM messages WHERE user_id = ? AND id = ?", (user_id, message_id))
        return cursor.rowcount > 0


def search_conversation(
    user_id: str,
    query: str,
    limit: int = 12,
    session_id: str | None = None,
    embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    del embedding
    init_db()
    like = f"%{query.strip()}%"
    params: list[Any] = [user_id, like]
    session_filter = ""
    if session_id:
        session_filter = "AND session_id = ?"
        params.append(session_id)
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, role, content, emotion, mode, pinned, created_at
            FROM messages
            WHERE user_id = ? AND deleted = 0 AND content LIKE ? {session_filter}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_relevant_conversation(
    user_id: str,
    query: str,
    limit: int = 4,
    session_id: str | None = None,
    embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    del embedding
    # Raw archive recall has intentionally been reduced. This returns only recent retained text.
    return search_conversation(user_id, query, limit=limit, session_id=session_id)


def backfill_message_embeddings(user_id: str | None = None, limit: int = 200) -> dict[str, int]:
    del user_id, limit
    return {"updated": 0, "skipped": 0}


def clear_user_saved_text(user_id: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM session_summaries WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))
        if _table_exists(conn, "message_embeddings"):
            conn.execute(
                """
                DELETE FROM message_embeddings
                WHERE message_id NOT IN (SELECT id FROM messages)
                """
            )
    ensure_default_sessions(user_id)


def clear_all_saved_text(include_users: bool = False) -> None:
    init_db()
    with _connect() as conn:
        for table in ["message_embeddings", "messages", "memories", "session_summaries", "chat_sessions"]:
            if _table_exists(conn, table):
                conn.execute(f"DELETE FROM {table}")
        if include_users:
            conn.execute("DELETE FROM auth_sessions")
            conn.execute("DELETE FROM users")


def export_user_data(user_id: str) -> dict[str, Any]:
    init_db()
    return {
        "exported_at": now_iso(),
        "storage_policy": "Raw chat text is kept only as recent context. Long-term recall uses summaries and editable memories.",
        "sessions": list_sessions(user_id),
        "summaries": [
            summary
            for session in list_sessions(user_id)
            if (summary := get_session_summary(user_id, session["id"]))
        ],
        "recent_messages": [
            message
            for session in list_sessions(user_id)
            for message in list_conversation(user_id, session["id"], limit=RAW_MESSAGE_KEEP_LIMIT)
        ],
        "memories": list_memories(user_id, limit=500),
    }


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
