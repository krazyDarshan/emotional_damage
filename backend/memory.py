import json
import math
import re
import sqlite3
from pathlib import Path

from ollama_client import EMBED_MODEL, get_embedding


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "emotional_ai.sqlite"

DEFAULT_SESSIONS = [
    ("general", "General", "Everyday conversation and emotional check-ins.", "balanced companion mode"),
    ("relationship-talk", "Relationship Talk", "Love, conflict, attachment, communication, and dating.", "relationship support mode"),
    ("study-stress", "Study Stress", "Study pressure, exams, motivation, and discipline.", "study support mode"),
    ("personal-goals", "Personal Goals", "Habits, growth, identity, plans, and self-understanding.", "personal growth mode"),
    ("roleplay-mode", "Roleplay Mode", "Character mimicry, scenes, and fictional companion interactions.", "roleplay mode")
]

STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your", "are",
    "was", "were", "have", "has", "had", "but", "not", "from", "about",
    "what", "when", "where", "why", "how", "feel", "feels", "feeling",
    "remember", "forgot", "forget", "conversation", "chat"
}


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT DEFAULT 'general',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT,
                mode TEXT,
                pinned INTEGER DEFAULT 0,
                deleted INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                source_message TEXT,
                importance INTEGER DEFAULT 5,
                pinned INTEGER DEFAULT 0,
                deleted INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                mode_hint TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, user_id)
            )
            """
        )

        add_column_if_missing(db, "messages", "session_id", "TEXT DEFAULT 'general'")
        add_column_if_missing(db, "messages", "pinned", "INTEGER DEFAULT 0")
        add_column_if_missing(db, "messages", "deleted", "INTEGER DEFAULT 0")
        add_column_if_missing(db, "memories", "deleted", "INTEGER DEFAULT 0")

        db.execute("UPDATE messages SET session_id = 'general' WHERE session_id IS NULL OR session_id = ''")
        db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_user_content
            ON memories (user_id, content)
            """
        )
        db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_user_session_id
            ON messages (user_id, session_id, id)
            """
        )
        db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_deleted
            ON messages (user_id, deleted, id)
            """
        )
        ensure_default_sessions_for_connection(db, "local-user")
        db.commit()


def add_column_if_missing(db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def ensure_default_sessions_for_connection(db: sqlite3.Connection, user_id: str) -> None:
    for session_id, title, description, mode_hint in DEFAULT_SESSIONS:
        db.execute(
            """
            INSERT OR IGNORE INTO chat_sessions (id, user_id, title, description, mode_hint)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_id, title, description, mode_hint)
        )


def ensure_default_sessions(user_id: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        ensure_default_sessions_for_connection(db, user_id)
        db.commit()


def list_sessions(user_id: str) -> list[dict]:
    ensure_default_sessions(user_id)
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            """
            SELECT s.id, s.title, s.description, s.mode_hint, s.created_at, s.updated_at,
                   COUNT(m.id) AS message_count
            FROM chat_sessions s
            LEFT JOIN messages m
              ON m.user_id = s.user_id
             AND m.session_id = s.id
             AND m.deleted = 0
            WHERE s.user_id = ?
            GROUP BY s.id, s.title, s.description, s.mode_hint, s.created_at, s.updated_at
            ORDER BY CASE s.id
                WHEN 'general' THEN 0
                WHEN 'relationship-talk' THEN 1
                WHEN 'study-stress' THEN 2
                WHEN 'personal-goals' THEN 3
                WHEN 'roleplay-mode' THEN 4
                ELSE 5
            END, s.updated_at DESC
            """,
            (user_id,)
        ).fetchall()

    return [
        {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "mode_hint": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "message_count": row[6]
        }
        for row in rows
    ]


def get_session(user_id: str, session_id: str) -> dict:
    ensure_default_sessions(user_id)
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute(
            """
            SELECT id, title, description, mode_hint, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ? AND id = ?
            """,
            (user_id, session_id)
        ).fetchone()

    if not row:
        return create_session(user_id, "General", session_id="general")

    return {
        "id": row[0],
        "title": row[1],
        "description": row[2],
        "mode_hint": row[3],
        "created_at": row[4],
        "updated_at": row[5]
    }


def create_session(
    user_id: str,
    title: str,
    description: str = "",
    mode_hint: str = "",
    session_id: str | None = None
) -> dict:
    init_db()
    clean_title = clean_memory(title) or "New Chat"
    new_id = session_id or slugify(clean_title)
    with sqlite3.connect(DB_PATH) as db:
        original_id = new_id
        suffix = 2
        while db.execute(
            "SELECT 1 FROM chat_sessions WHERE user_id = ? AND id = ?",
            (user_id, new_id)
        ).fetchone():
            if session_id:
                break
            new_id = f"{original_id}-{suffix}"
            suffix += 1

        db.execute(
            """
            INSERT OR REPLACE INTO chat_sessions (id, user_id, title, description, mode_hint, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (new_id, user_id, clean_title, description, mode_hint)
        )
        db.commit()

    return get_session(user_id, new_id)


def update_session(user_id: str, session_id: str, title: str, description: str, mode_hint: str) -> dict:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """
            UPDATE chat_sessions
            SET title = ?, description = ?, mode_hint = ?, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ? AND id = ?
            """,
            (clean_memory(title) or "Chat", description, mode_hint, user_id, session_id)
        )
        db.commit()
    return get_session(user_id, session_id)


def save_message(
    user_id: str,
    role: str,
    content: str,
    emotion: str | None = None,
    mode: str | None = None,
    session_id: str = "general",
    pinned: bool = False,
    embed: bool = True
) -> int:
    init_db()
    ensure_default_sessions(user_id)
    with sqlite3.connect(DB_PATH) as db:
        cursor = db.execute(
            """
            INSERT INTO messages (user_id, session_id, role, content, emotion, mode, pinned)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, session_id, role, content, emotion, mode, int(pinned))
        )
        db.execute(
            """
            UPDATE chat_sessions
            SET updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ? AND id = ?
            """,
            (user_id, session_id)
        )
        db.commit()
        message_id = int(cursor.lastrowid)

    if embed:
        try:
            store_message_embedding(message_id, content)
        except Exception:
            pass

    return message_id


def store_message_embedding(message_id: int, content: str) -> bool:
    vector = get_embedding(content)
    if not vector:
        return False

    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """
            INSERT OR REPLACE INTO message_embeddings (message_id, model, vector_json)
            VALUES (?, ?, ?)
            """,
            (message_id, EMBED_MODEL, json.dumps(vector, separators=(",", ":")))
        )
        db.commit()
    return True


def backfill_message_embeddings(user_id: str, session_id: str | None = None, limit: int = 80) -> int:
    init_db()
    clauses = ["m.user_id = ?", "m.deleted = 0", "e.message_id IS NULL"]
    params: list[object] = [user_id]
    if session_id and session_id != "all":
        clauses.append("m.session_id = ?")
        params.append(session_id)

    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            f"""
            SELECT m.id, m.content
            FROM messages m
            LEFT JOIN message_embeddings e ON e.message_id = m.id
            WHERE {' AND '.join(clauses)}
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (*params, limit)
        ).fetchall()

    count = 0
    for message_id, content in rows:
        try:
            if store_message_embedding(int(message_id), content):
                count += 1
        except Exception:
            break
    return count


def get_recent_context(user_id: str, session_id: str = "general", limit: int = 10) -> str:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            """
            SELECT role, content
            FROM messages
            WHERE user_id = ? AND session_id = ? AND deleted = 0
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, session_id, limit)
        ).fetchall()

    rows.reverse()
    return "\n".join(f"{role}: {content}" for role, content in rows)


def get_relevant_conversation(
    user_id: str,
    query: str,
    session_id: str = "general",
    limit: int = 12,
    exclude_recent: int = 8
) -> str:
    results = search_conversation(
        user_id=user_id,
        query=query,
        session_id=session_id,
        limit=limit,
        exclude_recent=exclude_recent
    )
    return "\n".join(
        f"[{item['created_at']}] {item['role']}: {item['content']}"
        for item in sorted(results, key=lambda row: row["id"])
    )


def search_conversation(
    user_id: str,
    query: str,
    session_id: str = "all",
    limit: int = 30,
    exclude_recent: int = 0
) -> list[dict]:
    init_db()
    backfill_message_embeddings(user_id, None if session_id == "all" else session_id, limit=60)

    query_terms = tokenize(query)
    recall_request = asks_about_past(query)
    try:
        query_vector = get_embedding(query)
    except Exception:
        query_vector = []

    clauses = ["m.user_id = ?", "m.deleted = 0"]
    params: list[object] = [user_id]
    if session_id and session_id != "all":
        clauses.append("(m.session_id = ? OR m.pinned = 1)")
        params.append(session_id)

    with sqlite3.connect(DB_PATH) as db:
        recent_ids = set()
        if exclude_recent and session_id and session_id != "all":
            recent_ids = {
                row[0]
                for row in db.execute(
                    """
                    SELECT id
                    FROM messages
                    WHERE user_id = ? AND session_id = ? AND deleted = 0
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, session_id, exclude_recent)
                ).fetchall()
            }

        rows = db.execute(
            f"""
            SELECT m.id, m.session_id, m.role, m.content, m.emotion, m.mode,
                   m.pinned, m.created_at, e.vector_json
            FROM messages m
            LEFT JOIN message_embeddings e ON e.message_id = m.id
            WHERE {' AND '.join(clauses)}
            ORDER BY m.id DESC
            LIMIT 2000
            """,
            params
        ).fetchall()

    scored_rows = []
    for row in rows:
        message_id, row_session_id, role, content, emotion, mode, pinned, created_at, vector_json = row
        if message_id in recent_ids:
            continue

        keyword_overlap = len(query_terms & tokenize(content))
        semantic_score = 0.0
        if query_vector and vector_json:
            try:
                semantic_score = cosine_similarity(query_vector, json.loads(vector_json))
            except Exception:
                semantic_score = 0.0

        score = semantic_score * 100
        score += keyword_overlap * 6
        score += 10 if pinned else 0
        score += 4 if row_session_id == session_id else 0
        score += 3 if role == "user" else 0
        score += 2 if recall_request else 0

        if score > 2 or pinned:
            scored_rows.append(
                {
                    "id": message_id,
                    "session_id": row_session_id,
                    "role": role,
                    "content": content,
                    "emotion": emotion,
                    "mode": mode,
                    "pinned": bool(pinned),
                    "created_at": created_at,
                    "score": round(score, 3),
                    "semantic_score": round(semantic_score, 4)
                }
            )

    scored_rows.sort(key=lambda item: item["score"], reverse=True)
    return scored_rows[:limit]


def list_conversation(
    user_id: str,
    session_id: str = "all",
    limit: int = 80,
    offset: int = 0
) -> dict:
    init_db()
    clauses = ["user_id = ?", "deleted = 0"]
    params: list[object] = [user_id]
    if session_id and session_id != "all":
        clauses.append("session_id = ?")
        params.append(session_id)

    with sqlite3.connect(DB_PATH) as db:
        total = db.execute(
            f"SELECT COUNT(*) FROM messages WHERE {' AND '.join(clauses)}",
            params
        ).fetchone()[0]
        rows = db.execute(
            f"""
            SELECT id, session_id, role, content, emotion, mode, pinned, created_at
            FROM messages
            WHERE {' AND '.join(clauses)}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset)
        ).fetchall()

    rows.reverse()
    return {
        "total": total,
        "messages": [
            {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "emotion": row[4],
                "mode": row[5],
                "pinned": bool(row[6]),
                "created_at": row[7]
            }
            for row in rows
        ]
    }


def pin_message(user_id: str, message_id: int, pinned: bool) -> bool:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        cursor = db.execute(
            """
            UPDATE messages
            SET pinned = ?
            WHERE user_id = ? AND id = ? AND deleted = 0
            """,
            (int(pinned), user_id, message_id)
        )
        db.commit()
        return cursor.rowcount > 0


def delete_message(user_id: str, message_id: int) -> bool:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        db.execute("DELETE FROM message_embeddings WHERE message_id = ?", (message_id,))
        cursor = db.execute(
            "DELETE FROM messages WHERE user_id = ? AND id = ?",
            (user_id, message_id)
        )
        db.commit()
        return cursor.rowcount > 0


def save_memory(
    user_id: str,
    content: str,
    source_message: str | None = None,
    importance: int = 5,
    pinned: bool = False
) -> bool:
    init_db()
    cleaned = clean_memory(content)
    if len(cleaned) < 8:
        return False

    with sqlite3.connect(DB_PATH) as db:
        try:
            db.execute(
                """
                INSERT INTO memories (user_id, content, source_message, importance, pinned, deleted)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (user_id, cleaned, source_message, importance, int(pinned))
            )
            db.commit()
            return True
        except sqlite3.IntegrityError:
            db.execute(
                """
                UPDATE memories
                SET deleted = 0, importance = MAX(importance, ?), pinned = MAX(pinned, ?)
                WHERE user_id = ? AND content = ?
                """,
                (importance, int(pinned), user_id, cleaned)
            )
            db.commit()
            return False


def extract_and_save_memories(user_id: str, text: str) -> list[str]:
    memories = extract_memory_candidates(text)
    saved = []
    for content, importance, pinned in memories:
        if save_memory(
            user_id=user_id,
            content=content,
            source_message=text,
            importance=importance,
            pinned=pinned
        ):
            saved.append(clean_memory(content))
    return saved


def get_relevant_memories(user_id: str, query: str, limit: int = 8) -> str:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            """
            SELECT id, content, importance, pinned
            FROM memories
            WHERE user_id = ? AND deleted = 0
            ORDER BY pinned DESC, importance DESC, id DESC
            LIMIT 250
            """,
            (user_id,)
        ).fetchall()

    if not rows:
        return ""

    query_terms = tokenize(query)
    scored_rows = []
    for memory_id, content, importance, pinned in rows:
        memory_terms = tokenize(content)
        overlap = len(query_terms & memory_terms)
        score = overlap * 4 + int(importance) + int(pinned) * 8
        scored_rows.append((score, memory_id, content))

    scored_rows.sort(reverse=True)
    selected = scored_rows[:limit]
    mark_memories_used([memory_id for _, memory_id, _ in selected])
    return "\n".join(f"- {content}" for _, _, content in selected)


def list_memories(user_id: str, limit: int = 80) -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            """
            SELECT id, content, importance, pinned, created_at
            FROM memories
            WHERE user_id = ? AND deleted = 0
            ORDER BY pinned DESC, importance DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()

    return [
        {
            "id": row[0],
            "content": row[1],
            "importance": row[2],
            "pinned": bool(row[3]),
            "created_at": row[4]
        }
        for row in rows
    ]


def pin_memory(user_id: str, memory_id: int, pinned: bool) -> bool:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        cursor = db.execute(
            """
            UPDATE memories
            SET pinned = ?
            WHERE user_id = ? AND id = ? AND deleted = 0
            """,
            (int(pinned), user_id, memory_id)
        )
        db.commit()
        return cursor.rowcount > 0


def delete_memory(user_id: str, memory_id: int) -> bool:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        cursor = db.execute(
            "DELETE FROM memories WHERE user_id = ? AND id = ?",
            (user_id, memory_id)
        )
        db.commit()
        return cursor.rowcount > 0


def mark_memories_used(memory_ids: list[int]) -> None:
    if not memory_ids:
        return

    init_db()
    placeholders = ",".join("?" for _ in memory_ids)
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            f"UPDATE memories SET last_used_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
            memory_ids
        )
        db.commit()


def export_user_data(user_id: str) -> dict:
    init_db()
    return {
        "user_id": user_id,
        "sessions": list_sessions(user_id),
        "memories": list_memories(user_id, limit=5000),
        "messages": list_conversation(user_id, session_id="all", limit=10000)["messages"]
    }


def extract_memory_candidates(text: str) -> list[tuple[str, int, bool]]:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return []

    lowered = normalized.lower()
    candidates: list[tuple[str, int, bool]] = []

    explicit_patterns = [
        r"(?:please\s+)?remember(?:\s+that)?\s+(.+)",
        r"don't forget(?:\s+that)?\s+(.+)",
        r"do not forget(?:\s+that)?\s+(.+)",
        r"keep in mind(?:\s+that)?\s+(.+)"
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            candidates.append((sentence_case(match.group(1)), 10, True))

    if candidates:
        return dedupe_candidates(candidates)

    fact_patterns = [
        (r"\bmy name is\s+([^,.!?]+)", "The user's name is {value}.", 10),
        (r"\bcall me\s+([^,.!?]+)", "The user likes to be called {value}.", 9),
        (r"\bi am from\s+([^,.!?]+)", "The user is from {value}.", 8),
        (r"\bi live in\s+([^,.!?]+)", "The user lives in {value}.", 8),
        (r"\bi work at\s+([^,.!?]+)", "The user works at {value}.", 8),
        (r"\bi study at\s+([^,.!?]+)", "The user studies at {value}.", 8),
        (r"\bmy (?:girlfriend|boyfriend|partner|wife|husband|crush) is\s+([^,.!?]+)", "The user's relationship detail: {value}.", 8),
        (r"\bi (?:really\s+)?like\s+([^,.!?]+)", "The user likes {value}.", 6),
        (r"\bi (?:really\s+)?love\s+([^,.!?]+)", "The user loves {value}.", 7),
        (r"\bi hate\s+([^,.!?]+)", "The user dislikes {value}.", 7),
        (r"\bi prefer\s+([^,.!?]+)", "The user prefers {value}.", 6),
        (r"\bi feel\s+([^,.!?]+)\s+when\s+([^,.!?]+)", "The user often feels {value} when {extra}.", 7),
        (r"\bi get\s+([^,.!?]+)\s+when\s+([^,.!?]+)", "The user often gets {value} when {extra}.", 7)
    ]

    for pattern, template, importance in fact_patterns:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            value = clean_memory(match.group(1))
            if len(value) < 2:
                continue
            if "{extra}" in template:
                extra = clean_memory(match.group(2))
                content = template.format(value=value, extra=extra)
            else:
                content = template.format(value=value)
            candidates.append((content, importance, False))

    return dedupe_candidates(candidates)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def clean_memory(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip(" .,:;!?\"'"))
    return cleaned[:280]


def sentence_case(text: str) -> str:
    cleaned = clean_memory(text)
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def tokenize(text: str) -> set[str]:
    terms = re.findall(r"[a-zA-Z0-9']{3,}", text.lower())
    return {term for term in terms if term not in STOP_WORDS}


def asks_about_past(text: str) -> bool:
    lowered = text.lower()
    phrases = [
        "remember",
        "forgot",
        "forget",
        "earlier",
        "before",
        "last time",
        "previous",
        "old chat",
        "conversation",
        "we talked",
        "what did i say",
        "what did we discuss"
    ]
    return any(phrase in lowered for phrase in phrases)


def dedupe_candidates(candidates: list[tuple[str, int, bool]]) -> list[tuple[str, int, bool]]:
    seen = set()
    result = []
    for content, importance, pinned in candidates:
        cleaned = clean_memory(content)
        key = cleaned.lower()
        if key and key not in seen:
            seen.add(key)
            result.append((cleaned, importance, pinned))
    return result


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "chat"
