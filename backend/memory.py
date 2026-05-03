import sqlite3
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "emotional_ai.sqlite"


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT,
                mode TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        db.commit()


def save_message(
    user_id: str,
    role: str,
    content: str,
    emotion: str | None = None,
    mode: str | None = None
) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """
            INSERT INTO messages (user_id, role, content, emotion, mode)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, role, content, emotion, mode)
        )
        db.commit()


def get_recent_context(user_id: str, limit: int = 8) -> str:
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            """
            SELECT role, content
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()

    rows.reverse()
    return "\n".join(f"{role}: {content}" for role, content in rows)
