"""SQLite 儲存層（規格 §10）。單使用者，不用 ORM。

同步函式為主；呼叫端在 async context 用 asyncio.to_thread 包起來，
避免磁碟 IO 卡住 event loop。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    course_id   TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL,
    final_md    TEXT,
    handcopy_md TEXT,
    audio_path  TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    text        TEXT NOT NULL,
    avg_logprob REAL
);
CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id, start_s);

CREATE TABLE IF NOT EXISTS sections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    seq         INTEGER NOT NULL,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    title       TEXT NOT NULL,
    bullets     TEXT NOT NULL,
    user_note   TEXT
);
CREATE INDEX IF NOT EXISTS idx_sections_session ON sections(session_id, seq);
"""

_lock = threading.Lock()
_conn = None  # type: Optional[sqlite3.Connection]


def init(db_path=None) -> None:
    global _conn
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = sqlite3.connect(str(path), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(SCHEMA)
        # 舊資料庫補欄位，不動既有資料
        cur = _conn.cursor()
        _migrate(cur, "sessions", "handcopy_md", "TEXT")
        _migrate(cur, "sessions", "audio_path", "TEXT")
        # 段落筆記從「一串平鋪的要點」改成「導言＋分子題」，舊資料照讀，
        # list_sections 會把只有 bullets 的舊紀錄包成單一無標題的組
        _migrate(cur, "sections", "summary", "TEXT")
        _migrate(cur, "sections", "groups", "TEXT")
        cur.close()
        _conn.commit()


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    if _conn is None:
        raise RuntimeError("storage.init() 尚未呼叫")
    with _lock:
        cur = _conn.cursor()
        try:
            yield cur
            _conn.commit()
        except Exception:
            _conn.rollback()
            raise
        finally:
            cur.close()


# sessions ------------------------------------------------------------
def create_session(session_id: str, course_id: str, started_at: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO sessions (id, course_id, started_at) VALUES (?,?,?)",
            (session_id, course_id, started_at),
        )


def finish_session(session_id: str, ended_at: str, duration_s: float, final_md,
                   handcopy_md=None, audio_path=None) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE sessions SET ended_at=?, duration_s=?, final_md=?, "
            "handcopy_md=?, audio_path=? WHERE id=?",
            (ended_at, duration_s, final_md, handcopy_md, audio_path, session_id),
        )


def _migrate(cur, table, column, decl):
    """既有資料庫加欄位。SQLite 沒有 IF NOT EXISTS，只能先查再加。"""
    cur.execute("PRAGMA table_info(%s)" % table)
    if column not in {r[1] for r in cur.fetchall()}:
        cur.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, decl))


def get_session(session_id: str):
    with _cursor() as cur:
        cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def list_sessions(limit: int = 100):
    with _cursor() as cur:
        cur.execute(
            "SELECT id, course_id, started_at, ended_at, duration_s "
            "FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def delete_session(session_id: str):
    """刪除一堂課的所有資料。回傳該堂的錄音檔路徑（呼叫端負責刪檔）。"""
    with _cursor() as cur:
        cur.execute("SELECT audio_path FROM sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
        audio = row["audio_path"] if row else None
        cur.execute("DELETE FROM segments WHERE session_id=?", (session_id,))
        cur.execute("DELETE FROM sections WHERE session_id=?", (session_id,))
        cur.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return audio


def storage_usage():
    """回傳 (session 數, segment 數, 資料庫 bytes)。"""
    with _cursor() as cur:
        n_ses = cur.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        n_seg = cur.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
    from pathlib import Path
    db = Path(config.DB_PATH)
    return n_ses, n_seg, (db.stat().st_size if db.exists() else 0)


# segments ------------------------------------------------------------
def insert_segment(session_id: str, start_s: float, end_s: float, text: str,
                   avg_logprob) -> int:
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO segments (session_id, start_s, end_s, text, avg_logprob) "
            "VALUES (?,?,?,?,?)",
            (session_id, start_s, end_s, text, avg_logprob),
        )
        return int(cur.lastrowid)


def list_segments(session_id: str):
    with _cursor() as cur:
        cur.execute(
            "SELECT id, start_s, end_s, text, avg_logprob FROM segments "
            "WHERE session_id=? ORDER BY start_s",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def segments_between(session_id: str, start_s: float, end_s: float):
    with _cursor() as cur:
        cur.execute(
            "SELECT id, start_s, end_s, text FROM segments "
            "WHERE session_id=? AND start_s >= ? AND start_s < ? ORDER BY start_s",
            (session_id, start_s, end_s),
        )
        return [dict(r) for r in cur.fetchall()]


# sections ------------------------------------------------------------
def _dump(x):
    return json.dumps(x, ensure_ascii=False) if x else None


def insert_section(session_id: str, seq: int, start_s: float, end_s: float,
                   title: str, bullets, user_note, summary=None,
                   groups=None) -> int:
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO sections (session_id, seq, start_s, end_s, title, "
            "bullets, user_note, summary, groups) VALUES (?,?,?,?,?,?,?,?,?)",
            (session_id, seq, start_s, end_s, title,
             json.dumps(bullets, ensure_ascii=False), user_note,
             summary or None, _dump(groups)),
        )
        return int(cur.lastrowid)


def update_section(section_id: int, start_s: float, end_s: float, title: str,
                   bullets, user_note, summary=None, groups=None) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE sections SET start_s=?, end_s=?, title=?, bullets=?, "
            "user_note=?, summary=?, groups=? WHERE id=?",
            (start_s, end_s, title, json.dumps(bullets, ensure_ascii=False),
             user_note, summary or None, _dump(groups), section_id),
        )


def list_sections(session_id: str):
    with _cursor() as cur:
        cur.execute(
            "SELECT id, seq, start_s, end_s, title, bullets, user_note, "
            "summary, groups FROM sections WHERE session_id=? ORDER BY seq",
            (session_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        try:
            r["bullets"] = json.loads(r["bullets"])
        except (json.JSONDecodeError, TypeError):
            r["bullets"] = [str(r["bullets"])]
        try:
            r["groups"] = json.loads(r["groups"]) if r.get("groups") else []
        except (json.JSONDecodeError, TypeError):
            r["groups"] = []
        # 舊紀錄沒有分組，就當成一組沒有子標題的要點，顯示端不用分兩套路徑
        if not r["groups"] and r["bullets"]:
            r["groups"] = [{"heading": "", "points": r["bullets"]}]
        r["summary"] = r.get("summary") or ""
    return rows


def update_segment_text(seg_id: int, text: str) -> None:
    with _cursor() as cur:
        cur.execute("UPDATE segments SET text=? WHERE id=?", (text, seg_id))


def replace_sections(session_id: str, sections) -> None:
    """整批換掉一堂課的段落摘要（重新生成筆記時用）。"""
    with _cursor() as cur:
        cur.execute("DELETE FROM sections WHERE session_id=?", (session_id,))
        for i, s in enumerate(sections, 1):
            cur.execute(
                "INSERT INTO sections (session_id, seq, start_s, end_s, title, "
                "bullets, user_note, summary, groups) VALUES (?,?,?,?,?,?,?,?,?)",
                (session_id, i, s["start_s"], s["end_s"], s["title"],
                 json.dumps(s["bullets"], ensure_ascii=False), s.get("user_note"),
                 s.get("summary") or None, _dump(s.get("groups"))),
            )


def count_sections(session_id: str) -> int:
    with _cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM sections WHERE session_id=?", (session_id,))
        return int(cur.fetchone()["n"])
