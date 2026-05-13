"""
database.py — SQLite data layer for CogniCare AI.

Key fixes vs original:
  - Column name was 'embedding' in INSERT but 'embeddings' in CREATE TABLE → unified to 'embeddings'
  - Added get_today_summary() used by daily_summary.py
  - update_person_last_seen now also accepts id-based lookup
  - insert_person guards duplicate column name
"""

import sqlite3
import json
from datetime import datetime, date
from config import DB_PATH


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_table():
    conn = get_conn()
    c = conn.cursor()

    # ── Persons ──────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            relationship TEXT,
            embeddings   TEXT,
            reminder     TEXT,
            last_seen    TEXT,
            visit_count  INTEGER DEFAULT 0
        )
    """)

    # ── Reminders ────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_text    TEXT NOT NULL,
            reminder_time    TEXT NOT NULL,
            category         TEXT DEFAULT 'General',
            completed_status INTEGER DEFAULT 0
        )
    """)

    # ── Logs ─────────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            event_type        TEXT NOT NULL,
            event_description TEXT
        )
    """)

    conn.commit()
    conn.close()


# ── Persons ───────────────────────────────────────────────────────────────────

def insert_person(name: str, relationship: str, embeddings, reminder: str = ""):
    """Insert a new person.  embeddings can be a list or a JSON string."""
    conn = get_conn()
    c = conn.cursor()
    data = embeddings if isinstance(embeddings, str) else json.dumps(embeddings)
    c.execute(
        "INSERT INTO persons(name, relationship, embeddings, reminder) VALUES (?,?,?,?)",
        (name, relationship, data, reminder),
    )
    conn.commit()
    conn.close()


def get_all_persons():
    """Returns (name, relationship, embeddings_json, reminder, last_seen, visit_count)."""
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT name, relationship, embeddings, reminder, last_seen, visit_count FROM persons"
    )
    rows = c.fetchall()
    conn.close()
    return rows


def update_person_last_seen(name: str):
    conn = get_conn()
    c = conn.cursor()
    ts = datetime.now().isoformat()
    c.execute(
        """
        UPDATE persons
        SET last_seen = ?, visit_count = COALESCE(visit_count, 0) + 1
        WHERE name = ?
        """,
        (ts, name),
    )
    conn.commit()
    conn.close()


# ── Reminders ─────────────────────────────────────────────────────────────────

def insert_reminder(reminder_text: str, reminder_time: str, category: str = "General"):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO reminders(reminder_text, reminder_time, category) VALUES (?,?,?)",
        (reminder_text, reminder_time, category),
    )
    conn.commit()
    conn.close()


def get_all_reminders():
    """Returns (id, reminder_text, reminder_time, category, completed_status)."""
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, reminder_text, reminder_time, category, completed_status "
        "FROM reminders ORDER BY reminder_time"
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_pending_reminders_for_time(time_str: str):
    """Returns reminders whose reminder_time == time_str and are not completed."""
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, reminder_text, category FROM reminders "
        "WHERE reminder_time = ? AND completed_status = 0",
        (time_str,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def mark_reminder_complete(reminder_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE reminders SET completed_status = 1 WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()


def delete_reminder(reminder_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()


# ── Logs ──────────────────────────────────────────────────────────────────────

def insert_log(event_type: str, event_description: str):
    conn = get_conn()
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute(
            "INSERT INTO logs(timestamp, event_type, event_description) VALUES (?,?,?)",
            (ts, event_type, event_description),
        )
        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Log failure: {e}")
    finally:
        conn.close()


def get_logs(limit: int = 100):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, timestamp, event_type, event_description "
        "FROM logs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_today_summary():
    """
    Returns all log rows for today.
    Used by daily_summary.py.
    Row format: (id, timestamp, event_type, event_description)
    """
    conn = get_conn()
    c = conn.cursor()
    today = date.today().strftime("%Y-%m-%d")
    c.execute(
        "SELECT id, timestamp, event_type, event_description FROM logs "
        "WHERE timestamp LIKE ? ORDER BY id ASC",
        (f"{today}%",),
    )
    rows = c.fetchall()
    conn.close()
    return rows