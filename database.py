import sqlite3
import json
from datetime import datetime, date
from config import DB_PATH

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_table():
    conn = get_conn()
    c = conn.cursor()

    # Persons Table (The AI's Social Memory)
    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT,
            relationship TEXT,
            embeddings   TEXT,
            reminder     TEXT,
            last_seen    TEXT,
            visit_count  INTEGER DEFAULT 0
        )
    """)

    # Reminders Table (The Patient's Schedule)
    c.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_text    TEXT NOT NULL,
            reminder_time    TEXT NOT NULL,
            category         TEXT DEFAULT 'General',
            completed_status INTEGER DEFAULT 0
        )
    """)

    # Logs Table (The System Audit Trail)
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

def insert_person(name, relationship, embeddings, reminder=""):
    conn = get_conn()
    c = conn.cursor()
    data = embeddings if isinstance(embeddings, str) else json.dumps(embeddings)
    c.execute(
        "INSERT INTO persons(name, relationship, embeddings, reminder) VALUES (?,?,?,?)",
        (name, relationship, data, reminder)
    )
    conn.commit()
    conn.close()

def get_all_persons():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name, relationship, embeddings, reminder, last_seen, visit_count FROM persons")
    rows = c.fetchall()
    conn.close()
    return rows

def update_person_last_seen(name: str):
    conn = get_conn()
    c = conn.cursor()
    ts = datetime.now().isoformat()
    c.execute("""
        UPDATE persons
        SET last_seen = ?, visit_count = COALESCE(visit_count, 0) + 1
        WHERE name = ?
    """, (ts, name))
    conn.commit()
    conn.close()

# ── Reminders (Crucial for Reminder Engine) ───────────────────────────────────

def insert_reminder(reminder_text, reminder_time, category="General"):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO reminders(reminder_text, reminder_time, category) VALUES (?,?,?)",
        (reminder_text, reminder_time, category)
    )
    conn.commit()
    conn.close()

def get_all_reminders():
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, reminder_text, reminder_time, category, completed_status "
        "FROM reminders ORDER BY reminder_time"
    )
    rows = c.fetchall()
    conn.close()
    return rows

def get_pending_reminders_for_time(time_str):
    """FIXED: Needed by reminder_engine.py"""
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, reminder_text, category FROM reminders "
        "WHERE reminder_time=? AND completed_status=0",
        (time_str,)
    )
    rows = c.fetchall()
    conn.close()
    return rows

def mark_reminder_complete(reminder_id):
    """FIXED: Needed by reminder_engine.py"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE reminders SET completed_status=1 WHERE id=?", (reminder_id,))
    conn.commit()
    conn.close()

def delete_reminder(reminder_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM reminders WHERE id=?", (reminder_id,))
    conn.commit()
    conn.close()

# ── Logs ──────────────────────────────────────────────────────────────────────

def get_logs(limit=100):
    """FIXED: Named correctly for dashboard.py"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, timestamp, event_type, event_description FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def insert_log(event_type, event_description):
    """Logs an event (Recognition, Voice, etc.) to the database."""
    conn = get_conn()
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute(
            "INSERT INTO logs(timestamp, event_type, event_description) VALUES (?,?,?)",
            (ts, event_type, event_description)
        )
        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Log failure: {e}")
    finally:
        conn.close()