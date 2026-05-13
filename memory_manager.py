"""
memory_manager.py — Convenience helpers for person memory queries.
"""

from database import get_conn


def get_last_seen_person(name: str):
    """Returns the ISO timestamp of when 'name' was last seen, or None."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT last_seen FROM persons WHERE name = ?", (name,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_visit_count(name: str) -> int:
    """Returns total visit count for the named person."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT visit_count FROM persons WHERE name = ?", (name,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0