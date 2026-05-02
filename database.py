import sqlite3
import json

DB_PATH = "face.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def create_table():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            relationship TEXT,
            embeddings TEXT,   -- ✅ renamed (plural)
            reminder TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_person(name, relationship, embeddings, reminder):
    conn = get_conn()
    c = conn.cursor()

    # ✅ ensure correct format (avoid double encoding)
    if isinstance(embeddings, str):
        data = embeddings
    else:
        data = json.dumps(embeddings)

    c.execute("""
        INSERT INTO persons(name, relationship, embeddings, reminder)
        VALUES (?, ?, ?, ?)
    """, (name, relationship, data, reminder))

    conn.commit()
    conn.close()


def get_all_persons():
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT name, relationship, embeddings, reminder FROM persons")

    rows = c.fetchall()
    conn.close()

    return rows