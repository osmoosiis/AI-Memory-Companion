import sqlite3
import json


def create_table():
    conn = sqlite3.connect('face.db')
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            relationship TEXT,
            embedding TEXT,
            reminder TEXT
        )
    """)

    conn.commit()
    conn.close()



def insert_person(name, relationship, embedding, reminder):
    conn = sqlite3.connect('face.db')
    c = conn.cursor()

    c.execute("""
        INSERT INTO persons(name, relationship, embedding, reminder)
        VALUES (?, ?, ?, ?)
    """, (name, relationship, json.dumps(embedding), reminder))

    conn.commit()
    conn.close()

def get_all_persons():
    conn = sqlite3.connect('face.db')
    c = conn.cursor()

    c.execute("SELECT name, relationship, embedding, reminder FROM persons")
    rows = c.fetchall()

    conn.close()
    return rows