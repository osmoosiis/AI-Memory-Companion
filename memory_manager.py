from database import get_conn

def get_last_seen_person(name):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT last_seen FROM persons WHERE name = ?
    """, (name,))

    row = c.fetchone()
    return row[0] if row else None