
import json

import cv2
import numpy as np
import sqlite3
from deepface import DeepFace

from camera_utils import capture_frame_interactive
from database import DB_PATH


def get_conn():
    return sqlite3.connect(DB_PATH)


# ── Read ──────────────────────────────────────────────────────────────────────

def list_persons():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, name, relationship, reminder, last_seen, visit_count FROM persons")
    rows = c.fetchall()
    conn.close()
    return rows


def print_persons(rows):
    if not rows:
        print("\n  (No persons registered)\n")
        return
    print()
    print(f"  {'ID':<5} {'Name':<20} {'Relationship':<20} {'Visits':<8} {'Last Seen':<22} Reminder")
    print("  " + "-" * 90)
    for pid, name, rel, reminder, last_seen, visits in rows:
        print(
            f"  {pid:<5} {name:<20} {rel:<20} {(visits or 0):<8} "
            f"{(last_seen or 'Never'):<22} {reminder or '—'}"
        )
    print()


# ── Write ─────────────────────────────────────────────────────────────────────

def delete_person(person_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()


def delete_all():
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM persons")
    conn.commit()
    conn.close()


def update_reminder(person_id, new_reminder):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE persons SET reminder = ? WHERE id = ?", (new_reminder, person_id))
    conn.commit()
    conn.close()


def get_embedding_by_id(person_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT embeddings FROM persons WHERE id = ?", (person_id,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def update_embedding(person_id, new_embedding):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE persons SET embeddings = ? WHERE id = ?",
        (json.dumps(new_embedding), person_id),
    )
    conn.commit()
    conn.close()


# ── Add photos ────────────────────────────────────────────────────────────────

def add_photos_to_person(person_id, name):
    existing_embs = get_embedding_by_id(person_id)
    if existing_embs is None:
        print("  Could not load existing embeddings.")
        return
    # Normalise to list-of-lists
    if existing_embs and isinstance(existing_embs[0], (int, float)):
        existing_embs = [existing_embs]

    new_embeddings = []

    while True:
        print(f"\n  New photos added so far: {len(new_embeddings)}")
        print("  1. Capture from webcam")
        print("  2. Load from image file")
        print("  3. Save and update")
        print("  4. Cancel")
        choice = input("  Choose: ").strip()

        if choice == "1":
            frame = capture_frame_interactive()
            if frame is not None:
                try:
                    result = DeepFace.represent(
                        frame, model_name="Facenet", enforce_detection=False
                    )
                    emb = result[0]["embedding"]
                    new_embeddings.append(emb)
                    print(f"  Captured! Total new: {len(new_embeddings)}")
                except Exception as e:
                    print(f"  Failed: {e}")

        elif choice == "2":
            path = input("  Image path: ").strip()
            img  = cv2.imread(path)
            if img is None:
                print("  Could not load image.")
                continue
            try:
                result = DeepFace.represent(
                    img, model_name="Facenet", enforce_detection=False
                )
                emb = result[0]["embedding"]
                new_embeddings.append(emb)
                print(f"  Loaded! Total new: {len(new_embeddings)}")
            except Exception as e:
                print(f"  Failed: {e}")

        elif choice == "3":
            if not new_embeddings:
                print("  No new photos added.")
                continue
            all_embeddings = existing_embs + new_embeddings
            # Store all embeddings (not averaged) so find_best_match works better
            update_embedding(person_id, all_embeddings)
            print(f"\n  '{name}' updated with {len(new_embeddings)} new photo(s)! "
                  f"Total samples: {len(all_embeddings)}.")
            break

        elif choice == "4":
            print("  Cancelled.")
            break
        else:
            print("  Invalid option.")


# ── Main CLI loop ─────────────────────────────────────────────────────────────

def main():
    while True:
        print("=" * 50)
        print("   CogniCare AI — Database Manager")
        print("=" * 50)
        print("  1. List all registered persons")
        print("  2. Delete a person by ID")
        print("  3. Delete ALL persons")
        print("  4. Edit reminder message")
        print("  5. Add more photos to a person")
        print("  6. Quit")
        print("=" * 50)

        choice = input("Choose: ").strip()

        if choice == "1":
            print_persons(list_persons())

        elif choice == "2":
            rows = list_persons()
            print_persons(rows)
            if not rows:
                continue
            try:
                pid  = int(input("Enter ID to delete: ").strip())
                ids  = [r[0] for r in rows]
                if pid not in ids:
                    print(f"  No person with ID {pid}.\n")
                    continue
                name = next(r[1] for r in rows if r[0] == pid)
                if input(f"  Delete '{name}'? (y/n): ").strip().lower() == "y":
                    delete_person(pid)
                    print(f"  '{name}' deleted.\n")
                else:
                    print("  Cancelled.\n")
            except ValueError:
                print("  Invalid input.\n")

        elif choice == "3":
            rows = list_persons()
            if not rows:
                print("  Nothing to delete.\n")
                continue
            if (
                input(
                    f"  Delete ALL {len(rows)} person(s)? This cannot be undone. (y/n): "
                ).strip().lower()
                == "y"
            ):
                delete_all()
                print("  All persons deleted.\n")
            else:
                print("  Cancelled.\n")

        elif choice == "4":
            rows = list_persons()
            print_persons(rows)
            if not rows:
                continue
            try:
                pid = int(input("Enter ID to edit: ").strip())
                ids = [r[0] for r in rows]
                if pid not in ids:
                    print(f"  No person with ID {pid}.\n")
                    continue
                name         = next(r[1] for r in rows if r[0] == pid)
                new_reminder = input(f"  New reminder for '{name}': ").strip()
                update_reminder(pid, new_reminder)
                print(f"  Reminder updated for '{name}'.\n")
            except ValueError:
                print("  Invalid input.\n")

        elif choice == "5":
            rows = list_persons()
            print_persons(rows)
            if not rows:
                continue
            try:
                pid = int(input("Enter ID to update: ").strip())
                ids = [r[0] for r in rows]
                if pid not in ids:
                    print(f"  No person with ID {pid}.\n")
                    continue
                name = next(r[1] for r in rows if r[0] == pid)
                add_photos_to_person(pid, name)
            except ValueError:
                print("  Invalid input.\n")

        elif choice == "6":
            print("Bye!")
            break

        else:
            print("  Invalid option, try again.\n")


if __name__ == "__main__":
    main()