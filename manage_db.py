import sqlite3
import json
import numpy as np
import cv2
from deepface import DeepFace

DB_PATH = "face.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def list_persons():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, name, relationship, reminder FROM persons")
    rows = c.fetchall()
    conn.close()
    return rows


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
    c.execute("SELECT embedding FROM persons WHERE id = ?", (person_id,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def update_embedding(person_id, new_embedding):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE persons SET embedding = ? WHERE id = ?", (json.dumps(new_embedding), person_id))
    conn.commit()
    conn.close()


def print_persons(rows):
    if not rows:
        print("\n  (No persons registered)\n")
        return
    print()
    print(f"  {'ID':<5} {'Name':<20} {'Relationship':<20} {'Reminder'}")
    print("  " + "-" * 70)
    for row in rows:
        pid, name, relationship, reminder = row
        print(f"  {pid:<5} {name:<20} {relationship:<20} {reminder}")
    print()


def add_photos_to_person(person_id, name):
    existing_emb = get_embedding_by_id(person_id)
    if existing_emb is None:
        print("  Could not load existing embedding.")
        return

    new_embeddings = []

    while True:
        print()
        print(f"  New photos added so far: {len(new_embeddings)}")
        print("  1. Capture from webcam")
        print("  2. Load from image file")
        print("  3. Save and update")
        print("  4. Cancel")
        choice = input("  Choose: ").strip()

        if choice == "1":
            cap = cv2.VideoCapture(0)
            print("  Press 'c' to capture, 'q' to cancel.")
            while True:
                ret, frame = cap.read()
                cv2.imshow("Capture - press C", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    try:
                        result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
                        emb = result[0]["embedding"]
                        new_embeddings.append(emb)
                        print(f"  Captured! New photos: {len(new_embeddings)}")
                    except Exception as e:
                        print(f"   Failed: {e}")
                    break
                elif key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        elif choice == "2":
            path = input("  Enter image file path: ").strip()
            try:
                img = cv2.imread(path)
                if img is None:
                    print("   Could not load image.")
                    continue
                result = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
                emb = result[0]["embedding"]
                new_embeddings.append(emb)
                print(f"  Loaded! New photos: {len(new_embeddings)}")
            except Exception as e:
                print(f"  Failed: {e}")

        elif choice == "3":
            if len(new_embeddings) == 0:
                print("   No new photos added.")
                continue
            # Average existing embedding with all new ones
            all_embeddings = [existing_emb] + new_embeddings
            avg_embedding = np.mean(all_embeddings, axis=0).tolist()
            update_embedding(person_id, avg_embedding)
            print(f"\n '{name}' updated with {len(new_embeddings)} new photo(s)!")
            break

        elif choice == "4":
            print("  Cancelled.")
            break
        else:
            print("  Invalid option.")


def main():
    while True:
        print("=" * 40)
        print("   AI Memory Companion — DB Manager")
        print("=" * 40)
        print("  1. List all registered persons")
        print("  2. Delete a person by ID")
        print("  3. Delete ALL persons")
        print("  4. Edit reminder message")
        print("  5. Add more photos to a person")
        print("  6. Quit")
        print("=" * 40)

        choice = input("Choose an option: ").strip()

        if choice == "1":
            rows = list_persons()
            print_persons(rows)

        elif choice == "2":
            rows = list_persons()
            print_persons(rows)
            if not rows:
                continue
            try:
                pid = int(input("Enter ID to delete: ").strip())
                ids = [r[0] for r in rows]
                if pid not in ids:
                    print(f"   No person with ID {pid}.\n")
                    continue
                name = next(r[1] for r in rows if r[0] == pid)
                confirm = input(f"  Delete '{name}'? (y/n): ").strip().lower()
                if confirm == "y":
                    delete_person(pid)
                    print(f"   '{name}' deleted.\n")
                else:
                    print("  Cancelled.\n")
            except ValueError:
                print("   Invalid input.\n")

        elif choice == "3":
            rows = list_persons()
            if not rows:
                print("  Nothing to delete.\n")
                continue
            confirm = input(f"  Delete ALL {len(rows)} person(s)? This cannot be undone. (y/n): ").strip().lower()
            if confirm == "y":
                delete_all()
                print("   All persons deleted.\n")
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
                    print(f"   No person with ID {pid}.\n")
                    continue
                name = next(r[1] for r in rows if r[0] == pid)
                new_reminder = input(f"  New reminder for '{name}': ").strip()
                update_reminder(pid, new_reminder)
                print(f"  Reminder updated for '{name}'.\n")
            except ValueError:
                print("   Invalid input.\n")

        elif choice == "5":
            rows = list_persons()
            print_persons(rows)
            if not rows:
                continue
            try:
                pid = int(input("Enter ID to update: ").strip())
                ids = [r[0] for r in rows]
                if pid not in ids:
                    print(f"   No person with ID {pid}.\n")
                    continue
                name = next(r[1] for r in rows if r[0] == pid)
                add_photos_to_person(pid, name)
            except ValueError:
                print("   Invalid input.\n")

        elif choice == "6":
            print("Bye!")
            break

        else:
            print("   Invalid option, try again.\n")


if __name__ == "__main__":
    main()