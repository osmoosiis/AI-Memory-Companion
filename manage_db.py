import sqlite3
import json


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


def main():
    while True:
        print("=" * 40)
        print("   AI Memory Companion — DB Manager")
        print("=" * 40)
        print("  1. List all registered persons")
        print("  2. Delete a person by ID")
        print("  3. Delete ALL persons")
        print("  4. Edit reminder message")
        print("  5. Quit")
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
                print("  Invalid input.\n")

        elif choice == "3":
            rows = list_persons()
            if not rows:
                print("  Nothing to delete.\n")
                continue
            confirm = input(f"  Delete ALL {len(rows)} person(s)? This cannot be undone. (y/n): ").strip().lower()
            if confirm == "y":
                delete_all()
                print("All persons deleted.\n")
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
                    print(f"No person with ID {pid}.\n")
                    continue
                name = next(r[1] for r in rows if r[0] == pid)
                new_reminder = input(f"  New reminder for '{name}': ").strip()
                update_reminder(pid, new_reminder)
                print(f"Reminder updated for '{name}'.\n")
            except ValueError:
                print(" Invalid input.\n")

        elif choice == "5":
            print("Bye!")
            break

        else:
            print("Invalid option, try again.\n")


if __name__ == "__main__":
    main()