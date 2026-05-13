"""
register_person.py — CLI tool to register a new person into CogniCare.

Fix vs original:
  - insert_person() column is now 'embeddings' (unified in database.py).
  - get_all_persons() returns 6 columns now — unpacking updated accordingly.
"""

import cv2
import json

from database import insert_person, create_table, get_all_persons
from face_module import get_embedding, cosine_similarity
from camera_utils import capture_frame_interactive
from config import THRESHOLD

create_table()

print("=== Register New Person ===\n")

name         = input("Enter name: ").strip()
relationship = input("Enter relationship: ").strip()
reminder     = input("Enter reminder message: ").strip()

embeddings = []

while True:
    print(f"\nCaptured so far: {len(embeddings)} sample(s)")
    print("1. Capture from webcam")
    print("2. Load from image file")
    print("3. Save to database")
    print("4. Cancel")

    choice = input("Choose: ").strip()

    # ── Webcam ────────────────────────────────────────────────────────────────
    if choice == "1":
        frame = capture_frame_interactive()
        if frame is not None:
            emb = get_embedding(frame)
            if emb:
                embeddings.append(emb)
                print(f"  Captured ✔  (total: {len(embeddings)})")
            else:
                print("  No face detected ❌")

    # ── Image file ────────────────────────────────────────────────────────────
    elif choice == "2":
        path = input("  Image path: ").strip()
        img  = cv2.imread(path)
        if img is None:
            print("  Could not load image ❌")
            continue
        emb = get_embedding(img)
        if emb:
            embeddings.append(emb)
            print(f"  Loaded ✔  (total: {len(embeddings)})")
        else:
            print("  No face detected ❌")

    # ── Save ──────────────────────────────────────────────────────────────────
    elif choice == "3":
        if len(embeddings) < 3:
            print("  Please capture at least 3 samples for reliable recognition ❌")
            continue

        # Duplicate check
        existing  = get_all_persons()   # returns 6-tuple now
        duplicate = False

        for ename, erel, emb_json, ereminder, elast_seen, evisit_count in existing:
            stored_embeddings = json.loads(emb_json)
            # Normalise stored format
            if stored_embeddings and isinstance(stored_embeddings[0], (int, float)):
                stored_embeddings = [stored_embeddings]

            for stored_emb in stored_embeddings:
                score = cosine_similarity(embeddings[0], stored_emb)
                if score > THRESHOLD:
                    print(f"  Warning: looks similar to {ename} (score {score:.2f})")
                    confirm = input("  Save anyway? (y/n): ").strip().lower()
                    if confirm != "y":
                        duplicate = True
                    break

            if duplicate:
                break

        if not duplicate:
            insert_person(name, relationship, json.dumps(embeddings), reminder)
            print(f"\n  ✅ Saved '{name}' with {len(embeddings)} sample(s).")
        break

    elif choice == "4":
        print("  Cancelled.")
        break
    else:
        print("  Invalid option.")