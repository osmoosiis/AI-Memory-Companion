import cv2
import json

from database import insert_person, create_table, get_all_persons
from face_module import get_embedding, cosine_similarity
from camera_utils import capture_frame_interactive
from config import THRESHOLD

create_table()

print("=== CogniCare — Register New Person ===\n")

name         = input("Enter name: ").strip()
relationship = input("Enter relationship (e.g. daughter, caregiver): ").strip()
reminder     = input("Enter reminder message (optional, press Enter to skip): ").strip()

embeddings = []

while True:

    print(f"\n  Samples captured: {len(embeddings)}")
    print("  1. Capture from webcam")
    print("  2. Load from image file")
    print("  3. Save to database")
    print("  4. Cancel")

    choice = input("Choose: ").strip()

    # ── Webcam ────────────────────────────────────────────────────────────────
    if choice == "1":
        frame = capture_frame_interactive()
        if frame is not None:
            emb = get_embedding(frame)
            if emb:
                embeddings.append(emb)
                print(f"  ✔ Sample captured. Total: {len(embeddings)}")
            else:
                print("  ✖ No face detected in this frame. Move closer to the camera and try again.")

    # ── Image file ────────────────────────────────────────────────────────────
    elif choice == "2":
        path = input("  Image path: ").strip()
        img  = cv2.imread(path)
        if img is None:
            print(f"  ✖ Could not load image at '{path}'. Check the path and try again.")
            continue
        emb = get_embedding(img)
        if emb:
            embeddings.append(emb)
            print(f"  ✔ Face extracted. Total: {len(embeddings)}")
        else:
            print("  ✖ No face detected in this image. Try a clearer, front-facing photo.")

    # ── Save ──────────────────────────────────────────────────────────────────
    elif choice == "3":
        if len(embeddings) < 3:
            print(f"  ✖ Need at least 3 samples for reliable recognition. "
                  f"You have {len(embeddings)}. Please capture more.")
            continue

        # Duplicate check using raw cosine similarity (not find_best_match)
        existing  = get_all_persons()   # (name, rel, emb_json, reminder, last_seen, visit_count)
        duplicate = False

        for ename, erel, emb_json, ereminder, elast_seen, evisit_count in existing:

            try:
                stored_embeddings = json.loads(emb_json)
            except Exception:
                continue

            # Normalize flat single embedding
            if stored_embeddings and isinstance(stored_embeddings[0], (int, float)):
                stored_embeddings = [stored_embeddings]

            for stored_emb in stored_embeddings:
                # Compare each new sample against each stored embedding
                for new_emb in embeddings:
                    score = cosine_similarity(new_emb, stored_emb)
                    if score >= THRESHOLD:
                        print(f"\n  ⚠ This face looks very similar to '{ename}' "
                              f"already in the database (score: {score:.2f}).")
                        confirm = input("  Save anyway? (y/n): ").strip().lower()
                        if confirm != "y":
                            duplicate = True
                        break
                if duplicate:
                    break

            if duplicate:
                break

        if not duplicate:
            insert_person(name, relationship, json.dumps(embeddings), reminder)
            print(f"\n   '{name}' registered successfully with {len(embeddings)} sample(s).")
            print(f"     They will now be recognized by CogniCare.")
        break

    elif choice == "4":
        print("  Registration cancelled.")
        break

    else:
        print("  Invalid option. Please enter 1, 2, 3, or 4.")