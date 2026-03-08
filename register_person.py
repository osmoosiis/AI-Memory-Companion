import cv2
import json
import numpy as np
from deepface import DeepFace
from database import insert_person, create_table, get_all_persons
from face_module import compare_embeddings
from config import THRESHOLD

create_table()

print("=== Register New Person ===")
print("You can capture multiple photos OR load from image files.")
print()

name = input("Enter name: ").strip()
relationship = input("Enter relationship: ").strip()
reminder = input("Enter reminder message: ").strip()

embeddings = []

while True:
    print()
    print(f"  Photos captured so far: {len(embeddings)}")
    print("  1. Capture from webcam")
    print("  2. Load from image file")
    print("  3. Done (save person)")
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
                    embeddings.append(emb)
                    print(f"   Captured! Total: {len(embeddings)}")
                except Exception as e:
                    print(f"  Failed to get embedding: {e}")
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
            embeddings.append(emb)
            print(f"   Loaded! Total: {len(embeddings)}")
        except Exception as e:
            print(f"   Failed: {e}")

    elif choice == "3":
        if len(embeddings) == 0:
            print("  No photos captured yet.")
            continue

        # Average all embeddings into one
        avg_embedding = np.mean(embeddings, axis=0).tolist()

        # Duplicate check
        existing = get_all_persons()
        duplicate = False
        for ename, erel, emb_json, ereminder in existing:
            dist = compare_embeddings(avg_embedding, json.loads(emb_json))
            if dist < THRESHOLD:
                print(f"This face looks like '{ename}' already in the database (distance: {dist:.3f}).")
                confirm = input("  Register anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("  Registration cancelled.")
                    duplicate = True
                break

        if not duplicate:
            insert_person(name, relationship, avg_embedding, reminder)
            print(f"\n '{name}' registered with {len(embeddings)} photo(s)!")
        break

    elif choice == "4":
        print("Cancelled.")
        break
    else:
        print("  Invalid option.")