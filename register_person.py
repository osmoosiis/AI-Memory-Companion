import cv2
import json
from database import insert_person, create_table, get_all_persons
from face_module import get_embedding, cosine_similarity
from camera_utils import capture_frame_interactive
from config import THRESHOLD

create_table()

print("=== Register New Person ===\n")

name = input("Enter name: ").strip()
relationship = input("Enter relationship: ").strip()
reminder = input("Enter reminder message: ").strip()

embeddings = []

while True:
    print(f"\nCaptured: {len(embeddings)}")
    print("1. Webcam")
    print("2. Image file")
    print("3. Save")
    print("4. Cancel")

    choice = input("Choose: ")

    # 📸 webcam
    if choice == "1":
        frame = capture_frame_interactive()
        if frame is not None:
            emb = get_embedding(frame)

            if emb:
                print("Embedding length:", len(emb))  # debug
                embeddings.append(emb)
                print("Captured ✔")
            else:
                print("Failed ❌")

    # 🖼️ image
    elif choice == "2":
        path = input("Path: ")
        img = cv2.imread(path)

        if img is None:
            print("Invalid image ❌")
            continue

        emb = get_embedding(img)

        if emb:
            print("Embedding length:", len(emb))
            embeddings.append(emb)
            print("Loaded ✔")
        else:
            print("Failed ❌")

    # 💾 save
    elif choice == "3":
        if len(embeddings) < 3:
            print("Capture at least 3 images ❌")
            continue

        existing = get_all_persons()
        duplicate = False

        for ename, erel, emb_json, ereminder in existing:
            stored_embeddings = json.loads(emb_json)

            for stored_emb in stored_embeddings:
                score = cosine_similarity(embeddings[0], stored_emb)

                if score > THRESHOLD:
                    print(f"Looks like {ename} (score {score:.2f})")
                    confirm = input("Save anyway? (y/n): ")

                    if confirm != "y":
                        duplicate = True
                    break

            if duplicate:
                break

        if not duplicate:
            insert_person(
                name,
                relationship,
                json.dumps(embeddings),
                reminder
            )
            print(f"\nSaved {name} with {len(embeddings)} samples ✔")

        break

    elif choice == "4":
        break