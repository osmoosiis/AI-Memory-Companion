import cv2
import json
from deepface import DeepFace
from database import insert_person, create_table, get_all_persons
from face_module import compare_embeddings
from config import THRESHOLD

create_table()

cap = cv2.VideoCapture(0)
print("Press 'c' to capture face, 'q' to quit.")

frame = None
while True:
    ret, f = cap.read()
    cv2.imshow("Register Person", f)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        frame = f
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

if frame is None:
    print("No frame captured.")
    exit()

embedding = DeepFace.represent(
    frame,
    model_name="Facenet",
    enforce_detection=False
)[0]["embedding"]

print(f"Embedding captured (length: {len(embedding)})")

# Duplicate check
existing = get_all_persons()
for name, relationship, emb_json, reminder in existing:
    dist = compare_embeddings(embedding, json.loads(emb_json))
    if dist < THRESHOLD:
        print(f"  This face looks like '{name}' already in the database (distance: {dist:.3f}).")
        confirm = input("Register anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Registration cancelled.")
            exit()
        break

name = input("Enter name: ").strip()
relationship = input("Enter relationship: ").strip()
reminder = input("Enter reminder message: ").strip()

insert_person(name, relationship, embedding, reminder)
print(f" '{name}' registered successfully!")