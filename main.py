import cv2 as cv
import json
import time

from config import COOLDOWN_SECONDS, IP_WEBCAM_URL
from database import create_table, get_all_persons
from face_module import get_embedding, is_match
from tts_module import speak

create_table()

def load_known_persons():
    rows = get_all_persons()
    persons = []
    for name, relationship, embedding_json, reminder in rows:
        persons.append({
            "name": name,
            "relationship": relationship,
            "embedding": json.loads(embedding_json),
            "reminder": reminder,
        })
    return persons

known_persons = load_known_persons()
print(f"Loaded {len(known_persons)} known person(s) from database.")

last_spoken: dict[str, float] = {}

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

source = IP_WEBCAM_URL if IP_WEBCAM_URL else 0
cap = cv.VideoCapture(source)

print("Starting AI Memory Companion. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        embedding = get_embedding(face_crop)

        label = "Unknown"
        color = (0, 0, 255)  # Red for unknown

        if embedding is not None:
            for person in known_persons:
                if is_match(embedding, person["embedding"]):
                    label = f"{person['name']} ({person['relationship']})"
                    color = (0, 255, 0)  # Green for recognized

                    now = time.time()
                    last = last_spoken.get(person["name"], 0)
                    if now - last > COOLDOWN_SECONDS:
                        msg = f"This is {person['name']}, your {person['relationship']}. {person['reminder']}"
                        print(f"[TTS] {msg}")
                        speak(msg)
                        last_spoken[person["name"]] = now
                    break

        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv.imshow("AI Memory Companion", frame)

    if cv.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()