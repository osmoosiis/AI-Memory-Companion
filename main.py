import cv2 as cv
import json
import time
from collections import defaultdict

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

# Stability buffer - track recent results per face position
STABILITY_THRESHOLD = 5  # must match this many frames in a row to confirm
recent_results = []  # list of last N recognition results
stable_label = "Unknown"
stable_color = (0, 0, 255)

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

    if len(faces) == 0:
        # No face visible, reset buffer
        recent_results.clear()
        stable_label = "Unknown"
        stable_color = (0, 0, 255)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        embedding = get_embedding(face_crop)

        current_label = "Unknown"
        current_person = None

        if embedding is not None:
            for person in known_persons:
                if is_match(embedding, person["embedding"]):
                    current_label = person["name"]
                    current_person = person
                    break

        # Add to rolling buffer
        recent_results.append(current_label)
        if len(recent_results) > STABILITY_THRESHOLD:
            recent_results.pop(0)

        # Only update stable label if last N frames all agree
        if len(recent_results) == STABILITY_THRESHOLD and len(set(recent_results)) == 1:
            stable_label = recent_results[0]
            if stable_label != "Unknown" and current_person:
                stable_color = (0, 255, 0)
                display_label = f"{current_person['name']} ({current_person['relationship']})"

                # TTS
                now = time.time()
                last = last_spoken.get(current_person["name"], 0)
                if now - last > COOLDOWN_SECONDS:
                    msg = f"This is {current_person['name']}, your {current_person['relationship']}. {current_person['reminder']}"
                    print(f"[TTS] {msg}")
                    speak(msg)
                    last_spoken[current_person["name"]] = now
            else:
                stable_color = (0, 0, 255)
                display_label = "Unknown"
        else:
            # Still building up buffer, show current stable state
            display_label = stable_label if stable_label != "Unknown" else "Unknown"

        cv.rectangle(frame, (x, y), (x+w, y+h), stable_color, 2)
        cv.putText(frame, display_label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, stable_color, 2)

    cv.imshow("AI Memory Companion", frame)

    if cv.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()