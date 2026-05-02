import cv2 as cv
import json
import time
from config import COOLDOWN_SECONDS, IP_WEBCAM_URL, CAMERA_INDEX
from database import create_table, get_all_persons
from face_module import get_embedding, find_best_match
from person_audio import speak

create_table()


def load_known_persons():
    rows = get_all_persons()
    persons = []

    for name, relationship, embedding_json, reminder in rows:
        loaded = json.loads(embedding_json)

        if isinstance(loaded[0], float):
            loaded = [loaded]

        persons.append({
            "name": name,
            "relationship": relationship,
            "embeddings": loaded,
            "reminder": reminder
        })

    return persons


known_persons = load_known_persons()
print(f"Loaded {len(known_persons)} person(s)")

last_spoken = {}
last_person_seen = None

STABILITY_THRESHOLD = 7
recent_results = []
stable_label = "Unknown"

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv.VideoCapture(IP_WEBCAM_URL if IP_WEBCAM_URL else CAMERA_INDEX)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 🔥 RESIZE FOR SPEED
    frame = cv.resize(frame, (640, 480))

    frame_count += 1

    # 🔥 PROCESS FEWER FRAMES
    if frame_count % 5 != 0:
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        recent_results.clear()
        stable_label = "Unknown"
        continue

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]

        embedding = get_embedding(face_crop)

        current_label = "Unknown"
        current_person = None

        if embedding is not None:
            current_person = find_best_match(embedding, known_persons)
            if current_person:
                current_label = current_person["name"]

        # 🔥 STABILITY BUFFER
        if recent_results and current_label != recent_results[-1]:
            recent_results.clear()

        recent_results.append(current_label)

        if len(recent_results) > STABILITY_THRESHOLD:
            recent_results.pop(0)

        if len(recent_results) == STABILITY_THRESHOLD and len(set(recent_results)) == 1:
            stable_label = recent_results[0]

            # 🔥 SMART SPEAKING (NO REPEAT SPAM)
            if stable_label != "Unknown" and current_person:
                if stable_label != last_person_seen:
                    msg = f"This is {current_person['name']}, your {current_person['relationship']}. {current_person['reminder']}"
                    print(f"[TTS] {msg}")
                    speak(msg)
                    last_person_seen = stable_label

        # 🔲 DRAW UI
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.putText(
            frame,
            stable_label,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv.imshow("AI Memory Companion", frame)

    if cv.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()