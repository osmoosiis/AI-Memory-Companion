"""
main.py — CogniCare AI entry point.
STABLE VERSION — fixed repeated announcements + voice starvation.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2 as cv
import json
import time
from datetime import datetime

from state import state

from config import (
    IP_WEBCAM_URL,
    CAMERA_INDEX,
    STABILITY_THRESHOLD,
    VISIT_COOLDOWN,
    WHO_IS_THIS_DELAY,
    UNKNOWN_ALERT_FRAMES,
    DAILY_SUMMARY_HOUR,
)

from database import (
    create_table,
    get_all_persons,
    insert_log,
    update_person_last_seen,
)

from face_module import (
    get_embedding,
    find_best_match,
)

from audio_manager import (
    start as start_audio,
    enqueue,
    speaking,
)

from reminder_engine import start as start_reminders
from voice_assistant import start as start_voice

from context_engine import (
    generate_who_is_this_whisper,
    generate_unknown_alert,
    generate_orientation_message,
)

from daily_summary import speak_summary


# =========================================================
# INIT DATABASE
# =========================================================

create_table()


# =========================================================
# LOAD KNOWN PERSONS
# =========================================================

def load_known_persons():

    rows = get_all_persons()

    persons = []

    for (
        name,
        relationship,
        embedding_json,
        reminder,
        last_seen,
        visit_count
    ) in rows:

        try:

            loaded = json.loads(embedding_json)

            # Normalize single embedding
            if loaded and isinstance(loaded[0], (int, float)):
                loaded = [loaded]

            persons.append({
                "name": name,
                "relationship": relationship,
                "embeddings": loaded,
                "reminder": reminder,
                "last_seen": last_seen,
                "visit_count": visit_count or 0,
            })

        except Exception as e:

            print(f"[DATABASE ERROR] {name}: {e}")

    return persons


known_persons = load_known_persons()

print(f"[COGNICARE] System Ready. Loaded {len(known_persons)} person(s)")


# =========================================================
# START SERVICES
# =========================================================

start_audio()

start_reminders()

start_voice()


# Startup orientation
enqueue(generate_orientation_message())


# =========================================================
# CAMERA
# =========================================================

source = IP_WEBCAM_URL if IP_WEBCAM_URL else CAMERA_INDEX

cap = cv.VideoCapture(source)

if not cap.isOpened():

    print("[CAMERA ERROR] Could not open camera")

    exit()


# =========================================================
# TRACKING VARIABLES
# =========================================================

last_visit_times = {}

last_announced_person = None

last_ai_process_time = 0.0

AI_PROCESS_INTERVAL = 0.7

daily_summary_spoken_date = None

last_seen_face_time = time.time()

FACE_LOST_RESET_TIME = 8

last_recognition_time = {}

RECOGNITION_COOLDOWN = 60


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("[CAMERA ERROR] Could not read frame")

        time.sleep(0.1)

        continue

    display_frame = cv.resize(frame, (640, 480))

    now_ts = time.time()

    detected_person_obj = None

    score = 0.0


    # =====================================================
    # AI PROCESSING
    # =====================================================

    if now_ts - last_ai_process_time >= AI_PROCESS_INTERVAL:

        last_ai_process_time = now_ts

        embedding = get_embedding(display_frame)

        current_label = "Unknown"

        if embedding:

            detected_person_obj, score = find_best_match(
                embedding,
                known_persons
            )

            if detected_person_obj:

                current_label = detected_person_obj["name"]

        # Stability buffer
        state.identity_buffer.append(current_label)

        if len(state.identity_buffer) > STABILITY_THRESHOLD:

            state.identity_buffer.pop(0)

        if len(set(state.identity_buffer)) == 1:

            state.stable_label = state.identity_buffer[0]

        # Unknown tracking
        if state.stable_label == "Unknown":

            state.unknown_consecutive += 1

        else:

            state.unknown_consecutive = 0

            state.unknown_alerted = False


    # =====================================================
    # RECOGNITION ANNOUNCEMENTS
    # =====================================================

    if (
        state.stable_label != "Unknown"
        and
        state.stable_label != last_announced_person
    ):

        now_ts2 = time.time()

        last_visit = last_visit_times.get(
            state.stable_label,
            0
        )

        last_rec = last_recognition_time.get(
            state.stable_label,
            0
        )

        if (
            (now_ts2 - last_visit) > VISIT_COOLDOWN
            and
            (now_ts2 - last_rec) > RECOGNITION_COOLDOWN
        ):

            update_person_last_seen(state.stable_label)

            last_visit_times[state.stable_label] = now_ts2

            last_recognition_time[state.stable_label] = now_ts2

            if detected_person_obj:

                detected_person_obj["visit_count"] = (
                    detected_person_obj.get("visit_count", 0)
                    + 1
                )

        if not speaking():

            person = detected_person_obj or {
                "name": state.stable_label,
                "relationship": "visitor"
            }

            # SHORTER speech
            msg = (
                f"This is {person.get('name')}, "
                f"your {person.get('relationship', 'visitor')}."
            )

            enqueue(msg)

            last_announced_person = state.stable_label

            state.last_person = state.stable_label

            state.face_stable_since = now_ts2

            insert_log(
                "RECOGNITION",
                f"Recognized: {state.stable_label} ({score:.2f})"
            )


    # =====================================================
    # WHO IS THIS WHISPER
    # =====================================================

    if (
        state.stable_label != "Unknown"
        and
        state.face_stable_since is not None
        and
        (now_ts - state.face_stable_since) >= WHO_IS_THIS_DELAY
        and
        state.stable_label not in state.who_whispered
        and
        detected_person_obj is not None
        and
        not speaking()
    ):

        whisper_msg = generate_who_is_this_whisper(
            detected_person_obj
        )

        enqueue(whisper_msg)

        state.who_whispered.add(state.stable_label)

        insert_log(
            "WHO_IS_THIS",
            f"Whispered: {state.stable_label}"
        )


    # =====================================================
    # UNKNOWN PERSON ALERT
    # =====================================================

    if (
        state.unknown_consecutive >= UNKNOWN_ALERT_FRAMES
        and
        not state.unknown_alerted
        and
        not speaking()
    ):

        alert_msg = generate_unknown_alert(
            state.unknown_consecutive
        )

        enqueue(alert_msg, priority=True)

        state.unknown_alerted = True

        insert_log(
            "UNKNOWN_ALERT",
            f"Unknown person detected"
        )


    # =====================================================
    # FACE LOST RESET
    # =====================================================

    if state.stable_label != "Unknown":

        last_seen_face_time = time.time()

    if time.time() - last_seen_face_time > FACE_LOST_RESET_TIME:

        last_announced_person = None

        state.last_person = None

        state.face_stable_since = None

        state.who_whispered.clear()


    # =====================================================
    # DAILY SUMMARY
    # =====================================================

    now_dt = datetime.now()

    if (
        now_dt.hour == DAILY_SUMMARY_HOUR
        and
        now_dt.minute == 0
        and
        daily_summary_spoken_date != now_dt.date()
    ):

        speak_summary()

        daily_summary_spoken_date = now_dt.date()


    # =====================================================
    # UI DRAWING
    # =====================================================

    is_known = state.stable_label != "Unknown"

    status_color = (
        (0, 220, 0)
        if is_known
        else (0, 80, 255)
    )

    cv.rectangle(
        display_frame,
        (0, 0),
        (640, 60),
        (30, 30, 30),
        -1
    )

    cv.putText(
        display_frame,
        f"CogniCare AI | {state.stable_label}",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    # Speaking indicator
    if speaking():

        cv.putText(
            display_frame,
            "Speaking...",
            (20, 470),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
        )

    cv.imshow(
        "CogniCare AI",
        display_frame
    )

    if cv.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv.destroyAllWindows()

insert_log(
    "SYSTEM",
    "CogniCare AI shut down"
)

print("[COGNICARE] Shutdown complete")