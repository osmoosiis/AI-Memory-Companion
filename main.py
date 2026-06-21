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

            # Normalize single embedding stored as flat list
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

    print(f"[CAMERA ERROR] Could not open camera (source={source}). "
          "Check CAMERA_INDEX in config.py.")

    exit()


# =========================================================
# TRACKING VARIABLES
# =========================================================

last_visit_times        = {}
last_announced_person   = None
last_ai_process_time    = 0.0
AI_PROCESS_INTERVAL     = 0.7
daily_summary_spoken_date = None
last_seen_face_time     = time.time()
FACE_LOST_RESET_TIME    = 8
last_recognition_time   = {}
RECOGNITION_COOLDOWN    = 60

# These are set inside the AI block and read in announcement/whisper blocks
_current_person_obj     = None   # the best-matching person dict (confident only)
_current_confident      = False  # whether the last AI result was a confident match
_current_score          = 0.0


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


    # =====================================================
    # AI PROCESSING
    # =====================================================

    if now_ts - last_ai_process_time >= AI_PROCESS_INTERVAL:

        last_ai_process_time = now_ts

        embedding = get_embedding(display_frame)

        current_label     = "Unknown"
        _current_person_obj = None
        _current_confident  = False
        _current_score      = 0.0

        if embedding:

            person_obj, score, confident = find_best_match(
                embedding,
                known_persons
            )

            if person_obj is not None:
                # Always store for UI display (even uncertain matches)
                _current_person_obj = person_obj
                _current_score      = score

                if confident:
                    # Only use the real name for state tracking when confident
                    current_label      = person_obj["name"]
                    _current_confident = True
                else:
                    # Uncertain — show "Maybe" in overlay but keep state as Unknown
                    current_label = f"Maybe {person_obj['name']}"

        # Stability buffer — only feed confirmed names or Unknown
        # NOTE: person_obj may be unbound if embedding was None, so use _current_confident
        stable_input = (
            _current_person_obj["name"]
            if (_current_confident and _current_person_obj)
            else "Unknown"
        )

        state.identity_buffer.append(stable_input)

        if len(state.identity_buffer) > STABILITY_THRESHOLD:
            state.identity_buffer.pop(0)

        if len(set(state.identity_buffer)) == 1:
            state.stable_label = state.identity_buffer[0]

        # Unknown tracking
        if state.stable_label == "Unknown":
            state.unknown_consecutive += 1
        else:
            state.unknown_consecutive = 0
            state.unknown_alerted     = False


    # =====================================================
    # RECOGNITION ANNOUNCEMENTS  (confident matches only)
    # =====================================================

    if (
        state.stable_label != "Unknown"
        and state.stable_label != last_announced_person
        and _current_confident                          # ← NEW: must be confident
        and _current_person_obj is not None
    ):

        now_ts2    = now_ts
        last_visit = last_visit_times.get(state.stable_label, 0)
        last_rec   = last_recognition_time.get(state.stable_label, 0)

        if (
            (now_ts2 - last_visit) > VISIT_COOLDOWN
            and (now_ts2 - last_rec) > RECOGNITION_COOLDOWN
        ):

            update_person_last_seen(state.stable_label)

            last_visit_times[state.stable_label]      = now_ts2
            last_recognition_time[state.stable_label] = now_ts2

            _current_person_obj["visit_count"] = (
                _current_person_obj.get("visit_count", 0) + 1
            )

        if not speaking():

            msg = (
                f"This is {_current_person_obj.get('name')}, "
                f"your {_current_person_obj.get('relationship', 'visitor')}."
            )

            enqueue(msg)

            last_announced_person  = state.stable_label
            state.last_person      = state.stable_label
            state.face_stable_since = now_ts

            insert_log(
                "RECOGNITION",
                f"Recognized: {state.stable_label} ({_current_score:.2f})"
            )


    # =====================================================
    # WHO IS THIS WHISPER
    # =====================================================

    if (
        state.stable_label != "Unknown"
        and state.face_stable_since is not None
        and (now_ts - state.face_stable_since) >= WHO_IS_THIS_DELAY
        and state.stable_label not in state.who_whispered
        and _current_person_obj is not None
        and _current_confident                          # ← only whisper when confident
        and not speaking()
    ):

        whisper_msg = generate_who_is_this_whisper(_current_person_obj)

        enqueue(whisper_msg)

        state.who_whispered.add(state.stable_label)

        insert_log("WHO_IS_THIS", f"Whispered: {state.stable_label}")


    # =====================================================
    # UNKNOWN PERSON ALERT
    # =====================================================

    if (
        state.unknown_consecutive >= UNKNOWN_ALERT_FRAMES
        and not state.unknown_alerted
        and not speaking()
    ):

        alert_msg = generate_unknown_alert(state.unknown_consecutive)

        enqueue(alert_msg, priority=True)

        state.unknown_alerted = True

        insert_log("UNKNOWN_ALERT", "Unknown person detected")


    # =====================================================
    # FACE LOST RESET
    # =====================================================

    if state.stable_label != "Unknown":
        last_seen_face_time = time.time()

    if time.time() - last_seen_face_time > FACE_LOST_RESET_TIME:

        last_announced_person   = None
        state.last_person       = None
        state.face_stable_since = None
        state.who_whispered.clear()


    # =====================================================
    # DAILY SUMMARY  (once per day at DAILY_SUMMARY_HOUR)
    # =====================================================

    now_dt = datetime.now()

    if (
        now_dt.hour == DAILY_SUMMARY_HOUR
        and now_dt.minute == 0
        and daily_summary_spoken_date != now_dt.date()
    ):

        speak_summary()

        daily_summary_spoken_date = now_dt.date()


    # =====================================================
    # UI DRAWING
    # =====================================================

    is_known = (
        state.stable_label != "Unknown"
        and _current_confident
    )

    is_maybe = (
        _current_person_obj is not None
        and not _current_confident
    )

    if is_known:
        status_color = (0, 220, 0)       # green
    elif is_maybe:
        status_color = (0, 200, 255)     # yellow-ish
    else:
        status_color = (0, 80, 255)      # red/orange for unknown

    # Determine display label
    if is_known:
        display_label = state.stable_label
    elif is_maybe and _current_person_obj:
        display_label = f"Maybe {_current_person_obj['name']}"
    else:
        display_label = "Unknown"

    cv.rectangle(display_frame, (0, 0), (640, 60), (30, 30, 30), -1)

    cv.putText(
        display_frame,
        f"CogniCare AI | {display_label}",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    if _current_score > 0:
        cv.putText(
            display_frame,
            f"Score: {_current_score:.2f}",
            (500, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

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

    cv.imshow("CogniCare AI", display_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv.destroyAllWindows()

insert_log("SYSTEM", "CogniCare AI shut down")

print("[COGNICARE] Shutdown complete")