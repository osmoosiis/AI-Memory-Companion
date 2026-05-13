"""
reminder_engine.py — Scheduled reminder checker.
FINAL STABLE VERSION
"""

import threading
import time
from datetime import datetime

from audio_manager import enqueue

from context_engine import generate_reminder_message

from database import (
    get_pending_reminders_for_time,
    mark_reminder_complete,
    insert_log,
)


# =========================================================
# GLOBALS
# =========================================================

_stop_event = threading.Event()

_spoken_this_minute = set()

_pending_ack = {}

_thread = None

ACKNOWLEDGE_WINDOW = 30


# =========================================================
# CHECK REMINDERS
# =========================================================

def _check_reminders():

    now = datetime.now().strftime("%H:%M")

    due = get_pending_reminders_for_time(now)

    for rid, text, category in due:

        key = (rid, now)

        # Prevent repeated speaking in same minute
        if key in _spoken_this_minute:
            continue

        msg = generate_reminder_message(
            text,
            category,
            person=None
        )

        print(f"[REMINDER ENGINE] Triggering: {category} — {text}")

        enqueue(msg)

        insert_log(
            "REMINDER",
            f"Audio triggered: {category}: {text}"
        )

        _pending_ack[rid] = time.time()

        _spoken_this_minute.add(key)


# =========================================================
# THREAD LOOP
# =========================================================

def _runner():

    print("[REMINDER ENGINE] Started")

    while not _stop_event.is_set():

        try:

            _check_reminders()

            # Clear stale minute entries
            current_min = datetime.now().strftime("%H:%M")

            stale = {
                k for k in _spoken_this_minute
                if k[1] != current_min
            }

            _spoken_this_minute.difference_update(stale)

            # Remove expired acknowledgements
            now_ts = time.time()

            expired = [
                rid for rid, ts in _pending_ack.items()
                if now_ts - ts > ACKNOWLEDGE_WINDOW
            ]

            for rid in expired:

                _pending_ack.pop(rid, None)

        except Exception as e:

            print(f"[REMINDER ENGINE ERROR] {e}")

        time.sleep(5)


# =========================================================
# VOICE ACKNOWLEDGEMENT
# =========================================================

def check_voice_acknowledgement(text: str) -> bool:

    keywords = [
        "done",
        "okay",
        "ok",
        "taken",
        "finished",
        "yes",
        "got it"
    ]

    t = text.lower()

    if not any(k in t for k in keywords):
        return False

    acked = False

    for rid in list(_pending_ack.keys()):

        mark_reminder_complete(rid)

        insert_log(
            "VOICE",
            f"Reminder {rid} acknowledged via voice"
        )

        _pending_ack.pop(rid, None)

        acked = True

    return acked


# =========================================================
# START
# =========================================================

def start():

    global _thread

    if _thread and _thread.is_alive():
        return

    _stop_event.clear()

    _thread = threading.Thread(
        target=_runner,
        daemon=True,
        name="ReminderEngine"
    )

    _thread.start()


# =========================================================
# STOP
# =========================================================

def stop():

    _stop_event.set()

    print("[REMINDER ENGINE] Stopped")