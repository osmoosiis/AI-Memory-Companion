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


#global var

_stop_event = threading.Event()

_spoken_this_minute: set = set()

_pending_ack: dict = {}   # {reminder_id: timestamp_when_spoken}

_thread = None

ACKNOWLEDGE_WINDOW = 120  # seconds the user has to voice-acknowledge


# CHECK REMINDERS

def _check_reminders():

    now = datetime.now().strftime("%H:%M")

    due = get_pending_reminders_for_time(now)

    for rid, text, category in due:

        key = (rid, now)

        # Prevent repeated speaking within the same minute
        if key in _spoken_this_minute:
            continue

        msg = generate_reminder_message(text, category, person=None)

        print(f"[REMINDER ENGINE] Triggering: {category} — {text}")

        # force=True ensures this is never silently dropped by cooldown
        enqueue(msg, force=True)

        insert_log("REMINDER", f"Audio triggered: {category}: {text}")

        _pending_ack[rid] = time.time()

        _spoken_this_minute.add(key)


# THREAD LOOP

def _runner():

    print("[REMINDER ENGINE] Started")

    while not _stop_event.is_set():

        try:

            _check_reminders()

            # Clear stale minute-keyed entries
            current_min = datetime.now().strftime("%H:%M")

            stale = {k for k in _spoken_this_minute if k[1] != current_min}

            _spoken_this_minute.difference_update(stale)

            # Expire unacknowledged reminders
            now_ts  = time.time()
            expired = [
                rid for rid, ts in _pending_ack.items()
                if now_ts - ts > ACKNOWLEDGE_WINDOW
            ]

            for rid in expired:
                _pending_ack.pop(rid, None)

        except Exception as e:

            print(f"[REMINDER ENGINE ERROR] {e}")

        time.sleep(5)


#voice acknowledgement

def check_voice_acknowledgement(text: str) -> bool:
    """
    If the user says an acknowledgement keyword, mark the OLDEST
    pending reminder as complete (one at a time — not all at once).
    Returns True if a reminder was acknowledged.
    """

    keywords = ["done", "okay", "ok", "taken", "finished", "yes", "got it", "complete"]

    if not any(k in text.lower() for k in keywords):
        return False

    if not _pending_ack:
        return False

    # Mark only the oldest unacknowledged reminder
    oldest_rid = min(_pending_ack, key=_pending_ack.get)

    mark_reminder_complete(oldest_rid)

    insert_log("VOICE", f"Reminder {oldest_rid} acknowledged via voice")

    _pending_ack.pop(oldest_rid, None)

    return True



# START


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



def stop():

    _stop_event.set()

    print("[REMINDER ENGINE] Stopped")