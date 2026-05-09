import threading
import time
from datetime import datetime
from database import get_pending_reminders_for_time, mark_reminder_complete, insert_log
from audio_manager import enqueue
from context_engine import generate_reminder_message

_stop_event = threading.Event()
_spoken_this_minute = set()
_pending_ack = {}
ACKNOWLEDGE_WINDOW = 30 

def _check_reminders():
    now = datetime.now().strftime("%H:%M")
    due = get_pending_reminders_for_time(now)

    for rid, text, category in due:
        key = (rid, now)
        if key in _spoken_this_minute:
            continue

        msg = generate_reminder_message(text, category, None)
        print(f"[REMINDER ENGINE] Triggering: {category} - {text}")
        
        enqueue(msg)
        insert_log("REMINDER", f"Audio triggered: {category}: {text}")

        _pending_ack[rid] = time.time()
        _spoken_this_minute.add(key)

def _runner():
    while not _stop_event.is_set():
        try:
            _check_reminders()
            # Clear old "spoken" keys from previous minutes
            current_min = datetime.now().strftime("%H:%M")
            to_remove = [k for k in _spoken_this_minute if k[1] != current_min]
            for k in to_remove:
                _spoken_this_minute.remove(k)
        except Exception as e:
            print(f"[REMINDER ENGINE ERROR] {e}")
        time.sleep(5)

def check_voice_acknowledgement(text):
    """
    Checks if the user said something like 'done' or 'okay' 
    to stop a pending reminder.
    """
    text = text.lower()
    # Simple check for acknowledgement keywords
    keywords = ["done", "okay", "ok", "taken", "finished"]
    
    if any(word in text for word in keywords):
        # If there are pending reminders in our local tracker, mark them done
        for rid in list(_pending_ack.keys()):
            mark_reminder_complete(rid)
            insert_log("VOICE", f"Reminder {rid} acknowledged via voice.")
            _pending_ack.pop(rid, None)
            return True
    return False

def start():
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    print("[REMINDER ENGINE] Started")