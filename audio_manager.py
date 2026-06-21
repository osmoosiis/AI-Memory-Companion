import threading
import time
from queue import Queue, Empty

from person_audio import speak as _speak


_speech_queue = Queue()

_stop_event = threading.Event()

_worker_thread = None

_speak_lock = threading.Lock()

# Used by voice assistant to mute mic during speech
is_speaking = False

# Message cooldown tracking
_cooldown_map = {}

MESSAGE_COOLDOWN = 60   # seconds — raised from 10 to prevent silently dropping reminders



def speaking() -> bool:
    return is_speaking



def _worker():

    global is_speaking

    print("[AUDIO MANAGER] Worker started")

    while not _stop_event.is_set():

        try:
            message = _speech_queue.get(timeout=0.5)

            if message is None:
                break

            with _speak_lock:
                is_speaking = True

            try:
                print(f"[AUDIO MANAGER] Speaking: {message}")
                _speak(message)

            except Exception as e:
                print(f"[AUDIO MANAGER] Speech Error: {e}")

            finally:

                time.sleep(0.3)

                with _speak_lock:
                    is_speaking = False

                _speech_queue.task_done()

        except Empty:
            continue

        except Exception as e:

            print(f"[AUDIO MANAGER] Worker Crash: {e}")

            with _speak_lock:
                is_speaking = False



def start():

    global _worker_thread

    if _worker_thread and _worker_thread.is_alive():
        return

    _stop_event.clear()

    _worker_thread = threading.Thread(
        target=_worker,
        daemon=True,
        name="AudioWorker"
    )

    _worker_thread.start()

    print("[AUDIO MANAGER] Started")



def enqueue(message: str, priority: bool = False, force: bool = False):
    """
    Add a message to the speech queue.

    Args:
        message:  Text to speak.
        priority: If True, clears the queue before adding (urgent messages).
        force:    If True, bypasses the cooldown check (use for reminders).
    """

    if not message or not message.strip():
        return

    now  = time.time()
    last = _cooldown_map.get(message, 0)

    # Skip duplicates during cooldown unless forced
    if not force and (now - last < MESSAGE_COOLDOWN):
        print(f"[AUDIO MANAGER] Cooldown skip: {message[:40]}...")
        return

    _cooldown_map[message] = now

    # Clear queue for priority speech
    if priority:
        with _speech_queue.mutex:
            _speech_queue.queue.clear()

    _speech_queue.put(message)




def speak(message: str):
    enqueue(message)



def speak_direct(message: str):

    global is_speaking

    if not message or not message.strip():
        return

    # Clear pending queue
    with _speech_queue.mutex:
        _speech_queue.queue.clear()

    # Wait for current speech to finish (max 15s)
    waited = 0.0

    while is_speaking and waited < 15:
        time.sleep(0.1)
        waited += 0.1

    with _speak_lock:
        is_speaking = True

    try:

        print(f"[AUDIO DIRECT] {message}")
        _speak(message)

    except Exception as e:

        print(f"[AUDIO MANAGER] Direct Speech Error: {e}")

    finally:

        with _speak_lock:
            is_speaking = False




def stop():

    global _worker_thread

    _stop_event.set()

    try:
        _speech_queue.put(None)
    except Exception:
        pass

    if _worker_thread:
        _worker_thread.join(timeout=3)

    print("[AUDIO MANAGER] Stopped")