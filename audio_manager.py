"""
audio_manager.py
Thread-safe queue-based speech manager.
"""

import threading
import time
from queue import Queue, Empty

from person_audio import speak as _speak


# =====================================================
# INTERNAL STATE
# =====================================================

_speech_queue = Queue()

_stop_event = threading.Event()

_worker_thread = None

_speak_lock = threading.Lock()

# Used by voice assistant to mute mic during speech
is_speaking = False

# Message cooldown tracking
_cooldown_map = {}

MESSAGE_COOLDOWN = 10


# =====================================================
# STATUS
# =====================================================

def speaking() -> bool:
    return is_speaking


# =====================================================
# WORKER THREAD
# =====================================================

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


# =====================================================
# START
# =====================================================

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


# =====================================================
# ENQUEUE
# =====================================================

def enqueue(message: str, priority: bool = False):

    if not message:
        return

    now = time.time()

    last = _cooldown_map.get(message, 0)

    # Skip duplicates during cooldown
    if now - last < MESSAGE_COOLDOWN:
        print("[AUDIO MANAGER] Cooldown skip")
        return

    _cooldown_map[message] = now

    # Clear queue for priority speech
    if priority:
        with _speech_queue.mutex:
            _speech_queue.queue.clear()

    _speech_queue.put(message)


# =====================================================
# NORMAL SPEAK
# =====================================================

def speak(message: str):

    enqueue(message)


# =====================================================
# DIRECT SPEAK
# =====================================================

def speak_direct(message: str):

    global is_speaking

    if not message:
        return

    # Clear pending queue
    with _speech_queue.mutex:
        _speech_queue.queue.clear()

    # Wait for current speech to finish
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


# =====================================================
# STOP
# =====================================================

def stop():

    global _worker_thread

    _stop_event.set()

    try:
        _speech_queue.put(None)

    except:
        pass

    if _worker_thread:
        _worker_thread.join(timeout=3)

    print("[AUDIO MANAGER] Stopped")