"""
audio_manager.py — Thread-safe, queue-based speech system.
"""
import threading
import time
from queue import Queue, Empty
from person_audio import speak as _speak

# Internal state
_speech_queue = Queue()
_stop_event = threading.Event()
_worker_thread = None

# Global Speaking Lock (Mutes the mic when the AI talks)
is_speaking = False
_last_message = ""
_last_time = 0
MESSAGE_COOLDOWN = 4 

def speaking():
    """Returns the current speaking status."""
    global is_speaking
    return is_speaking

def _worker():
    """Background worker that pulls messages from the queue and speaks them."""
    global is_speaking
    while not _stop_event.is_set():
        try:
            # Check for new messages every 0.5 seconds
            message = _speech_queue.get(timeout=0.5)
            if message is None:
                break

            is_speaking = True
            try:
                print(f"[TTS] {message}")
                _speak(message)
            except Exception as e:
                print(f"[AUDIO MANAGER] Speech Error: {e}")
            finally:
                # Hardware buffer: wait a moment after speaking before opening the mic
                time.sleep(0.4)
                is_speaking = False
                _speech_queue.task_done()

        except Empty:
            continue
        except Exception as e:
            is_speaking = False
            print(f"[AUDIO MANAGER] Critical Worker Error: {e}")

def start():
    """Initializes the audio worker thread."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _stop_event.clear()
    _worker_thread = threading.Thread(target=_worker, daemon=True, name="AudioWorker")
    _worker_thread.start()
    print("[AUDIO MANAGER] Started")

def enqueue(message: str, priority: bool = False):
    """Adds a message to the speech queue with a cooldown filter."""
    global _last_message, _last_time
    if not message:
        return

    # Prevent the AI from saying the same thing repeatedly within the cooldown period
    current_time = time.time()
    if message == _last_message and current_time - _last_time < MESSAGE_COOLDOWN:
        return

    _last_message = message
    _last_time = current_time

    # Priority clear: stops current queue if an emergency or high-priority alert occurs
    if priority:
        with _speech_queue.mutex:
            _speech_queue.queue.clear()

    _speech_queue.put(message)

def speak(message: str):
    """Bridge function for voice_assistant.py to access the queue."""
    enqueue(message)

def stop():
    """Stops the audio manager worker."""
    global _stop_event
    _stop_event.set()
    if _worker_thread:
        _worker_thread.join()