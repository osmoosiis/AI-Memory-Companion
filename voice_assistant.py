

import os
import threading
import time
import tempfile
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import whisper

import audio_manager

from context_engine import (
    generate_time_response,
    generate_daily_reminder_summary,
)

from database import insert_log

from reminder_engine import check_voice_acknowledgement




print("[WHISPER] Loading model...")

model = whisper.load_model("base")

print("[WHISPER] Model loaded")




running = True
_thread = None

SAMPLE_RATE     = 16000
RECORD_SECONDS  = 2       # reduced from 3 → faster command cycle
SILENCE_THRESH  = 0.008   # RMS below this = silence, skip transcription




def record_audio() -> np.ndarray:

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    return np.squeeze(audio)




def transcribe(audio: np.ndarray) -> str:

    # Fast energy check — skip silent frames without hitting Whisper
    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < SILENCE_THRESH:
        return ""

    path = None

    try:

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        wav.write(path, SAMPLE_RATE, audio)

        result = model.transcribe(path, language="en", fp16=False)

        text = result["text"].strip()

        # Filter Whisper hallucinations
        junk_phrases = [
            "thank you",
            "thanks for watching",
            "subtitle",
            "subtitles",
            "www.",
            ".com",
        ]

        if any(j in text.lower() for j in junk_phrases):
            return ""

        return text

    except Exception as e:

        print(f"[WHISPER TRANSCRIBE ERROR] {e}")
        return ""

    finally:

        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass



def _respond(
    message: str,
    log_type: str,
    log_desc: str,
    priority: bool = False
):

    print(f"[ASSISTANT] {message}")

    audio_manager.enqueue(message, priority=priority)

    insert_log(log_type, log_desc)




def handle_command(text: str):

    t = text.lower()

    print(f"[VOICE] Heard: {t}")

    # ── Time ──────────────────────────────────────────────────────────────────

    if any(k in t for k in ["time", "clock", "hour"]):

        response = generate_time_response()
        _respond(response, "TIME_QUERY", f"User asked time -> {response}")
        return

    # ── Date ──────────────────────────────────────────────────────────────────

    if any(k in t for k in ["day", "date", "today"]):

        now = datetime.now()
        response = (
            f"Today is {now.strftime('%A')}, "
            f"{now.strftime('%B %d, %Y')}."
        )
        _respond(response, "DATE_QUERY", response)
        return

    # ── Reminders ─────────────────────────────────────────────────────────────

    if any(k in t for k in ["reminder", "schedule", "task"]):

        response = generate_daily_reminder_summary()
        _respond(response, "REMINDER_QUERY", "User checked reminders")
        return

    # ── Location / Safety ─────────────────────────────────────────────────────

    if any(k in t for k in ["where am i", "am i safe", "where is this"]):

        response = "You are at home. You are safe."
        _respond(response, "ORIENTATION_QUERY", response)
        return

    # ── Identity ──────────────────────────────────────────────────────────────

    if any(k in t for k in ["who are you", "your name"]):

        response = "I am CogniCare, your memory companion."
        _respond(response, "IDENTITY_QUERY", response)
        return

    # ── Help / Emergency ──────────────────────────────────────────────────────

    if any(k in t for k in ["help", "emergency", "call"]):

        response = "I am alerting your caregiver now."
        _respond(response, "HELP_REQUEST", response, priority=True)
        return

    # ── Reminder acknowledgement ──────────────────────────────────────────────

    if check_voice_acknowledgement(t):

        response = "Okay. I marked that reminder complete."
        _respond(response, "REMINDER_ACK", response)
        return

    print(f"[VOICE] No command matched: {t}")



def listen_loop():

    global running

    print("[WHISPER] Voice assistant started")

    while running:

        try:

            # Don't listen while speaking
            if audio_manager.speaking():
                time.sleep(0.2)
                continue

            audio = record_audio()

            # Recheck after recording (speech may have started during recording)
            if audio_manager.speaking():
                continue

            text = transcribe(audio)

            if text and len(text.strip()) > 2:
                handle_command(text)

            else:
                # Small sleep to avoid hammering CPU when idle
                time.sleep(0.05)

        except Exception as e:

            print(f"[WHISPER ERROR] {e}")
            time.sleep(1)


def start():

    global _thread, running

    if _thread and _thread.is_alive():
        return

    running = True

    _thread = threading.Thread(
        target=listen_loop,
        daemon=True,
        name="VoiceThread"
    )

    _thread.start()



def stop():

    global running
    running = False