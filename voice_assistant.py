"""
voice_assistant.py
Whisper-based voice assistant for CogniCare.
"""

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


# =========================================================
# LOAD WHISPER MODEL
# =========================================================

print("[WHISPER] Loading model...")

model = whisper.load_model("base")

print("[WHISPER] Model loaded")


# =========================================================
# GLOBALS
# =========================================================

running = True
_thread = None


# =========================================================
# RECORD AUDIO
# =========================================================

def record_audio(
    duration: float = 3,
    samplerate: int = 16000
) -> np.ndarray:

    print("[VOICE] Listening...")

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = np.squeeze(audio)

    return audio


# =========================================================
# TRANSCRIBE
# =========================================================

def transcribe(audio: np.ndarray, samplerate: int = 16000) -> str:

    # Skip silence
    volume = np.abs(audio).mean()

    if volume < 0.01:
        return ""

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as f:

        path = f.name

    try:

        wav.write(path, samplerate, audio)

        result = model.transcribe(
            path,
            language="en",
            fp16=False
        )

        text = result["text"].strip()

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

    finally:

        if os.path.exists(path):
            os.remove(path)


# =========================================================
# RESPONSE HELPER
# =========================================================

def _respond(
    message: str,
    log_type: str,
    log_desc: str,
    priority: bool = False
):

    print(f"[ASSISTANT] {message}")

    audio_manager.enqueue(
        message,
        priority=priority
    )

    insert_log(log_type, log_desc)


# =========================================================
# COMMAND HANDLER
# =========================================================

def handle_command(text: str):

    t = text.lower()

    print(f"[VOICE] Heard: {t}")

    # ==========================================
    # TIME
    # ==========================================

    if any(k in t for k in ["time", "clock", "hour"]):

        response = generate_time_response()

        _respond(
            response,
            "TIME_QUERY",
            f"User asked time -> {response}"
        )

        return

    # ==========================================
    # DATE
    # ==========================================

    if any(k in t for k in ["day", "date", "today"]):

        now = datetime.now()

        response = (
            f"Today is {now.strftime('%A')}, "
            f"{now.strftime('%B %d, %Y')}."
        )

        _respond(
            response,
            "DATE_QUERY",
            response
        )

        return

    # ==========================================
    # REMINDERS
    # ==========================================

    if any(k in t for k in ["reminder", "schedule", "task"]):

        response = generate_daily_reminder_summary()

        _respond(
            response,
            "REMINDER_QUERY",
            "User checked reminders"
        )

        return

    # ==========================================
    # SAFETY
    # ==========================================

    if any(k in t for k in [
        "where am i",
        "am i safe",
        "where is this"
    ]):

        response = (
            "You are at home. "
            "You are safe."
        )

        _respond(
            response,
            "ORIENTATION_QUERY",
            response
        )

        return

    # ==========================================
    # IDENTITY
    # ==========================================

    if any(k in t for k in [
        "who are you",
        "your name"
    ]):

        response = (
            "I am CogniCare, "
            "your memory companion."
        )

        _respond(
            response,
            "IDENTITY_QUERY",
            response
        )

        return

    # ==========================================
    # HELP
    # ==========================================

    if any(k in t for k in [
        "help",
        "emergency",
        "call"
    ]):

        response = (
            "I am alerting your caregiver now."
        )

        _respond(
            response,
            "HELP_REQUEST",
            response,
            priority=True
        )

        return

    # ==========================================
    # REMINDER ACK
    # ==========================================

    from reminder_engine import check_voice_acknowledgement

    if check_voice_acknowledgement(t):

        response = (
            "Okay. "
            "I marked that reminder complete."
        )

        _respond(
            response,
            "REMINDER_ACK",
            response
        )

        return

    print(f"[VOICE] No command match: {t}")


# =========================================================
# LISTEN LOOP
# =========================================================

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

            # Recheck after recording
            if audio_manager.speaking():

                continue

            text = transcribe(audio)

            if text and len(text.strip()) > 2:

                handle_command(text)

        except Exception as e:

            print(f"[WHISPER ERROR] {e}")

            time.sleep(1)


# =========================================================
# START
# =========================================================

def start():

    global _thread
    global running

    if _thread and _thread.is_alive():
        return

    running = True

    _thread = threading.Thread(
        target=listen_loop,
        daemon=True,
        name="VoiceThread"
    )

    _thread.start()


# =========================================================
# STOP
# =========================================================

def stop():

    global running

    running = False