"""
voice_assistant.py — Whisper-based listener and command handler.
"""
import os
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import threading
import time
from datetime import datetime

# Local imports
import audio_manager
from database import insert_log

# Initialize the Whisper model
# "small" is a good balance for Pop!_OS performance
model = whisper.load_model("small")

running = True
_thread = None

def record_audio(duration=3, samplerate=16000):
    """Records a short snippet of audio from the microphone."""
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return np.squeeze(audio)

def transcribe(audio, samplerate=16000):
    """Converts recorded audio to text using Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        wav.write(path, samplerate, audio)
        # Using language="en" speeds up the inference significantly
        result = model.transcribe(path, language="en", fp16=False)
        text = result["text"].strip()
        
        # Hallucination filter for silent rooms
        if any(h in text.lower() for h in ["thank you", "watching", "ご視聴", "subtitle"]):
            return ""
            
        return text
    finally:
        if os.path.exists(path):
            os.remove(path)

def handle_command(text):
    """Processes the transcribed text into actions."""
    text = text.lower()
    
    # ─── LOGIC: DATE AND DAY ───
    if any(k in text for k in ["day", "date", "today"]):
        now = datetime.now()
        day_name = now.strftime("%A") # Saturday
        date_str = now.strftime("%B %d") # May 09
        
        response = f"Today is {day_name}, {date_str}."
        print(f"[ASSISTANT] Responding: {response}")
        
        insert_log("VOICE_COMMAND", f"Date Inquiry: {response}")
        audio_manager.speak(response)
        return

    elif "reminder" in text:
        audio_manager.speak("I have listed your reminders on the dashboard for you.")
        insert_log("VOICE_COMMAND", "User checked reminders.")
        return
    
    # Add a fallback for names (e.g., "Who are you?")
    elif "who are you" in text or "your name" in text:
        audio_manager.speak("I am your memory companion, here to help you.")
        return

    else:
        print(f"[ASSISTANT] No command match for: {text}")

def listen_loop():
    """Main loop that continuously listens when the AI is not speaking."""
    global running
    print("[WHISPER] Voice assistant service started")
    
    last_speech_time = 0

    while running:
        try:
            # Check if the AI is currently talking via the audio_manager flag
            if audio_manager.is_speaking:
                last_speech_time = time.time()
                time.sleep(0.5)
                continue

            # Muzzle Logic: Wait 2 seconds after the AI finishes speaking before listening
            if time.time() - last_speech_time < 2.0:
                time.sleep(0.1)
                continue

            # Capture audio
            audio = record_audio()
            
            # Final check before transcription to ensure silence
            if audio_manager.is_speaking:
                continue

            text = transcribe(audio).strip()

            if len(text) > 3:
                print(f"[WHISPER RAW] {text}")
                handle_command(text)

        except Exception as e:
            print(f"[WHISPER ERROR] {e}")
            time.sleep(1)

def start():
    """Starts the voice assistant in a background thread."""
    global _thread, running
    if _thread and _thread.is_alive():
        return
    running = True
    _thread = threading.Thread(target=listen_loop, daemon=True)
    _thread.start()

def stop():
    """Stops the voice assistant."""
    global running
    running = False