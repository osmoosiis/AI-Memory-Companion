"""
person_audio.py
Edge-TTS synthesis + ffplay playback.
"""

import asyncio
import os
import subprocess
import tempfile

import edge_tts

VOICE = "en-IN-NeerjaNeural"


async def _generate_audio(text: str, file_path: str):
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate="-10%"
    )

    await communicate.save(file_path)


def speak(text: str):
    """
    Synthesise and play audio synchronously.
    Called from AudioWorker thread.
    """

    if not text or not text.strip():
        return

    path = None

    try:
        print(f"[TTS] Generating: {text}")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as f:
            path = f.name

        # Generate MP3 using Edge-TTS
        asyncio.run(_generate_audio(text, path))

        # Play audio
        subprocess.run(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-af",
                "volume=1.0",
                path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print("[TTS] Playback finished")

    except Exception as e:
        print(f"[TTS ERROR] {e}")

    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)

        except Exception as e:
            print(f"[TTS CLEANUP ERROR] {e}")