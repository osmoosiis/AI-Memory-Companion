import asyncio
import edge_tts
import os
import tempfile
import subprocess

VOICE = "en-IN-NeerjaNeural"

async def _generate_audio(text: str, file_path: str):
    communicate = edge_tts.Communicate(text, VOICE, rate="-10%")
    await communicate.save(file_path)

def speak(text: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            path = f.name

        asyncio.run(_generate_audio(text, path))

        # LAG FIX: Added 'nice -n 10' to lower priority and '-nodisp' to save CPU
        subprocess.run(
            ["nice", "-n", "10", "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if os.path.exists(path):
            os.remove(path)

    except Exception as e:
        print(f"[TTS ERROR] {e}")