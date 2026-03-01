 
import pyttsx3

_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 180)  
        _engine.setProperty('volume', 1.0)
    return _engine

def speak(text: str):
    """Speak the given text aloud (blocking)."""
    engine = _get_engine()
    engine.say(text)
    engine.runAndWait()