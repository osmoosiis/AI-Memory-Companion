import pyttsx3

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()

        # ✅ better voice tuning
        _engine.setProperty('rate', 140)
        _engine.setProperty('volume', 1.0)

        voices = _engine.getProperty('voices')
        if len(voices) > 1:
            _engine.setProperty('voice', voices[1].id)  # usually female voice

    return _engine


def speak(text: str):
    """Speak the given text aloud."""
    engine = _get_engine()
    engine.say(text)
    engine.runAndWait()