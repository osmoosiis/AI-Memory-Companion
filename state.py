"""
state.py — Global runtime state shared across all modules.
No circular imports; pure data container.
"""
import time


class State:
    def __init__(self):
        self.last_person         = None          # name of last announced person
        self.last_bbox           = None
        self.stable_label        = "Unknown"
        self.identity_buffer     = []
        self.voice_buffer        = ""
        self.speaking            = False

        # ── "Who Is This?" timer (Objective 1) ──────────────────────────────
        # Tracks how long the current face has been continuously stable
        self.face_stable_since   = None          # float (time.time()) or None
        self.who_whispered       = set()         # names already whispered this session

        # ── Unknown-person safety tracking ──────────────────────────────────
        self.unknown_consecutive = 0             # frames labelled Unknown in a row
        self.unknown_alerted     = False         # prevent repeated unknown alerts

        # ── Session stats ────────────────────────────────────────────────────
        self.session_start       = time.time()


state = State()