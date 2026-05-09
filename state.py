"""
Global runtime state (NO circular imports anymore)
"""

class State:
    def __init__(self):
        self.last_person = None
        self.last_bbox = None
        self.stable_label = "Unknown"
        self.identity_buffer = []
        self.voice_buffer = ""
        self.speaking = False

state = State()