THRESHOLD_HIGH = 0.70   # Confident match — announce person
THRESHOLD_LOW  = 0.60   # Uncertain match — show "Maybe" in UI only, don't announce
THRESHOLD      = 0.72   # Duplicate-check threshold in register_person.py

COOLDOWN_SECONDS    = 5
STABILITY_THRESHOLD = 7

CAMERA_INDEX   = 1        # Default webcam; change to 1 for secondary camera
IP_WEBCAM_URL  = None

DB_PATH = "face.db"

REMINDER_CATEGORIES = ["Medication", "Meal", "Hydration", "Appointment", "Safety", "Orientation", "General"]

# ── Dementia-care timing constants ──────────────────────────────────────────
VISIT_COOLDOWN          = 300   # seconds before re-announcing same person
WHO_IS_THIS_DELAY       = 5     # seconds a face must be stable before whisper
UNKNOWN_ALERT_FRAMES    = 30    # consecutive Unknown frames before safety alert (increased to reduce false positives)
DAILY_SUMMARY_HOUR      = 20    # 8 PM — when the daily recap is spoken