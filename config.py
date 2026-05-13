THRESHOLD_HIGH = 0.62
THRESHOLD_LOW  = 0.55
THRESHOLD      = 0.72   # for register_person duplicate check

COOLDOWN_SECONDS    = 5
STABILITY_THRESHOLD = 7

CAMERA_INDEX   = 1         # FIX: default to 0 (most systems); was 1
IP_WEBCAM_URL  = None

DB_PATH = "face.db"

REMINDER_CATEGORIES = ["Medication", "Meal", "Hydration", "Appointment", "Safety", "Orientation", "General"]

# ── Dementia-care timing constants ──────────────────────────────────────────
VISIT_COOLDOWN          = 300   # seconds before re-announcing same person
WHO_IS_THIS_DELAY       = 5     # seconds a face must be stable before whisper
UNKNOWN_ALERT_FRAMES    = 3     # consecutive Unknown frames before safety alert
DAILY_SUMMARY_HOUR      = 20    # 8 PM — when the daily recap is spoken