"""
context_engine.py — Contextual message generator for CogniCare AI.

Improvements vs original:
  - generate_recognition_message() follows the 4-part dementia-friendly
    structure: Hook → Context → Instruction → Confirmation cue.
  - generate_reminder_message() uses category-specific, single-verb
    instructions (Medication, Meal, Hydration, Safety, Orientation, ADL).
  - New helpers: generate_orientation_message(), generate_safety_alert().
  - All existing public functions preserved for backwards compatibility.
"""

from datetime import datetime, date
from database import get_all_reminders


# ── Time helpers ──────────────────────────────────────────────────────────────

def _time_of_day() -> str:
    h = datetime.now().hour
    if 5  <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"


def _format_last_seen(last_seen_str) -> str | None:
    if not last_seen_str:
        return None
    try:
        last  = datetime.fromisoformat(last_seen_str).date()
        today = date.today()
        delta = (today - last).days
        if delta == 0:  return "earlier today"
        if delta == 1:  return "yesterday"
        if delta < 7:   return f"{delta} days ago"
        if delta < 14:  return "last week"
        return last.strftime("%B %d")
    except Exception:
        return None


# ── Reminder helpers ──────────────────────────────────────────────────────────

def _get_pending_reminders() -> list:
    return [r for r in get_all_reminders() if r[4] == 0]


def _nearest_reminder(pending: list):
    if not pending:
        return None
    now = datetime.now().strftime("%H:%M")
    return sorted(pending, key=lambda r: abs(_time_diff_minutes(now, r[2])))[0]


def _time_diff_minutes(t1: str, t2: str) -> int:
    try:
        h1, m1 = map(int, t1.split(":"))
        h2, m2 = map(int, t2.split(":"))
        return abs((h1 * 60 + m1) - (h2 * 60 + m2))
    except Exception:
        return 999


def _is_overdue(reminder_time: str) -> bool:
    now = datetime.now().strftime("%H:%M")
    h1, m1 = map(int, now.split(":"))
    h2, m2 = map(int, reminder_time.split(":"))
    return (h1 * 60 + m1) > (h2 * 60 + m2)


# ── Recognition message (4-part dementia-friendly structure) ─────────────────

def generate_recognition_message(person: dict) -> str:
    """
    Builds a contextual greeting following the dementia-care structure:
      1. Hook     — gentle attention-getter with time-of-day
      2. Context  — who the person is and when you last saw them
      3. Instruction — personal reminder or upcoming task (if any)
      4. Confirmation cue — reassurance phrase
    """
    name         = person.get("name", "someone")
    relationship = person.get("relationship", "visitor")
    reminder_msg = person.get("reminder", "")
    last_seen    = _format_last_seen(person.get("last_seen"))
    visit_count  = person.get("visit_count", 0) or 0
    tod          = _time_of_day()

    pending  = _get_pending_reminders()
    med_due  = next((r for r in pending if r[3] == "Medication"),    None)
    appt_due = next((r for r in pending if r[3] == "Appointment"),   None)
    nearest  = _nearest_reminder(pending)

    # 1. HOOK
    parts = [f"Good {tod}."]

    # 2. CONTEXT — who + relationship + last seen
    if last_seen and last_seen != "earlier today":
        parts.append(
            f"The person in front of you is {name}, your {relationship}. "
            f"You last saw {name} {last_seen}."
        )
    elif last_seen == "earlier today":
        parts.append(
            f"This is {name}, your {relationship}. You already saw {name} earlier today."
        )
    else:
        parts.append(f"This is {name}, your {relationship}.")

    if visit_count and visit_count > 5:
        parts.append(f"{name} visits you often and cares about you.")

    # 3. INSTRUCTION — personal reminder + any urgent health task
    if reminder_msg:
        parts.append(reminder_msg.rstrip(".") + ".")

    if med_due:
        med_text = med_due[1]
        if _is_overdue(med_due[2]):
            parts.append(
                f"Your {med_text} was due at {med_due[2]} and has not been taken yet. "
                f"{name} can help you with that now."
            )
        else:
            parts.append(
                f"Please remember: your {med_text} is coming up at {med_due[2]}."
            )
    elif appt_due:
        parts.append(
            f"You have an appointment — {appt_due[1]} — at {appt_due[2]}. "
            f"{name} may be able to accompany you."
        )
    elif nearest and nearest[3] not in ("Medication", "Appointment"):
        diff = _time_diff_minutes(datetime.now().strftime("%H:%M"), nearest[2])
        if diff <= 30:
            parts.append(f"Coming up soon: {nearest[1]} at {nearest[2]}.")

    # 4. CONFIRMATION CUE
    parts.append("You are safe and at home.")

    return " ".join(parts)


# ── "Who Is This?" whisper (5-second stable-face feature) ────────────────────

def generate_who_is_this_whisper(person: dict) -> str:
    """
    Short, gentle whisper triggered after the face has been stable for 5 s.
    Keeps it very brief so it doesn't overwhelm the patient.
    """
    name         = person.get("name", "someone")
    relationship = person.get("relationship", "visitor")
    reminder_msg = person.get("reminder", "")

    msg = f"Just so you know — this is {name}, your {relationship}."
    if reminder_msg:
        msg += f" {reminder_msg.rstrip('.')}."
    return msg


# ── Reminder message (category-specific, single-verb instructions) ────────────

def generate_reminder_message(reminder_text: str, category: str,
                               person: dict | None = None) -> str:
    """
    Dementia-friendly reminder following: Hook → Context → Instruction → Confirmation.
    Uses single, clear verbs and includes the current time for orientation.
    """
    now_str = datetime.now().strftime("%I:%M %p").lstrip("0")
    tod     = _time_of_day()

    templates = {
        "Medication": (
            f"It is {now_str}. "
            f"Time to take your {reminder_text}. "
            f"Please take your medicine now and drink a full glass of water."
        ),
        "Meal": (
            f"It is {now_str}, {tod}. "
            f"Time for your meal. {reminder_text}. "
            f"Please go to the kitchen and sit down to eat."
        ),
        "Hydration": (
            f"You have not had water in a while. "
            f"Please drink some water now to stay hydrated."
        ),
        "Appointment": (
            f"Reminder: you have {reminder_text}. "
            f"Please prepare to leave soon."
        ),
        "Safety": (
            f"Safety reminder: {reminder_text}. "
            f"Please take care of this now."
        ),
        "Orientation": (
            f"It is {now_str} on {datetime.now().strftime('%A')}. "
            f"{reminder_text}. You are at home and everything is safe."
        ),
        "General": reminder_text,
    }

    base = templates.get(category, reminder_text)

    # If a known person is present, personalise
    if person and person.get("name"):
        name = person["name"]
        rel  = person.get("relationship", "visitor")
        if category == "Medication":
            base += f" Your {rel}, {name}, is here and can help you."
        elif category == "Appointment":
            base += f" {name} may be able to take you."

    return base


# ── Unknown person safety alert ───────────────────────────────────────────────

def generate_unknown_alert(consecutive_frames: int = 1) -> str:
    if consecutive_frames < 3:
        return (
            "There is someone nearby who I do not recognise. "
            "Please ask them who they are, or press the help button."
        )
    return (
        "Warning: an unknown person has been near the camera for a while. "
        "If you do not know this person, please press the help button "
        "or call for a caregiver immediately."
    )


# ── Orientation helpers ───────────────────────────────────────────────────────

def generate_orientation_message() -> str:
    """Daily orientation message — good to play on startup or on request."""
    now = datetime.now()
    tod = _time_of_day()
    return (
        f"Good {tod}. Today is {now.strftime('%A, %B %d, %Y')}. "
        f"It is {now.strftime('%I:%M %p').lstrip('0')}. "
        f"You are at home and you are safe."
    )


# ── Safety alert ──────────────────────────────────────────────────────────────

def generate_safety_alert(alert_text: str) -> str:
    return (
        f"Safety reminder: {alert_text}. "
        f"Please be careful and take action now."
    )


# ── Time / reminder summary (used by voice_assistant.py) ─────────────────────

def generate_time_response() -> str:
    now = datetime.now()
    t   = now.strftime("%I:%M %p").lstrip("0")
    tod = _time_of_day()
    day = now.strftime("%A, %B %d")
    return f"It is {t}, {tod}. Today is {day}."


def generate_daily_reminder_summary() -> str:
    pending = _get_pending_reminders()
    if not pending:
        return "You have no pending reminders for today. Well done!"
    if len(pending) == 1:
        r = pending[0]
        return f"You have one reminder left: {r[1]} at {r[2]}."
    return (
        f"You have {len(pending)} reminders remaining today. "
        f"The next one is {pending[0][1]} at {pending[0][2]}."
    )