"""
context_engine.py — Contextual message generator.
Combines recognized person + time + reminders + last-seen memory
to produce smart, human-sounding assistance messages.
"""

from datetime import datetime
from database import get_all_reminders


# ── Time helpers ──────────────────────────────────────────────────────────────

def _time_of_day() -> str:
    h = datetime.now().hour
    if 5  <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"


def _format_last_seen(last_seen_str: str | None) -> str | None:
    """Convert ISO date string to friendly phrase."""
    if not last_seen_str:
        return None
    try:
        from datetime import date
        last = datetime.fromisoformat(last_seen_str).date()
        today = date.today()
        delta = (today - last).days
        if delta == 0:
            return "earlier today"
        if delta == 1:
            return "yesterday"
        if delta < 7:
            return f"{delta} days ago"
        if delta < 14:
            return "last week"
        return last.strftime("%B %d")
    except Exception:
        return None


# ── Reminder helpers ──────────────────────────────────────────────────────────

def _get_pending_reminders() -> list:
    rows = get_all_reminders()
    return [r for r in rows if r[4] == 0]


def _nearest_reminder(pending: list) -> tuple | None:
    if not pending:
        return None
    now = datetime.now().strftime("%H:%M")
    upcoming = sorted(pending, key=lambda r: abs(
        _time_diff_minutes(now, r[2])
    ))
    return upcoming[0] if upcoming else None


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


# ── Main generator ────────────────────────────────────────────────────────────

def generate_recognition_message(person: dict) -> str:
    """
    Build a rich contextual greeting when a person is recognised.

    person dict keys: name, relationship, reminder, last_seen, visit_count
    """
    name         = person.get("name", "someone")
    relationship = person.get("relationship", "visitor")
    reminder_msg = person.get("reminder", "")
    last_seen    = _format_last_seen(person.get("last_seen"))
    visit_count  = person.get("visit_count", 0)
    tod          = _time_of_day()

    pending   = _get_pending_reminders()
    nearest   = _nearest_reminder(pending)
    med_due   = next((r for r in pending if r[3] == "Medication"), None)
    appt_due  = next((r for r in pending if r[3] == "Appointment"), None)

    # ── Build greeting ────────────────────────────────────────────────────────
    greeting = f"Good {tod}. This is {name}, your {relationship}."

    # ── Last-seen context ─────────────────────────────────────────────────────
    if last_seen:
        if last_seen == "earlier today":
            greeting += f" You saw {name} earlier today."
        else:
            greeting += f" You last saw {name} {last_seen}."

    # ── Visit frequency context ───────────────────────────────────────────────
    if visit_count and visit_count > 5:
        greeting += f" {name} visits you often."

    # ── Personal reminder ─────────────────────────────────────────────────────
    if reminder_msg:
        greeting += f" {reminder_msg}."

    # ── Medication context — highest priority ─────────────────────────────────
    if med_due:
        med_text = med_due[1]
        if _is_overdue(med_due[2]):
            greeting += (f" Your {med_text} was due at {med_due[2]} and is still pending. "
                         f"{name} can help you with that.")
        else:
            greeting += f" Your {med_text} is coming up at {med_due[2]}."

    # ── Appointment context ───────────────────────────────────────────────────
    elif appt_due:
        greeting += f" You have an appointment: {appt_due[1]} at {appt_due[2]}."

    # ── Generic nearest reminder ──────────────────────────────────────────────
    elif nearest and nearest[3] != "Medication":
        diff = _time_diff_minutes(datetime.now().strftime("%H:%M"), nearest[2])
        if diff <= 30:
            greeting += f" Coming up soon: {nearest[1]}."

    return greeting.strip()


def generate_unknown_alert(consecutive_frames: int) -> str:
    if consecutive_frames < 3:
        return "An unrecognised person is nearby."
    return ("An unknown individual has been detected. "
            "If you do not recognise this person, please press the help button "
            "or ask a caregiver.")


def generate_reminder_message(reminder_text: str, category: str, person: dict | None) -> str:
    """Contextual reminder — richer if a known person is present."""
    base = {
        "Medication":   f"It is time for your {reminder_text}.",
        "Meal":         f"It is time for your meal. {reminder_text}.",
        "Hydration":    "Please drink some water to stay hydrated.",
        "Appointment":  f"Reminder: you have {reminder_text}.",
        "General":      reminder_text,
    }.get(category, reminder_text)

    if person and person.get("name"):
        name = person["name"]
        rel  = person.get("relationship", "visitor")
        if category == "Medication":
            base += f" Your {rel} {name} is here and can help you."
        elif category == "Appointment":
            base += f" {name} may be able to accompany you."

    return base


def generate_time_response() -> str:
    now = datetime.now()
    t   = now.strftime("%I:%M %p").lstrip("0")
    tod = _time_of_day()
    return f"It is {t}, {tod}."


def generate_daily_reminder_summary() -> str:
    pending = _get_pending_reminders()
    if not pending:
        return "You have no pending reminders for today. Well done."
    if len(pending) == 1:
        r = pending[0]
        return f"You have one reminder left: {r[1]} at {r[2]}."
    return (f"You have {len(pending)} reminders remaining today. "
            f"The next one is {pending[0][1]} at {pending[0][2]}.")