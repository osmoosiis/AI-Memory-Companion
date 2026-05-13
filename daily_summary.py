"""
daily_summary.py — Warm natural-language recap of the day.

Fix vs original:
  - get_today_summary() now exists in database.py (was missing, causing ImportError).
  - Uses enqueue() instead of a missing 'enqueue' import path.
"""

from datetime import datetime
from database import get_today_summary
from audio_manager import enqueue


def generate_summary_text() -> str:
    logs = get_today_summary()
    if not logs:
        return "No activity has been recorded today yet."

    recognitions = [l for l in logs if l[2] == "RECOGNITION"]
    reminders    = [l for l in logs if l[2] == "REMINDER"]
    commands     = [l for l in logs if l[2] in ("VOICE_COMMAND", "TIME_QUERY", "DATE_QUERY",
                                                  "REMINDER_QUERY", "ORIENTATION_QUERY")]
    alerts       = [l for l in logs if l[2] in ("UNKNOWN_ALERT", "CRITICAL_ALERT")]

    parts = []

    # ── People seen ───────────────────────────────────────────────────────────
    if recognitions:
        seen_with_times: dict = {}
        for l in recognitions:
            raw  = l[3].replace("Recognised: ", "")
            name = raw.split(" (")[0]
            try:
                hour = int(l[1][11:13])
            except (IndexError, ValueError):
                hour = 12
            seen_with_times.setdefault(name, []).append(hour)

        people_parts = []
        for name, hours in seen_with_times.items():
            count = len(hours)
            avg_h = sum(hours) // len(hours)
            period = (
                "this morning"   if avg_h < 12 else
                "this afternoon" if avg_h < 17 else
                "this evening"
            )
            if count == 1:
                people_parts.append(f"{name} {period}")
            else:
                people_parts.append(f"{name} {count} times")

        if len(people_parts) == 1:
            parts.append(f"Today you saw {people_parts[0]}.")
        elif len(people_parts) == 2:
            parts.append(f"Today you saw {people_parts[0]} and {people_parts[1]}.")
        else:
            joined = ", ".join(people_parts[:-1])
            parts.append(f"Today you saw {joined}, and {people_parts[-1]}.")

    # ── Reminders ─────────────────────────────────────────────────────────────
    if reminders:
        med_rems  = [l for l in reminders if "Medication" in l[3]]
        meal_rems = [l for l in reminders if "Meal"       in l[3]]
        other     = len(reminders) - len(med_rems) - len(meal_rems)

        if med_rems:
            if len(med_rems) == 1:
                parts.append(f"You took your medication at {med_rems[0][1][11:16]}.")
            else:
                parts.append(f"You completed {len(med_rems)} medication reminders.")

        if meal_rems:
            parts.append(
                f"You had {len(meal_rems)} meal reminder{'s' if len(meal_rems) > 1 else ''} today."
            )
        if other > 0:
            parts.append(
                f"You completed {other} other reminder{'s' if other > 1 else ''} today."
            )
    else:
        parts.append("You have not completed any reminders today.")

    # ── Voice interactions ────────────────────────────────────────────────────
    if commands:
        parts.append(
            f"You asked me {len(commands)} question{'s' if len(commands) > 1 else ''} today."
        )

    # ── Security ─────────────────────────────────────────────────────────────
    if alerts:
        parts.append(
            f"Note: {len(alerts)} unknown visitor "
            f"alert{'s were' if len(alerts) > 1 else ' was'} recorded today."
        )

    # ── Closing ──────────────────────────────────────────────────────────────
    hour = datetime.now().hour
    if hour >= 20:
        parts.append("I hope you had a good day. Rest well tonight.")
    elif hour >= 17:
        parts.append("Good evening. Your day is nearly done.")

    return " ".join(parts)


def speak_summary():
    text = generate_summary_text()
    print(f"[SUMMARY] {text}")
    enqueue(text)
    return text