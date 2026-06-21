import json
import cv2 as cv
import numpy as np
import streamlit as st
from datetime import datetime

from database import (
    create_table, delete_reminder, get_all_persons, get_all_reminders,
    get_logs, insert_log, insert_person, insert_reminder, mark_reminder_complete,
)
from face_module import get_embedding
from audio_manager import start as start_audio, speak_direct

# --- Initialization (Preventing re-init on every rerun) ---
@st.cache_resource
def init_audio():
    try:
        start_audio()
        return True
    except Exception as e:
        st.error(f"Audio initialization failed: {e}")
        return False

init_audio()

st.set_page_config(page_title="AI Memory Companion", page_icon="🧠", layout="wide")

REMINDER_CATEGORIES = [
    "Medication", "Meal", "Hydration",
    "Safety", "Orientation", "Appointment", "General",
]
CAT_ICONS = {
    "Medication": "💊", "Meal": "🥗", "Hydration": "💧",
    "Safety": "🔒", "Orientation": "🧭", "Appointment": "📅", "General": "📌",
}

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#F4FBF8}
[data-testid="stSidebar"]{background:#FFFFFF!important;border-right:0.5px solid #D8EDE6}
.block-container{padding:1.5rem 2rem 2rem}
.cg-card{background:#FFFFFF;border:0.5px solid #D8EDE6;border-radius:14px;padding:16px 18px;margin-bottom:14px}
.cg-card-title{font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:12px}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.metric-box{background:#FFFFFF;border:0.5px solid #D8EDE6;border-radius:12px;padding:12px 14px}
.metric-label{font-size:11px;color:#6B9E90;margin-bottom:4px}
.metric-value{font-size:22px;font-weight:600}
.metric-note{font-size:10px;color:#9BB8AE;margin-top:2px}
.rem-row{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:0.5px solid #EDF5F2}
.rem-row:last-child{border:none}
.rem-icon{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.rem-title{font-size:13px;font-weight:500;color:#1a1a1a}
.rem-meta{font-size:11px;color:#9BB8AE;margin-top:1px}
.pill{display:inline-block;font-size:10px;padding:3px 9px;border-radius:99px;font-weight:500}
.pill-done{background:#EAF3DE;color:#3B6D11}
.pill-pending{background:#FAEEDA;color:#854F0B}
.pill-teal{background:#E1F5EE;color:#0F6E56}
.face-tile{border:0.5px solid #D8EDE6;border-radius:12px;padding:10px 8px;text-align:center;margin:4px}
.av{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;margin:0 auto 6px}
.av-t{background:#9FE1CB;color:#085041}
.av-b{background:#B5D4F4;color:#0C447C}
.av-p{background:#CECBF6;color:#3C3489}
.av-c{background:#F5C4B3;color:#712B13}
.face-name{font-size:12px;font-weight:500;color:#1a1a1a}
.face-rel{font-size:10px;color:#9BB8AE}
.face-seen{font-size:10px;color:#9BB8AE;margin-top:3px}
.log-row{display:flex;gap:8px;align-items:flex-start;padding:6px 9px;border-radius:8px;font-size:11px;margin-bottom:4px}
.lg-g{background:#E1F5EE}.lg-b{background:#E6F1FB}.lg-a{background:#FAEEDA}.lg-n{background:#F4F4F2}
.log-ts{color:#9BB8AE;min-width:38px}.log-type{font-weight:600;min-width:72px}.log-desc{color:#555;flex:1}
.pt-safe-chip{display:inline-flex;align-items:center;gap:6px;background:#E1F5EE;border:0.5px solid #5DCAA5;border-radius:99px;padding:5px 14px;font-size:12px;color:#0F6E56;font-weight:500}
.orient-band{background:#FFFFFF;border:0.5px solid #9FE1CB;border-radius:14px;padding:16px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.orient-day{font-size:17px;font-weight:600;color:#085041}
.orient-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#0F6E56;margin-bottom:3px}
.big-clock{font-size:32px;font-weight:600;color:#085041;line-height:1}
.task-box{background:#E1F5EE;border-radius:10px;padding:14px 16px;margin-bottom:12px}
.task-cat{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#0F6E56;font-weight:600;margin-bottom:4px}
.task-name{font-size:17px;font-weight:600;color:#085041}
.task-time{font-size:13px;color:#0F6E56;margin-top:3px}
.vis-box{background:#E1F5EE;border-radius:10px;padding:12px 14px;display:flex;align-items:center;gap:10px;margin-bottom:12px}
.vis-av{width:38px;height:38px;border-radius:50%;background:#9FE1CB;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:600;color:#085041;flex-shrink:0}
.vis-name{font-size:14px;font-weight:600;color:#085041}
.vis-rel{font-size:11px;color:#0F6E56}
.sched-chip{display:inline-block;border-radius:9px;padding:7px 12px;font-size:11px;font-weight:500;margin:3px}
.tip-card{background:#E1F5EE;border:0.5px solid #5DCAA5;border-radius:14px;padding:14px 16px;margin-bottom:12px}
h1,h2,h3{color:#085041!important}
.stTabs [data-baseweb="tab"]{font-size:13px!important}
div[data-testid="stForm"]{border:none!important;padding:0!important}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:0 0 16px;border-bottom:0.5px solid #D8EDE6;margin-bottom:16px">
      <div style="width:36px;height:36px;border-radius:12px;background:#E1F5EE;display:flex;align-items:center;justify-content:center;font-size:20px">🧠</div>
      <div>
        <div style="font-size:15px;font-weight:600;color:#085041">CogniCare</div>
        <div style="font-size:11px;color:#6B9E90">Memory companion</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("View", ["Patient view", "Caregiver", "Register face"], label_visibility="collapsed")

    st.divider()

    persons_count = len(get_all_persons())
    pending_count = sum(1 for r in get_all_reminders() if r[4] == 0)
    st.markdown(f"""
    <div style="font-size:12px;color:#0F6E56">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#1D9E75;margin-right:6px"></span>System online
    </div>
    <div style="font-size:11px;color:#9BB8AE;margin-top:4px">{persons_count} persons · {pending_count} pending</div>
    """, unsafe_allow_html=True)


# ── PATIENT VIEW ──────────────────────────────────────────────────────────────
def render_patient():
    now = datetime.now()
    col_t, col_c = st.columns([3, 1])
    with col_t:
        st.markdown("<h2 style='margin:0;padding:0'>Good day, friend 😊</h2>", unsafe_allow_html=True)
    with col_c:
        st.markdown("<div style='padding-top:6px'><span class='pt-safe-chip'>✅ You are safe at home</span></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='orient-band'>
      <div>
        <div class='orient-lbl'>Today is</div>
        <div class='orient-day'>{now.strftime('%A, %B %d, %Y')}</div>
      </div>
      <div style='text-align:right'>
        <div class='orient-lbl'>The time</div>
        <div class='big-clock'>{now.strftime('%I:%M %p').lstrip('0')}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_task, col_who = st.columns(2)
    reminders = get_all_reminders()
    pending   = [r for r in reminders if r[4] == 0]

    with col_task:
        st.markdown("<div class='cg-card'>", unsafe_allow_html=True)
        st.markdown("<div class='cg-card-title'>Your next task</div>", unsafe_allow_html=True)
        if pending:
            t = pending[0]
            icon = CAT_ICONS.get(t[3], "📌")
            st.markdown(f"""
            <div class='task-box'>
              <div class='task-cat'>{icon} {t[3]}</div>
              <div class='task-name'>{t[1]}</div>
              <div class='task-time'>at {t[2]}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(" I have done this", key="pt_done", use_container_width=True):
                mark_reminder_complete(t[0])
                speak_direct(f"Well done! Your {t[1]} is marked as completed.")
                st.balloons()
                st.rerun()
        else:
            st.markdown(
                "<div style='text-align:center;padding:24px 0'>"
                "<div style='font-size:36px'>😊</div>"
                "<div style='color:#0F6E56;font-weight:600;margin-top:6px'>All tasks done!</div>"
                "</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_who:
        st.markdown("<div class='cg-card'>", unsafe_allow_html=True)
        st.markdown("<div class='cg-card-title'>Who is here</div>", unsafe_allow_html=True)
        persons = get_all_persons()
        if persons:
            p = persons[0]
            initials = "".join(w[0].upper() for w in p[0].split()[:2]) if p[0] else "??"
            st.markdown(f"""
            <div class='vis-box'>
              <div class='vis-av'>{initials}</div>
              <div><div class='vis-name'>{p[0]}</div><div class='vis-rel'>Your {p[1].lower()}</div></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#9BB8AE;font-size:13px;margin-bottom:10px'>No one recognised yet.</div>", unsafe_allow_html=True)

        if st.button("🎤  Ask for help", key="pt_help", use_container_width=True):
            msg = "Hello! I am here. How can I help you today?"
            speak_direct(msg)
            st.info(f'Assistant: "{msg}"')
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='cg-card'><div class='cg-card-title'>Need help? Press a button below</div>", unsafe_allow_html=True)
    c_sos, c_talk, c_where, c_list = st.columns(4)

    with c_sos:
        if st.button("🆘 SOS", key="sos", use_container_width=True):
            speak_direct("Alerting your caregiver. Stay calm.")
            insert_log("CRITICAL_ALERT", "SOS Button Pressed")
            st.error("Alert Sent")

    with c_talk:
        if st.button("🎤 Talk", key="talk", use_container_width=True):
            speak_direct("I am listening.")

    with c_where:
        if st.button("🏠 Location", key="loc", use_container_width=True):
            speak_direct("You are safe at home.")

    with c_list:
        if st.button("📋 Tasks", key="tasks", use_container_width=True):
            rem_count = len(pending)
            speak_direct(f"You have {rem_count} tasks remaining.")

    st.markdown("</div>", unsafe_allow_html=True)


# ── CAREGIVER VIEW ────────────────────────────────────────────────────────────
def render_caregiver():
    now = datetime.now()
    st.markdown("## Caregiver overview")

    reminders = get_all_reminders()
    pending_n = sum(1 for r in reminders if r[4] == 0)
    logs_top  = get_logs(limit=10)

    # FIX: safe access — next_rem may be None when no pending reminders exist
    next_rem  = next((r for r in reminders if r[4] == 0), None)
    next_time = next_rem[2] if next_rem is not None else "—"
    next_cat  = next_rem[3] if next_rem is not None else "None"

    alert_count = sum(1 for l in logs_top if "ALERT" in l[2])

    st.markdown(f"""
    <div class='metric-grid'>
      <div class='metric-box'><div class='metric-label'>Known faces</div><div class='metric-value'>{len(get_all_persons())}</div></div>
      <div class='metric-box'><div class='metric-label'>Pending tasks</div><div class='metric-value'>{pending_n}</div></div>
      <div class='metric-box'><div class='metric-label'>Next reminder</div><div class='metric-value'>{next_time}</div><div class='metric-note'>{next_cat}</div></div>
      <div class='metric-box'><div class='metric-label'>Alerts today</div><div class='metric-value'>{alert_count}</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Schedule", "👁️ Persons", "🕒 Activity log"])

    # ── Schedule tab ──────────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns([1, 1.5])

        with c1:
            st.subheader("Add Reminder")
            with st.form("add_rem"):
                desc = st.text_input("Task description")
                tm   = st.time_input("Reminder time")
                cat  = st.selectbox("Category", REMINDER_CATEGORIES)
                if st.form_submit_button("Save reminder"):
                    if desc.strip():
                        insert_reminder(desc.strip(), tm.strftime("%H:%M"), cat)
                        st.success(f"Reminder saved: {desc} at {tm.strftime('%H:%M')}")
                        st.rerun()
                    else:
                        st.error("Please enter a task description.")

        with c2:
            st.subheader("Current Reminders")
            if not reminders:
                st.info("No reminders scheduled.")
            else:
                for r in reminders:
                    rid, text, time_str, category, completed = r
                    icon   = CAT_ICONS.get(category, "📌")
                    status = "✅" if completed else "⏳"
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(
                            f"{status} **{time_str}** — {icon} {text} "
                            f"<span style='color:#9BB8AE;font-size:11px'>({category})</span>",
                            unsafe_allow_html=True
                        )
                    with col_b:
                        if not completed:
                            if st.button("🗑", key=f"del_{rid}", help="Delete reminder"):
                                delete_reminder(rid)
                                st.rerun()

    # ── Persons tab ───────────────────────────────────────────────────────────
    with tab2:
        persons = get_all_persons()
        if not persons:
            st.info("No registered faces. Use 'Register face' to add people.")
        else:
            for p in persons:
                name, rel, _, reminder, last_seen, visits = p
                last_seen_str = last_seen[:16] if last_seen else "Never"
                st.markdown(
                    f"**{name}** — {rel} &nbsp;|&nbsp; "
                    f"Visits: {visits or 0} &nbsp;|&nbsp; "
                    f"Last seen: {last_seen_str}"
                    + (f" &nbsp;|&nbsp; *{reminder}*" if reminder else ""),
                    unsafe_allow_html=True
                )

    # ── Activity log tab ──────────────────────────────────────────────────────
    with tab3:
        logs = get_logs(limit=50)
        if not logs:
            st.info("No activity logged yet.")
        else:
            for log_id, ts, event_type, desc in logs:
                # Colour code by event type
                if "RECOGNITION" in event_type:
                    css = "lg-g"
                elif "ALERT" in event_type or "HELP" in event_type:
                    css = "lg-a"
                elif "REMINDER" in event_type:
                    css = "lg-b"
                else:
                    css = "lg-n"
                st.markdown(
                    f"<div class='log-row {css}'>"
                    f"<span class='log-ts'>{ts[11:16]}</span>"
                    f"<span class='log-type'>{event_type}</span>"
                    f"<span class='log-desc'>{desc or ''}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )


# ── REGISTRATION HELPERS ──────────────────────────────────────────────────────

def _augment_and_embed(img: np.ndarray) -> list:
    """
    Extract FaceNet embeddings from one image using multiple augmentations
    (flips, brightness, slight crops) so each real photo produces several
    training samples — improves recognition accuracy significantly.
    Returns a list of embedding vectors (may be empty if no face found).
    """
    results = []

    def _try(frame):
        emb = get_embedding(frame)
        if emb is not None:
            results.append(emb)

    # 1. Original
    _try(img)

    # 2. Horizontal flip
    _try(cv.flip(img, 1))

    # 3. Slightly brighter
    bright = cv.convertScaleAbs(img, alpha=1.15, beta=15)
    _try(bright)

    # 4. Slightly darker
    dark = cv.convertScaleAbs(img, alpha=0.85, beta=-15)
    _try(dark)

    # 5. Centre crop (90 %)
    h, w = img.shape[:2]
    margin_h, margin_w = int(h * 0.05), int(w * 0.05)
    cropped = img[margin_h: h - margin_h, margin_w: w - margin_w]
    if cropped.size:
        _try(cv.resize(cropped, (w, h)))

    # 6. Flip of bright
    _try(cv.flip(bright, 1))

    return results


def _img_to_thumbnail_b64(img: np.ndarray, size: int = 80) -> str:
    """Resize image and return a base-64 PNG data-URI for inline display."""
    import base64
    h, w  = img.shape[:2]
    scale = size / max(h, w)
    thumb = cv.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv.imencode(".png", thumb)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _reg_state_init():
    defaults = {
        "reg_embeddings":  [],    # flat list of embedding vectors
        "reg_previews":    [],    # list of {"b64": ..., "label": ..., "count": N}
        "reg_processed":   set(), # filenames/keys already processed
        "reg_cam_idx":     0,     # counter to force camera_input refresh
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _process_image(img: np.ndarray, label: str) -> int:
    """
    Augment-embed one image, append to session state, return count added.
    """
    embs = _augment_and_embed(img)
    st.session_state.reg_embeddings.extend(embs)
    b64  = _img_to_thumbnail_b64(img)
    st.session_state.reg_previews.append({
        "b64":   b64,
        "label": label,
        "count": len(embs),
    })
    return len(embs)


# ── FACE REGISTRATION ─────────────────────────────────────────────────────────
def render_registration():
    _reg_state_init()

    st.markdown("""
    <div style='background:#E1F5EE;border:0.5px solid #5DCAA5;border-radius:14px;
                padding:16px 20px;margin-bottom:18px'>
      <div style='font-size:20px;font-weight:700;color:#085041;margin-bottom:4px'>
        👤 Register New Person
      </div>
      <div style='font-size:13px;color:#0F6E56'>
        Capture or upload <b>5+ photos</b> from different angles and lighting
        for best recognition. Each photo is automatically augmented into
        multiple training samples.
      </div>
    </div>
    """, unsafe_allow_html=True)

    total_embs = len(st.session_state.reg_embeddings)

    # ── Progress bar ─────────────────────────────────────────────────────────
    target = 15   # 15 augmented samples = good recognition
    pct    = min(total_embs / target, 1.0)
    bar_color = "#1D9E75" if pct >= 1.0 else "#F5A623" if pct >= 0.4 else "#E74C3C"

    st.markdown(f"""
    <div style='margin-bottom:6px;font-size:13px;color:#085041;font-weight:600'>
      Training samples collected: {total_embs}
      {"Ready to register!" if pct >= 1.0 else f" — capture more photos for best accuracy"}
    </div>
    <div style='background:#D8EDE6;border-radius:99px;height:10px;margin-bottom:18px'>
      <div style='background:{bar_color};width:{pct*100:.0f}%;height:10px;
                  border-radius:99px;transition:width 0.3s'></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two capture columns ───────────────────────────────────────────────────
    col_cam, col_up = st.columns(2, gap="large")

    # ── Webcam capture ────────────────────────────────────────────────────────
    with col_cam:
        st.markdown("#### 📷 Webcam capture")
        st.markdown(
            "<div style='font-size:12px;color:#6B9E90;margin-bottom:8px'>"
            "Look straight at the camera, then take the shot. "
            "Repeat from different angles.</div>",
            unsafe_allow_html=True
        )

        cam_shot = st.camera_input(
            "Take a photo",
            key=f"cam_{st.session_state.reg_cam_idx}",
            label_visibility="collapsed",
        )

        if cam_shot is not None:
            raw = cam_shot.getvalue()
            arr = np.frombuffer(raw, np.uint8)
            img = cv.imdecode(arr, cv.IMREAD_COLOR)
            cam_key = f"cam_{st.session_state.reg_cam_idx}"

            if cam_key not in st.session_state.reg_processed:
                st.session_state.reg_processed.add(cam_key)

                if img is not None:
                    with st.spinner("Extracting face data..."):
                        added = _process_image(img, f"Webcam #{len(st.session_state.reg_previews)+1}")

                    if added > 0:
                        st.success(f"✔ Got {added} samples from this shot!")
                        # Bump index so camera widget resets for next shot
                        st.session_state.reg_cam_idx += 1
                        st.rerun()
                    else:
                        st.error("No face detected. Move closer and ensure good lighting.")
                        st.session_state.reg_cam_idx += 1
                        st.rerun()

    # ── File upload ───────────────────────────────────────────────────────────
    with col_up:
        st.markdown("####Upload photos")
        st.markdown(
            "<div style='font-size:12px;color:#6B9E90;margin-bottom:8px'>"
            "Upload JPG/PNG photos — WhatsApp images work well. "
            "Select multiple at once.</div>",
            unsafe_allow_html=True
        )

        uploaded_files = st.file_uploader(
            "Choose photos",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"],
            key="reg_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files:
            new_files = [
                f for f in uploaded_files
                if f"file_{f.name}_{f.size}" not in st.session_state.reg_processed
            ]

            if new_files:
                with st.spinner(f"Processing {len(new_files)} photo(s)..."):
                    ok, fail = 0, []
                    for f in new_files:
                        fkey = f"file_{f.name}_{f.size}"
                        st.session_state.reg_processed.add(fkey)
                        raw  = f.read()
                        arr  = np.frombuffer(raw, np.uint8)
                        img  = cv.imdecode(arr, cv.IMREAD_COLOR)
                        if img is None:
                            fail.append(f.name)
                            continue
                        added = _process_image(img, f.name[:20])
                        if added > 0:
                            ok += 1
                        else:
                            fail.append(f.name)

                if ok:
                    st.success(f"✔ {ok} photo(s) processed successfully.")
                if fail:
                    st.warning(f"⚠ No face found in: {', '.join(fail)}")
                st.rerun()

    # ── Sample previews ───────────────────────────────────────────────────────
    if st.session_state.reg_previews:
        st.markdown("---")
        st.markdown(
            f"**Captured samples ({len(st.session_state.reg_previews)} source photos)**"
        )
        thumb_cols = st.columns(min(len(st.session_state.reg_previews), 6))
        for i, prev in enumerate(st.session_state.reg_previews):
            with thumb_cols[i % 6]:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<img src='{prev['b64']}' style='border-radius:8px;border:2px solid #5DCAA5;"
                    f"width:72px;height:72px;object-fit:cover'/>"
                    f"<div style='font-size:10px;color:#6B9E90;margin-top:3px'>"
                    f"{prev['label']}<br>+{prev['count']} samples</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        if st.button("🗑️ Clear all samples", key="reg_clear"):
            for k in ["reg_embeddings", "reg_previews"]:
                st.session_state[k] = []
            st.session_state.reg_processed  = set()
            st.session_state.reg_cam_idx   += 1
            st.rerun()

    # ── Registration form ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Person details")

    if total_embs == 0:
        st.info(" Capture or upload at least one photo above before registering.")

    with st.form("reg_face"):
        col_n, col_r = st.columns(2)
        with col_n:
            name = st.text_input("Full name *", placeholder="e.g. Priya Sharma")
        with col_r:
            rel = st.selectbox(
                "Relationship *",
                ["Daughter", "Son", "Wife", "Husband", "Sister", "Brother",
                 "Caregiver", "Friend", "Doctor", "Other"]
            )
        reminder = st.text_area(
            "Reminder message (optional)",
            placeholder="e.g. This is your daughter Priya. She visits every Sunday morning.",
            height=80,
        )

        # Disable button if no samples yet
        can_register = total_embs > 0
        submitted    = st.form_submit_button(
            "Register person",
            disabled=not can_register,
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not name.strip():
            st.error("Please enter the person's name.")
            return

        embeddings = st.session_state.reg_embeddings

        if not embeddings:
            st.error("No face samples collected. Please add photos first.")
            return

        if len(embeddings) < 5:
            st.warning(
                f"Only {len(embeddings)} samples — recognition may be less reliable. "
                "Consider adding more photos later via Manage DB."
            )

        # ── Save ─────────────────────────────────────────────────────────────
        insert_person(name.strip(), rel, json.dumps(embeddings), reminder.strip())
        insert_log(
            "REGISTRATION",
            f"Registered: {name.strip()} ({rel}) with {len(embeddings)} augmented samples "
            f"from {len(st.session_state.reg_previews)} source photo(s)"
        )

        st.success(
            f" **{name.strip()}** registered successfully!  \n"
            f"{len(embeddings)} training samples from "
            f"{len(st.session_state.reg_previews)} photo(s)."
        )
        st.balloons()

        # Reset for next person
        for k in ["reg_embeddings", "reg_previews"]:
            st.session_state[k] = []
        st.session_state.reg_processed  = set()
        st.session_state.reg_cam_idx   += 1


# ── Router ────────────────────────────────────────────────────────────────────
def main():
    create_table()
    if "Patient" in mode:
        render_patient()
    elif "Caregiver" in mode:
        render_caregiver()
    else:
        render_registration()


if __name__ == "__main__":
    main()