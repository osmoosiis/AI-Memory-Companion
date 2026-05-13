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

# --- CSS STYLES (Keep as provided, they are excellent) ---
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
    
    persons_count  = len(get_all_persons())
    pending_count  = sum(1 for r in get_all_reminders() if r[4] == 0)
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
            if st.button("✅  I have done this", key="pt_done", use_container_width=True):
                mark_reminder_complete(t[0])
                speak_direct(f"Well done! Your {t[1]} is marked as completed.")
                st.balloons()
                st.rerun()
        else:
            st.markdown("<div style='text-align:center;padding:24px 0'><div style='font-size:36px'>😊</div><div style='color:#0F6E56;font-weight:600;margin-top:6px'>All tasks done!</div></div>", unsafe_allow_html=True)
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

    # --- SOS Section (Logic and UI) ---
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
    st.markdown(f"## Caregiver overview")
    
    reminders = get_all_reminders()
    pending_n = sum(1 for r in reminders if r[4] == 0)
    next_rem  = next((r for r in reminders if r[4] == 0), None)
    logs_top  = get_logs(limit=10)

    # Fixed safety check for next_rem
    next_time = next_rem[2] if next_rem else "—"
    next_cat = next_rem[3] if next_rem else "None"

    st.markdown(f"""
    <div class='metric-grid'>
      <div class='metric-box'><div class='metric-label'>Known faces</div><div class='metric-value'>{len(get_all_persons())}</div></div>
      <div class='metric-box'><div class='metric-label'>Pending tasks</div><div class='metric-value'>{pending_n}</div></div>
      <div class='metric-box'><div class='metric-label'>Next reminder</div><div class='metric-value'>{next_time}</div><div class='metric-note'>{next_cat}</div></div>
      <div class='metric-box'><div class='metric-label'>Alerts</div><div class='metric-value'>{sum(1 for l in logs_top if "ALERT" in l[2])}</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Schedule", "👁️ Persons", "🕒 Activity log"])

    with tab1:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("Add Reminder")
            with st.form("add_rem"):
                desc = st.text_input("Task")
                tm = st.time_input("Time")
                cat = st.selectbox("Category", REMINDER_CATEGORIES)
                if st.form_submit_button("Save"):
                    insert_reminder(desc, tm.strftime("%H:%M"), cat)
                    st.rerun()
        with c2:
            st.subheader("Current Reminders")
            for r in reminders:
                st.write(f"{CAT_ICONS.get(r[3], '📌')} {r[2]} - {r[1]}")

    with tab2:
        persons = get_all_persons()
        if not persons:
            st.info("No registered faces.")
        else:
            for p in persons:
                st.write(f"**{p[0]}** ({p[1]})")

    with tab3:
        for l in get_logs(limit=20):
            st.text(f"{l[1]} | {l[2]} | {l[3]}")


# ── FACE REGISTRATION ─────────────────────────────────────────────────────────
def render_registration():
    st.header("Register New Person")
    with st.form("reg_face"):
        name = st.text_input("Name")
        rel = st.selectbox("Relationship", ["Family", "Caregiver", "Friend", "Other"])
        files = st.file_uploader("Photos", accept_multiple_files=True)
        if st.form_submit_button("Register"):
            if name and files:
                embeddings = []
                for f in files:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
                    emb = get_embedding(img)
                    if emb is not None:
                        embeddings.append(emb)
                
                if embeddings:
                    insert_person(name, rel, json.dumps(embeddings), "")
                    st.success("Registration Complete!")
                else:
                    st.error("No faces detected in images.")

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