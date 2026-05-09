import streamlit as st
import pandas as pd
from datetime import datetime
import json
import cv2 as cv
import numpy as np
import time

# Import functions exactly as named in database.py
from database import (
    create_table, get_all_persons, get_all_reminders, 
    insert_person, insert_reminder, delete_reminder, get_logs,
    mark_reminder_complete
)
from face_module import get_embedding

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CogniCare AI", page_icon="🛡️", layout="wide")

# ── DYNAMIC STYLING ─────────────────────────────────────────────────────────
def apply_custom_styles(mode):
    if mode == "Patient Companion":
        st.markdown("""
            <style>
            .main { background-color: #E3F2FD; }
            .patient-card { 
                background-color: white; padding: 30px; border-radius: 20px; 
                border: 4px solid #1E88E5; text-align: center; margin-bottom: 20px;
            }
            .big-time { font-size: 80px !important; font-weight: bold; color: #0D47A1; }
            .big-text { font-size: 40px !important; color: #1565C0; }
            .stButton>button { 
                height: 150px !important; font-size: 30px !important; 
                border-radius: 20px !important; background-color: #1976D2 !important;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .main { background-color: #f5f7f9; }
            .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            .stButton>button { height: 3em; background-color: #007bff; color: white; }
            h1 { color: #1e3d59; }
            </style>
            """, unsafe_allow_html=True)

# ── PATIENT COMPANION VIEW ──────────────────────────────────────────────────
def render_patient_dashboard():
    apply_custom_styles("Patient Companion")
    
    now = datetime.now()
    st.markdown(f"<h1 style='text-align: center; color: #0D47A1;'>Good Day, Friend!</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center;'>Today is {now.strftime('%A, %B %d')}</h2>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f'<div class="patient-card"><p class="big-text">The Time is</p><p class="big-time">{now.strftime("%I:%M %p")}</p></div>', unsafe_allow_html=True)
        
        # Big "Help" Button (Simulates Voice/Emergency)
        if st.button("📢 ASK FOR HELP\n(Voice Assistant)"):
            st.toast("Activating Voice Assistant... I am listening.")
            # This visually confirms the voice activation objective
            st.info("Assistant: 'Hello! I am here. How can I help you today?'")

    with col_right:
        # Fetch actual reminders from database
        reminders = get_all_reminders()
        pending = [r for r in reminders if len(r) > 4 and r[4] == 0]
        
        st.markdown('<div class="patient-card">', unsafe_allow_html=True)
        if pending:
            current_task = pending[0]
            st.markdown(f'<p class="big-text">Your Next Task:</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size: 50px; color: #D32F2F; font-weight: bold;">{current_task[1]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="big-text">at {current_task[2]}</p>', unsafe_allow_html=True)
            
            if st.button("✅ I HAVE DONE THIS"):
                mark_reminder_complete(current_task[0])
                st.balloons()
                st.rerun()
        else:
            st.markdown('<p class="big-text">All tasks done!</p><p class="big-time">😊</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("<h3 style='text-align: center;'>You are safe. Your family is just a call away.</h3>", unsafe_allow_html=True)

# ── CAREGIVER DASHBOARD ─────────────────────────────────────────────────────
def render_caregiver_dashboard():
    apply_custom_styles("Caregiver")
    st.title("🛡️ Caregiver Oversight Terminal")
    
    # KPIs
    persons = get_all_persons()
    reminders = get_all_reminders()
    pending_count = len([r for r in reminders if r[4] == 0])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Known Faces", len(persons))
    c2.metric("Pending Tasks", pending_count)
    c3.metric("System Status", "Online ✅")

    tab1, tab2 = st.tabs(["📋 Management", "🕒 Interaction Logs"])

    with tab1:
        l_col, r_col = st.columns([1, 1])
        with l_col:
            st.subheader("New Reminder")
            with st.form("r_form"):
                txt = st.text_input("Task Name")
                tm = st.time_input("Time")
                if st.form_submit_button("Add Task"):
                    insert_reminder(txt, tm.strftime("%H:%M"))
                    st.rerun()
        with r_col:
            st.subheader("Active Schedule")
            if reminders:
                df = pd.DataFrame(reminders, columns=["ID", "Task", "Time", "Cat", "Status"])
                df["Status"] = df["Status"].apply(lambda x: "✅" if x==1 else "⏳")
                st.dataframe(df[["Time", "Task", "Status"]], use_container_width=True)

    with tab2:
        logs = get_logs(limit=50)
        if logs:
            df_l = pd.DataFrame(logs, columns=["ID", "Timestamp", "Type", "Details"])
            st.dataframe(df_l[["Timestamp", "Type", "Details"]], use_container_width=True)

# ── REGISTRATION ────────────────────────────────────────────────────────────
def render_registration():
    apply_custom_styles("Caregiver")
    st.title("👤 Face Registration")
    with st.form("reg"):
        name = st.text_input("Name")
        rel = st.selectbox("Relationship", ["Family", "Caretaker", "Doctor"])
        file = st.file_uploader("Photo", type=['jpg', 'jpeg', 'png'])
        if st.form_submit_button("Register"):
            if name and file:
                img = cv.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                emb = get_embedding(img)
                if emb:
                    insert_person(name, rel, json.dumps(emb), "")
                    st.success("Registered!")
                else: st.error("No face found.")

# ── MAIN ENGINE ─────────────────────────────────────────────────────────────
def main():
    st.sidebar.title("CogniCare AI")
    mode = st.sidebar.radio("Switch View", ["Patient Companion", "Caregiver Dashboard", "Face Registration"])
    
    if mode == "Patient Companion":
        render_patient_dashboard()
    elif mode == "Caregiver Dashboard":
        render_caregiver_dashboard()
    else:
        render_registration()

if __name__ == "__main__":
    create_table()
    main()