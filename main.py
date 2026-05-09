import os
# 1. Suppress TensorFlow/CUDA warnings immediately
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2 as cv
import json
import time
from datetime import datetime

# Core System Imports
from state import state 
from config import IP_WEBCAM_URL, CAMERA_INDEX, STABILITY_THRESHOLD
from database import (
    create_table, get_all_persons, insert_log, update_person_last_seen
)
from face_module import get_embedding, find_best_match
from audio_manager import start as start_audio, enqueue, speaking
from reminder_engine import start as start_reminders
from voice_assistant import start as start_voice 
from context_engine import generate_recognition_message

# ── INIT ─────────────────────────────────────────────────────────────────────
create_table()

def load_known_persons():
    """Loads all faces from the database into memory for fast matching."""
    rows = get_all_persons()
    persons = []
    for name, relationship, embedding_json, reminder, last_seen, visit_count in rows:
        try:
            loaded = json.loads(embedding_json)
            # Ensure embedding is a list of lists (required by face_module)
            if isinstance(loaded[0], float): 
                loaded = [loaded]
            persons.append({
                "name": name, 
                "relationship": relationship, 
                "embeddings": loaded,
                "reminder": reminder, 
                "last_seen": last_seen, 
                "visit_count": visit_count or 0,
            })
        except Exception as e:
            print(f"[DATABASE ERROR] Could not load {name}: {e}")
    return persons

known_persons = load_known_persons()
print(f"[COGNICARE] System Ready. Loaded {len(known_persons)} person(s)")

# ── START SERVICES ───────────────────────────────────────────────────────────
start_audio()
start_reminders()
start_voice()

# FIX: Startup Welcome Message
enqueue("Welcome. CogniCare AI is now active and monitoring.")

# ── CAMERA SETUP ─────────────────────────────────────────────────────────────
cap = cv.VideoCapture(IP_WEBCAM_URL if IP_WEBCAM_URL else CAMERA_INDEX)

# Performance & Tracking Variables
last_visit_times = {}
VISIT_COOLDOWN = 300  # 5 Minutes
frame_count = 0
AI_PROCESS_INTERVAL = 10     # LAG FIX: Only run heavy AI every 10 frames
last_announced_person = None  # DOUBLE SPEECH FIX: Track who we just greeted

while True:
    ret, frame = cap.read()
    if not ret: 
        print("[CAMERA ERROR] Could not read frame.")
        break
    
    frame_count += 1
    # Process at 640x480 for consistent speed
    display_frame = cv.resize(frame, (640, 480))
    
    # ── AI LOGIC (INTERVAL CONTROLLED) ───────────────────────────────────────
    # This prevents the webcam from lagging on CPU-only machines
    if frame_count % AI_PROCESS_INTERVAL == 0:
        embedding = get_embedding(display_frame) 
        
        current_label = "Unknown"
        detected_person_obj = None
        score = 0.0

        if embedding:
            detected_person_obj, score = find_best_match(embedding, known_persons)
            if detected_person_obj:
                current_label = detected_person_obj["name"]

        # Update the Stability Buffer in state.py
        state.identity_buffer.append(current_label)
        if len(state.identity_buffer) > STABILITY_THRESHOLD:
            state.identity_buffer.pop(0)

        # Update stable label only if buffer is consistent
        if len(set(state.identity_buffer)) == 1:
            state.stable_label = state.identity_buffer[0]
    
    # ── RECOGNITION & SPEECH LOGIC ──────────────────────────────────────────
    if state.stable_label != "Unknown" and state.stable_label != last_announced_person:
        
        # 1. Update Database (only on fresh visits)
        now_ts = time.time()
        last_t = last_visit_times.get(state.stable_label, 0)
        
        if (now_ts - last_t) > VISIT_COOLDOWN:
            update_person_last_seen(state.stable_label)
            last_visit_times[state.stable_label] = now_ts
            # If we found the person in the DB, increment session visit count
            if detected_person_obj:
                detected_person_obj["visit_count"] = (detected_person_obj.get("visit_count") or 0) + 1

        # 2. Voice Announcement
        if not speaking():
            # Generate personalized greeting (e.g. "Good afternoon Ishav...")
            msg = generate_recognition_message(detected_person_obj)
            enqueue(msg)
            
            # Immediately lock this person so we don't repeat the greeting
            last_announced_person = state.stable_label 
            state.last_person = state.stable_label
            
            insert_log("RECOGNITION", f"Recognised: {state.stable_label} (Score: {score:.2f})")

    # 3. Reset lock if frame is empty
    if state.stable_label == "Unknown":
        last_announced_person = None
        state.last_person = None

    # ── DRAW UI (RUNS EVERY FRAME) ──────────────────────────────────────────
    is_known = state.stable_label != "Unknown"
    # Green for family/friends, Orange for Unknown
    status_color = (0, 255, 0) if is_known else (0, 80, 255) 
    
    # Header Overlay
    cv.rectangle(display_frame, (0,0), (640, 60), (30,30,30), -1) 
    cv.putText(display_frame, f"CogniCare AI | Status: {state.stable_label}", (20, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Main Window Display
    cv.imshow("CogniCare AI", display_frame)
    
    # Exit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
insert_log("SYSTEM", "User shut down CogniCare AI")
print("[COGNICARE] Shutdown complete.")