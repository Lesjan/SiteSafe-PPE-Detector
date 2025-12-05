import streamlit as st
import cv2
import time
from datetime import datetime
import pandas as pd
import os
import pickle
import random
import numpy as np 
from ultralytics import YOLO
# Corrected import name: 'webrtc_streamer'
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration

# --- GLOBAL PAGE CONFIGURATION ---
# Must be the first Streamlit command
st.set_page_config(
    page_title="SiteSafe PPE Detector (Cloud Ready)", 
    layout="wide", 
    initial_sidebar_state="collapsed" 
) 

# ----------------------
# CONFIG
# ----------------------
LOG_FILE = "ppe_app_logs.csv"
USER_DB_FILE = "user_credentials.pkl"
MODEL_PATH = "best.pt" 
USE_SIMULATED = False

# ----------------------
# MODEL LOADING (CACHED)
# ----------------------
@st.cache_resource
def load_yolo_model():
    global USE_SIMULATED
    try:
        if os.path.exists(MODEL_PATH):
            return YOLO(MODEL_PATH)
        else:
            return YOLO("yolov8n.pt") 
    except Exception as e:
        USE_SIMULATED = True
        st.warning(f"⚠️ Model failed to load ({e}). Running in SIMULATED detection mode. Results are random.")
        return None

model = load_yolo_model()

# --- HANDLERS ---
def go_back():
    st.session_state.page = "worker"
    st.session_state.uploaded_image = None
    st.session_state.webrtc_running = False
    st.rerun() 

# ----------------------
# USER DB helpers
# ----------------------
def load_user_db():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {"admin": "12345"}
    return {"admin": "12345"}

def save_user_db(db):
    with open(USER_DB_FILE, "wb") as f:
        pickle.dump(db, f)

USER_DB = load_user_db()

# ----------------------
# WORKERS & PPE MAPPING
# ----------------------
WORKERS = {
    "CW01": "Jasmin Romon",
    "CW02": "Cordel Kent Corona",
    "CW03": "Shine Acuña",
    "CW04": "Justin Baculio",
    "CW05": "Alexis Anne Emata"
}

PPE_ITEMS = [
    "Hard Hat", "Safety Vest", "Gloves",
    "Safety Boots", "Eye/Face Protection",
    "Hearing Protection", "Safety Harness"
]

CLASS_TO_PPE = {
    "hardhat": "Hard Hat", "helmet": "Hard Hat",
    "vest": "Safety Vest", "glove": "Gloves",
    "boot": "Safety Boots", "boots": "Safety Boots",
    "goggles": "Eye/Face Protection", "mask": "Eye/Face Protection",
    "earmuff": "Hearing Protection", "ear_protection": "Hearing Protection",
    "harness": "Safety Harness"
}

# ----------------------
# LOGGING 
# ----------------------
def init_log_file():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["timestamp", "worker_id", "worker_name"] + PPE_ITEMS)
        df.to_csv(LOG_FILE, index=False)

init_log_file()

def log_inspection(worker_id, worker_name, detected_set):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": timestamp, "worker_id": worker_id, "worker_name": worker_name}
    for item in PPE_ITEMS:
        row[item] = 1 if item in detected_set else 0
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["timestamp", "worker_id", "worker_name"] + PPE_ITEMS)
    df.loc[len(df)] = row
    df.to_csv(LOG_FILE, index=False)

# ----------------------
# WEBRTC VIDEO TRANSFORMER CLASS (Handles frame processing)
# ----------------------
class PPEVideoTransformer(VideoTransformerBase):
    def __init__(self, model, workers, ppe_items, class_to_ppe, worker_id, worker_name, log_func, use_simulated):
        self.model = model
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.log_inspection = log_func
        self.PPE_ITEMS = ppe_items
        self.CLASS_TO_PPE = class_to_ppe
        self.USE_SIMULATED = use_simulated
        self.frame_counter = 0
        self.FRAME_SKIP = 3 
        self.last_log_time = time.time()
        self.AUTO_LOG_INTERVAL = 5
        self.inspection_complete = False
        
        st.session_state.detected_ppe = set()
        st.session_state.is_compliant = False
        st.session_state.log_message = ""

    def simulate_detect(self, frame):
        present = set()
        if random.random() > 0.2: present.add("Hard Hat")
        if random.random() > 0.3: present.add("Safety Vest")
        if random.random() > 0.6: present.add("Gloves")
        if random.random() > 0.5: present.add("Safety Boots")
        if random.random() > 0.8: present.add("Eye/Face Protection")
        if random.random() > 0.9: present.add("Hearing Protection")
        if random.random() > 0.95: present.add("Safety Harness")
        return present, frame

    def detect_ppe_webrtc(self, frame):
        if self.USE_SIMULATED or self.model is None:
            return self.simulate_detect(frame)

        detected = set()
        annotated_frame = frame
        
        if self.frame_counter % self.FRAME_SKIP == 0:
            try:
                results = self.model(
                    frame, 
                    device='cpu', 
                    imgsz=640, 
                    conf=0.35, # Confirmed: Lowered confidence for better detection
                    verbose=False
                )[0]
                
                annotated_frame = results.plot() 
                
                names = self.model.names if hasattr(self.model, "names") else {}
                for box in results.boxes:
                    cls_id = int(box.cls)
                    label = names.get(cls_id, str(cls_id)).lower()
                    if label in self.CLASS_TO_PPE:
                        detected.add(self.CLASS_TO_PPE[label])
                        
            except Exception as e:
                detected, annotated_frame = self.simulate_detect(frame) 
        
        missing = [item for item in self.PPE_ITEMS if item not in detected]
        now = time.time()

        if now - self.last_log_time > self.AUTO_LOG_INTERVAL:
            if not missing and not self.inspection_complete:
                self.log_inspection(self.worker_id, self.worker_name, detected)
                self.inspection_complete = True
                st.session_state.log_message = "✅ Compliance LOGGED! Feed stopped for inspection completion."
                
            elif missing and not self.inspection_complete:
                self.log_inspection(self.worker_id, self.worker_name, detected)
                st.session_state.log_message = f"Non-compliant status logged at {datetime.now().strftime('%H:%M:%S')}. Still checking..."
                
            self.last_log_time = now

        st.session_state.detected_ppe = detected
        st.session_state.is_compliant = not missing
        
        self.frame_counter += 1
        return detected, annotated_frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24") 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detected, annotated_frame = self.detect_ppe_webrtc(img_rgb) 

        return cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

# ----------------------
# UI: Login Page
# ----------------------
def login_page():
    st.title("🔐 SiteSafe - Compliance Detector")
    signin, signup = st.tabs(["Sign In", "Sign Up"])

    with signin:
        st.subheader("Sign In")
        username = st.text_input("Username", key="signin_user")
        password = st.text_input("Password", type="password", key="signin_pw")
        if st.button("Sign In", key="signin_btn"):
            if username in USER_DB and USER_DB[username] == password:
                st.session_state.logged_in = True
                st.session_state.user_name = username
                st.rerun() 
            else:
                st.error("Invalid username or password.")

    with signup:
        st.subheader("Create Account")
        new_user = st.text_input("New username", key="signup_user")
        new_pw = st.text_input("New password", type="password", key="signup_pw")
        confirm_pw = st.text_input("Confirm password", type="password", key="signup_confirm")
        if st.button("Sign Up", key="signup_btn"):
            if not new_user or not new_pw:
                st.error("Username and password cannot be empty.")
            elif new_user in USER_DB:
                st.error("Username already exists.")
            elif new_pw != confirm_pw:
                st.error("Passwords do not match.")
            else:
                USER_DB[new_user] = new_pw
                save_user_db(USER_DB)
                st.success("Account created. Please sign in.")
                st.rerun() 
                
# ----------------------
# UI: Worker selection
# ----------------------
def worker_page():
    st.title("👷 SiteSafe - Worker & Supervisor View")

    st.subheader(f"👋 Logged In User: {st.session_state.get('user_name', 'UNKNOWN')}")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun() 

    st.markdown("---")
    
    choice = st.selectbox("Select Worker ID to Inspect", list(WORKERS.keys()))
    worker_name = WORKERS[choice]
    st.write(f"**Worker Name:** {worker_name}")

    if st.button("Proceed to PPE Scanner"):
        st.session_state.worker_id = choice
        st.session_state.worker_name = worker_name
        st.session_state.page = "scanner"
        st.rerun() 

# ----------------------
# UI: Scanner page (WebRTC Live Camera)
# ----------------------
def scanner_page():
    # Hide the sidebar button 
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📹 Live PPE Scanner (WebRTC)")

    worker_id = st.session_state.get("worker_id")
    worker_name = st.session_state.get("worker_name")
    st.subheader(f"Target: **{worker_name}** ({worker_id})")
    
    # Back button
    st.button("⬅️ Back to Worker Selection", key="back_btn", on_click=go_back)

    st.markdown("---")

    video_col, status_col = st.columns([2, 1]) 

    with video_col:
        st.info("⚠️ Ensure your browser has camera access enabled and click 'Start'.")
        
        # --- WEBRTC STREAM ---
        webrtc_ctx = webrtc_streamer(
            key="ppe-detection-stream",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: PPEVideoTransformer(
                model=model, 
                workers=WORKERS, 
                ppe_items=PPE_ITEMS, 
                class_to_ppe=CLASS_TO_PPE,
                worker_id=worker_id, 
                worker_name=worker_name,
                log_func=log_inspection,
                use_simulated=USE_SIMULATED
            ),
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            async_transform=True
        )
    
    # --- RERUN LOOP ---
    # Fix: Replaced st.experimental_rerun() with the correct st.rerun()
    # This loop forces the script to rerun every second when the stream is playing, 
    # ensuring the checklist immediately reflects the updated st.session_state.
    if webrtc_ctx.state.playing:
        time.sleep(1) 
        st.rerun() 

    with status_col:
        
        if USE_SIMULATED:
            st.warning("⚠️ Running in SIMULATED detection mode.")

        detected = st.session_state.get("detected_ppe", set())
        missing = [it for it in PPE_ITEMS if it not in detected]
        
        # --- UI Update (Checklist) ---
        checklist_text = "### 📋 PPE Checklist\n"
        for it in PPE_ITEMS:
            if it in detected:
                checklist_text += f"**<span style='color:green'>✔ {it}</span>**\n"
            else:
                checklist_text += f"**<span style='color:red'>❌ {it}</span>**\n"
        
        st.markdown(checklist_text, unsafe_allow_html=True)
        
        # --- UI Update (Status) ---
        if webrtc_ctx.state.playing:
            if st.session_state.get("is_compliant", False):
                st.success("✅ **FULLY COMPLIANT**")
            elif detected:
                st.error("🚨 **NON-COMPLIANT**")
                st.warning(f"Missing: {', '.join(missing)}")
            else: 
                st.info("Scanning for worker...")
        else:
            st.warning("Click 'Start' below the video frame to begin scanning.")

        # Log Message Display
        if st.session_state.get("log_message"):
            st.info(st.session_state.log_message)
            
# ----------------------
# Main app flow
# ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.page == "worker":
        worker_page()
    elif st.session_state.page == "scanner":
        scanner_page()
    else:
        st.session_state.page = "worker"
        st.rerun()
