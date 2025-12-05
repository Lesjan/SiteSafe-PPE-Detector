import streamlit as st
import cv2
import time
import os
import pickle
import random
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration
import requests

# ------------------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SiteSafe PPE Detector (Final Stable)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

LOG_FILE = "ppe_logs.csv"
USER_DB_FILE = "user_db.pkl"
MODEL_PATH = "best.pt"

# ------------------------------------------------------------------------------
# OPTION 2 — MODEL DOWNLOAD FROM GITHUB RAW
# ------------------------------------------------------------------------------
MODEL_URL = "https://raw.githubusercontent.com/<YOUR_USERNAME>/<YOUR_REPO>/<YOUR_BRANCH>/best.pt"


def download_model():
    """Download best.pt safely from GitHub RAW."""
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 1000000:  # <1MB == corrupted
            os.remove(MODEL_PATH)
        else:
            return

    try:
        st.info("Downloading PPE model... please wait (one-time download).")
        r = requests.get(MODEL_URL, timeout=30)

        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

            # Verify size
            if os.path.getsize(MODEL_PATH) < 1000000:
                raise ValueError("Downloaded model appears corrupted.")
        else:
            raise RuntimeError(f"HTTP {r.status_code}")

    except Exception as e:
        st.warning(f"⚠ Model download failed: {e}. Falling back to YOLOv8n.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)


# ------------------------------------------------------------------------------
# YOLO MODEL (CACHED)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    download_model()

    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH)
        except Exception as e:
            st.warning(f"⚠ Failed to load best.pt ({e}). Using YOLOv8n.")
            return YOLO("yolov8n.pt")

    return YOLO("yolov8n.pt")


model = load_model()

# ------------------------------------------------------------------------------
# USER DB
# ------------------------------------------------------------------------------
def load_user_db():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "rb") as f:
                return pickle.load(f)
        except:
            return {"admin": "12345"}
    return {"admin": "12345"}


def save_user_db(data):
    with open(USER_DB_FILE, "wb") as f:
        pickle.dump(data, f)


USER_DB = load_user_db()

# ------------------------------------------------------------------------------
# WORKERS
# ------------------------------------------------------------------------------
WORKERS = {
    "CW01": "Jasmin Romon",
    "CW02": "Cordel Kent Corona",
    "CW03": "Shine Acuña",
    "CW04": "Justin Baculio",
    "CW05": "Alexis Anne Emata",
}

# ------------------------------------------------------------------------------
# PPE MAPPING
# ------------------------------------------------------------------------------
PPE_ITEMS = [
    "Hard Hat",
    "Safety Vest",
    "Gloves",
    "Safety Boots",
    "Eye/Face Protection",
    "Hearing Protection",
    "Safety Harness"
]

CLASS_TO_PPE = {
    "hardhat": "Hard Hat",
    "helmet": "Hard Hat",
    "vest": "Safety Vest",
    "glove": "Gloves",
    "boot": "Safety Boots",
    "boots": "Safety Boots",
    "goggles": "Eye/Face Protection",
    "mask": "Eye/Face Protection",
    "earmuff": "Hearing Protection",
    "ear_protection": "Hearing Protection",
    "harness": "Safety Harness",
}

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
def init_log_file():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["timestamp", "worker_id", "worker_name"] + PPE_ITEMS)
        df.to_csv(LOG_FILE, index=False)


init_log_file()


def log_inspection(worker_id, worker_name, detected):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": timestamp, "worker_id": worker_id, "worker_name": worker_name}
    for item in PPE_ITEMS:
        row[item] = 1 if item in detected else 0

    df = pd.read_csv(LOG_FILE)
    df.loc[len(df)] = row
    df.to_csv(LOG_FILE, index=False)

# ------------------------------------------------------------------------------
# VIDEO TRANSFORMER
# ------------------------------------------------------------------------------
class PPEVideoTransformer(VideoTransformerBase):
    def __init__(self, worker_id, worker_name):
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.model = model
        self.names = self.model.names

        self.smoothing_history = []
        self.HISTORY = 7
        self.frame_counter = 0

        st.session_state.detected_live_ppe = set()
        st.session_state.last_update = time.time()
        st.session_state.force_rerun = False

    def smooth(self, detected):
        self.smoothing_history.append(detected)
        if len(self.smoothing_history) > self.HISTORY:
            self.smoothing_history.pop(0)

        smoothed = set()
        for it in PPE_ITEMS:
            cnt = sum(1 for h in self.smoothing_history if it in h)
            if cnt > self.HISTORY // 2:
                smoothed.add(it)

        return smoothed

    def run_yolo(self, frame):
        detected = set()
        result = self.model(frame, conf=0.35, verbose=False)[0]
        annotated = result.plot()

        for box in result.boxes:
            cls = int(box.cls)
            label = self.names.get(cls, "").lower()
            if label in CLASS_TO_PPE:
                detected.add(CLASS_TO_PPE[label])

        return detected, annotated

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.frame_counter % 5 == 0:
            try:
                raw_detect, annotated = self.run_yolo(rgb)
            except:
                raw_detect, annotated = set(), rgb

            stable = self.smooth(raw_detect)

            if stable != st.session_state.detected_live_ppe:
                st.session_state.detected_live_ppe = stable
                st.session_state.force_rerun = True
        else:
            annotated = rgb

        self.frame_counter += 1
        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

# ------------------------------------------------------------------------------
# LOGIN PAGE
# ------------------------------------------------------------------------------
def login_page():
    st.title("🔐 SiteSafe PPE Detector")

    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    # Sign In
    with tab1:
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")  # FIXED

        if st.button("Login"):
            if user in USER_DB and USER_DB[user] == pw:
                st.session_state.logged_in = True
                st.session_state.page = "workers"
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # Sign Up
    with tab2:
        new_user = st.text_input("New Username")
        new_pw = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Create Account"):
            if not new_user or not new_pw:
                st.error("Fields cannot be empty.")
            elif new_user in USER_DB:
                st.error("Username already exists.")
            elif new_pw != confirm:
                st.error("Passwords do not match.")
            else:
                USER_DB[new_user] = new_pw
                save_user_db(USER_DB)
                st.success("Account created. Please sign in.")

# ------------------------------------------------------------------------------
# WORKER PAGE
# ------------------------------------------------------------------------------
def worker_page():
    st.title("👷 Select Worker for PPE Inspection")

    worker_id = st.selectbox("Worker ID", list(WORKERS.keys()))
    worker_name = WORKERS[worker_id]

    st.write(f"Selected: **{worker_name}**")

    if st.button("Start Scanner"):
        st.session_state.worker_id = worker_id
        st.session_state.worker_name = worker_name
        st.session_state.page = "scanner"
        st.rerun()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

# ------------------------------------------------------------------------------
# SCANNER PAGE
# ------------------------------------------------------------------------------
def scanner_page():
    st.title("📹 PPE Live Scanner")

    wid = st.session_state.worker_id
    wname = st.session_state.worker_name

    st.subheader(f"Worker: **{wname}** ({wid})")

    st.button("⬅ Back", on_click=lambda: set_page("workers"))

    video_col, status_col = st.columns([2, 1])

    with video_col:
        webrtc_streamer(
            key="scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_transformer_factory=lambda: PPEVideoTransformer(wid, wname),
            async_transform=True,
        )

    if st.session_state.get("force_rerun"):
        st.session_state.force_rerun = False
        st.rerun()

    with status_col:
        st.markdown("### 📋 PPE Checklist")

        detected = st.session_state.get("detected_live_ppe", set())
        missing = [it for it in PPE_ITEMS if it not in detected]

        checklist = ""
        for it in PPE_ITEMS:
            if it in detected:
                checklist += f"<span style='color:green'>✔ **{it}**</span><br>"
            else:
                checklist += f"<span style='color:red'>❌ **{it}**</span><br>"

        st.markdown(checklist, unsafe_allow_html=True)

        if not detected:
            st.info("Click 'Start' below the video frame to begin scanning.")
        elif not missing:
            st.success("✅ FULLY COMPLIANT")
        else:
            st.error("🚨 NON-COMPLIANT")
            st.warning(f"Missing: {', '.join(missing)}")

# ------------------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------------------
def set_page(p):
    st.session_state.page = p
    st.rerun()

# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.page == "workers":
        worker_page()
    elif st.session_state.page == "scanner":
        scanner_page()
    else:
        st.session_state.page = "workers"
        st.rerun()
