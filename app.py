import streamlit as st
import cv2
import os
import pickle
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import threading
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

MODEL_URL = "https://raw.githubusercontent.com/lesjan/SiteSafe-PPE-Detector/main/best.pt"

def download_model():
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 1000000:
            os.remove(MODEL_PATH)
        else:
            return

    try:
        st.info("Downloading PPE model... please wait (one-time download).")
        r = requests.get(MODEL_URL, timeout=30)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            if os.path.getsize(MODEL_PATH) < 1000000:
                raise ValueError("Downloaded model appears corrupted.")
        else:
            raise RuntimeError(f"HTTP {r.status_code}")
    except Exception as e:
        st.warning(f"⚠ Model download failed: {e}. Falling back to YOLOv8n.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

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

USER_DB = {}
if os.path.exists(USER_DB_FILE):
    try:
        with open(USER_DB_FILE, "rb") as f:
            USER_DB = pickle.load(f)
    except:
        USER_DB = {"admin": "12345"}
else:
    USER_DB = {"admin": "12345"}

WORKERS = {
    "CW01": "Jasmin Romon",
    "CW02": "Cordel Kent Corona",
    "CW03": "Shine Acuña",
    "CW04": "Justin Baculio",
    "CW05": "Alexis Anne Emata",
}

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

# Logging init
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "worker_id", "worker_name"] + PPE_ITEMS).to_csv(LOG_FILE, index=False)

def log_inspection(worker_id, worker_name, detected):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": timestamp, "worker_id": worker_id, "worker_name": worker_name}
    for item in PPE_ITEMS:
        row[item] = 1 if item in detected else 0
    df = pd.read_csv(LOG_FILE)
    df.loc[len(df)] = row
    df.to_csv(LOG_FILE, index=False)

# Video processor class
class PPEVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.names = model.names
        self.smoothing_history = []
        self.HISTORY = 7
        self.detected = set()
        self.lock = threading.Lock()

    def smooth(self, detected):
        self.smoothing_history.append(detected)
        if len(self.smoothing_history) > self.HISTORY:
            self.smoothing_history.pop(0)
        smoothed = set()
        for item in PPE_ITEMS:
            count = sum(1 for d in self.smoothing_history if item in d)
            if count > self.HISTORY // 2:
                smoothed.add(item)
        return smoothed

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.5, verbose=False)[0]

        detected = set()
        for box in results.boxes:
            cls = int(box.cls)
            label = self.names.get(cls, "").lower()
            if label in CLASS_TO_PPE:
                detected.add(CLASS_TO_PPE[label])

        smoothed = self.smooth(detected)

        with self.lock:
            if smoothed != self.detected:
                self.detected = smoothed
                st.session_state.detected_live_ppe = smoothed
                st.session_state.force_rerun = True

        annotated = results.plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

def login_page():
    st.title("🔐 SiteSafe PPE Detector")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        user = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", key="login_btn"):
            if user in USER_DB and USER_DB[user] == pw:
                st.session_state.logged_in = True
                st.session_state.page = "workers"
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pw = st.text_input("New Password", type="password", key="signup_pw")
        confirm = st.text_input("Confirm Password", type="password", key="signup_pw_confirm")
        if st.button("Create Account", key="signup_btn"):
            if not new_user or not new_pw:
                st.error("Fields cannot be empty.")
            elif new_user in USER_DB:
                st.error("Username already exists.")
            elif new_pw != confirm:
                st.error("Passwords do not match.")
            else:
                USER_DB[new_user] = new_pw
                with open(USER_DB_FILE, "wb") as f:
                    pickle.dump(USER_DB, f)
                st.success("Account created. Please sign in.")

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

def scanner_page():
    st.title("📹 PPE Live Scanner")
    wid = st.session_state.worker_id
    wname = st.session_state.worker_name

    st.subheader(f"Worker: **{wname}** ({wid})")
    st.button("⬅ Back", on_click=lambda: set_page("workers"))

    video_col, status_col = st.columns([2, 1])

    with video_col:
        webrtc_ctx = webrtc_streamer(
            key="scanner",
            video_processor_factory=PPEVideoProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # Check for PPE detection update signal
    if st.session_state.get("force_rerun", False):
        st.session_state.force_rerun = False
        st.rerun()

    with status_col:
        st.markdown("### 📋 PPE Checklist")
        detected = st.session_state.get("detected_live_ppe", set())
        st.write("Detected PPE (stable):", detected)

        missing = [item for item in PPE_ITEMS if item not in detected]

        checklist_html = ""
        for item in PPE_ITEMS:
            if item in detected:
                checklist_html += f"<span style='color:green'>✔ **{item}**</span><br>"
            else:
                checklist_html += f"<span style='color:red'>❌ **{item}**</span><br>"

        st.markdown(checklist_html, unsafe_allow_html=True)

        if not detected:
            st.info("Click 'Start Scanner' to begin scanning.")
        elif not missing:
            st.success("✅ FULLY COMPLIANT")
        else:
            st.error("🚨 NON-COMPLIANT")
            st.warning(f"Missing: {', '.join(missing)}")

def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "detected_live_ppe" not in st.session_state:
    st.session_state.detected_live_ppe = set()
if "force_rerun" not in st.session_state:
    st.session_state.force_rerun = False

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
