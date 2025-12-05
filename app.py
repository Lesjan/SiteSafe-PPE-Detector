import streamlit as st
import cv2
import time
import os
import pickle
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration
import requests

# ----------------------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SiteSafe PPE Detector (Final Stable)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

LOG_FILE = "ppe_logs.csv"
VIOLATION_LOG = "violation_logs.csv"       # >>> NEW
USER_DB_FILE = "user_db.pkl"
MODEL_PATH = "best.pt"

MODEL_URL = "https://raw.githubusercontent.com/lesjan/SiteSafe-PPE-Detector/main/best.pt"


# ----------------------------------------------------------------------
# DOWNLOAD MODEL
# ----------------------------------------------------------------------
def download_model():
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 1000000:
            os.remove(MODEL_PATH)
        else:
            return

    try:
        st.info("Downloading PPE model... please wait.")
        r = requests.get(MODEL_URL, timeout=30)

        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

            if os.path.getsize(MODEL_PATH) < 1000000:
                raise ValueError("Model corrupted.")
        else:
            raise RuntimeError(f"HTTP ERROR {r.status_code}")

    except Exception as e:
        st.warning(f"⚠ Model download failed: {e}. Using YOLOv8n.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)


# ----------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    download_model()
    try:
        if os.path.exists(MODEL_PATH):
            return YOLO(MODEL_PATH)
        else:
            return YOLO("yolov8n.pt")
    except:
        return YOLO("yolov8n.pt")


model = load_model()


# ----------------------------------------------------------------------
# USER DB
# ----------------------------------------------------------------------
def load_user_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {"admin": "12345"}


def save_user_db(data):
    with open(USER_DB_FILE, "wb") as f:
        pickle.dump(data, f)


USER_DB = load_user_db()


# ----------------------------------------------------------------------
# WORKERS
# ----------------------------------------------------------------------
WORKERS = {
    "CW01": "Jasmin Romon",
    "CW02": "Cordel Kent Corona",
    "CW03": "Shine Acuña",
    "CW04": "Justin Baculio",
    "CW05": "Alexis Anne Emata",
}

# ----------------------------------------------------------------------
# PPE LIST
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
def init_log_files():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["timestamp", "worker_id", "worker_name"] + PPE_ITEMS)
        df.to_csv(LOG_FILE, index=False)

    if not os.path.exists(VIOLATION_LOG):   # >>> NEW
        df = pd.DataFrame(columns=["timestamp", "worker_id", "worker_name", "missing"])
        df.to_csv(VIOLATION_LOG, index=False)


init_log_files()


def save_inspection(worker_id, worker_name, detected):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {"timestamp": timestamp, "worker_id": worker_id, "worker_name": worker_name}
    for item in PPE_ITEMS:
        row[item] = 1 if item in detected else 0

    df = pd.read_csv(LOG_FILE)
    df.loc[len(df)] = row
    df.to_csv(LOG_FILE, index=False)


def save_violation(worker_id, worker_name, missing):  # >>> NEW
    if not missing:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_csv(VIOLATION_LOG)
    df.loc[len(df)] = {
        "timestamp": timestamp,
        "worker_id": worker_id,
        "worker_name": worker_name,
        "missing": ", ".join(missing)
    }
    df.to_csv(VIOLATION_LOG, index=False)


# ----------------------------------------------------------------------
# VIDEO TRANSFORMER
# ----------------------------------------------------------------------
class PPEVideoTransformer(VideoTransformerBase):
    def __init__(self, worker_id, worker_name):
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.model = model
        self.names = self.model.names

        self.history = []
        self.HISTORY = 7

        if "detected_live_ppe" not in st.session_state:
            st.session_state.detected_live_ppe = set()

    def smooth(self, detected):
        self.history.append(detected)
        if len(self.history) > self.HISTORY:
            self.history.pop(0)

        stable = set()
        for i in PPE_ITEMS:
            cnt = sum(1 for h in self.history if i in h)
            if cnt > self.HISTORY // 2:
                stable.add(i)
        return stable

    def run_yolo(self, frame):
        detected = set()
        result = self.model(frame, conf=0.5, verbose=False)[0]
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

        detected, annotated = self.run_yolo(rgb)
        stable = self.smooth(detected)

        st.session_state.detected_live_ppe = stable

        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------
# LOGIN PAGE
# ----------------------------------------------------------------------
def login_page():
    st.title("🔐 SiteSafe PPE Detector")

    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")

        if st.button("Login"):
            if user in USER_DB and USER_DB[user] == pw:
                st.session_state.logged_in = True
                st.session_state.page = "workers"
                st.rerun()
            else:
                st.error("Incorrect login.")

    with tab2:
        new_user = st.text_input("New Username")
        new_pw = st.text_input("New Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Create Account"):
            if not new_user:
                st.error("Missing username.")
            elif new_pw != confirm:
                st.error("Passwords don't match.")
            elif new_user in USER_DB:
                st.error("User exists.")
            else:
                USER_DB[new_user] = new_pw
                save_user_db(USER_DB)
                st.success("Account created.")


# ----------------------------------------------------------------------
# WORKER PAGE
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# SCANNER PAGE
# ----------------------------------------------------------------------
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

    with status_col:
        st.markdown("### 📋 PPE Checklist")

        detected = st.session_state.get("detected_live_ppe", set())
        missing = [i for i in PPE_ITEMS if i not in detected]

        checklist = ""
        for it in PPE_ITEMS:
            if it in detected:
                checklist += f"<span style='color:green'>✔ {it}</span><br>"
            else:
                checklist += f"<span style='color:red'>❌ {it}</span><br>"

        st.markdown(checklist, unsafe_allow_html=True)

        # >>> NEW — Save inspection button
        if st.button("💾 Save Inspection"):
            save_inspection(wid, wname, detected)
            save_violation(wid, wname, missing)
            st.success("Inspection saved!")

        # >>> NEW — Download buttons
        st.markdown("### 📥 Download Logs")

        ppe_log = pd.read_csv(LOG_FILE)
        viol_log = pd.read_csv(VIOLATION_LOG)

        st.download_button(
            "⬇ Download PPE Inspection Log (CSV)",
            data=ppe_log.to_csv(index=False),
            file_name="ppe_inspections.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇ Download Violations Log (CSV)",
            data=viol_log.to_csv(index=False),
            file_name="ppe_violations.csv",
            mime="text/csv"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def set_page(p):
    st.session_state.page = p
    st.rerun()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
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
