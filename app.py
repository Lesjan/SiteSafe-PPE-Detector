
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

# Optional imports for dashboard
try:
    import plotly.express as px
except ImportError:
    px = None

# ----- Config -----
LOG_FILE = "ppe_logs.csv"
VIOLATION_LOG = "ppe_violations.csv"
USER_DB_FILE = "user_db.pkl"
MODEL_PATH = "best.pt"

MODEL_URL = "https://raw.githubusercontent.com/lesjan/SiteSafe-PPE-Detector/main/best.pt"

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

WORKERS = {
    "CW01": "Jasmin Romon",
    "CW02": "Cordel Kent Corona",
    "CW03": "Shine Acuña",
    "CW04": "Justin Baculio",
    "CW05": "Alexis Anne Emata",
}   

# ----- Download model if needed -----
def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000:
        return
    try:
        st.info("Downloading PPE model...")
        r = requests.get(MODEL_URL, timeout=30)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        else:
            st.warning("Failed to download model, using default YOLOv8n")
    except Exception as e:
        st.warning(f"Model download failed: {e}")

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

# ----- User DB -----
def load_user_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {"admin": "12345"}

def save_user_db(data):
    with open(USER_DB_FILE, "wb") as f:
        pickle.dump(data, f)

USER_DB = load_user_db()

# ----- Logs -----
def init_log_file():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["timestamp", "worker_id", "worker_name"] + PPE_ITEMS)
        df.to_csv(LOG_FILE, index=False)

    if not os.path.exists(VIOLATION_LOG):
        dfv = pd.DataFrame(columns=["timestamp", "worker_id", "worker_name", "missing_ppe"])
        dfv.to_csv(VIOLATION_LOG, index=False)

init_log_file()

def log_inspection(worker_id, worker_name, detected):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": timestamp, "worker_id": worker_id, "worker_name": worker_name}
    for item in PPE_ITEMS:
        row[item] = 1 if item in detected else 0
    df = pd.read_csv(LOG_FILE)
    df.loc[len(df)] = row
    df.to_csv(LOG_FILE, index=False)

def log_violation(worker_id, worker_name, missing_ppe):
    if not missing_ppe:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": timestamp,
        "worker_id": worker_id,
        "worker_name": worker_name,
        "missing_ppe": ", ".join(missing_ppe),
    }
    df = pd.read_csv(VIOLATION_LOG)
    df.loc[len(df)] = row
    df.to_csv(VIOLATION_LOG, index=False)

# ----- Video Transformer -----
class PPEVideoTransformer(VideoTransformerBase):
    def __init__(self, worker_id, worker_name):
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.model = model
        self.names = self.model.names
        self.smoothing_history = []
        self.HISTORY = 7

        if "detected_live_ppe" not in st.session_state:
            st.session_state.detected_live_ppe = set()

    def smooth(self, detected):
        self.smoothing_history.append(detected)
        if len(self.smoothing_history) > self.HISTORY:
            self.smoothing_history.pop(0)
        smoothed = set()
        for it in PPE_ITEMS:
            count = sum(1 for h in self.smoothing_history if it in h)
            if count > self.HISTORY // 2:
                smoothed.add(it)
        return smoothed

    def run_yolo(self, frame):
        detected = set()
        result = self.model(frame, conf=0.5, verbose=False)[0]
        annotated = result.plot()
        for box in result.boxes:
            cls = int(box.cls)
            label = self.names.get(cls, "")
            if label in CLASS_TO_PPE:
                detected.add(CLASS_TO_PPE[label])
        return detected, annotated

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            raw_detect, annotated = self.run_yolo(rgb)
        except Exception as e:
            raw_detect, annotated = set(), rgb

        stable_detect = self.smooth(raw_detect)
        st.session_state.detected_live_ppe = stable_detect
        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

# ----- Pages -----

def login_page():
    st.title("🔐 SiteSafe PPE Detector")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    with tab1:
        user = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", key="login_btn"):
            if user in USER_DB and USER_DB[user] == pw:
                st.session_state.logged_in = True
                st.session_state.page = "Workers"
                st.session_state.worker_id = None
                st.session_state.worker_name = None
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
                save_user_db(USER_DB)
                st.success("Account created. Please sign in.")

def worker_page():
    st.title("👷 Select Worker for PPE Inspection")
    worker_id = st.selectbox("Worker ID", list(WORKERS.keys()))
    worker_name = WORKERS[worker_id]
    st.write(f"Selected: **{worker_name}**")
    if st.button("Start Scanner"):
        st.session_state.worker_id = worker_id
        st.session_state.worker_name = worker_name
        st.session_state.page = "Scanner"
        st.rerun()

def scanner_page():
    st.title("📹 PPE Live Scanner")

    if "worker_id" not in st.session_state or st.session_state.worker_id is None:
        st.warning("Please select a worker first on the 'Workers' page.")
        return

    wid = st.session_state.worker_id
    wname = st.session_state.worker_name

    st.subheader(f"Worker: **{wname} ({wid})**")

    video_col, status_col = st.columns([2,1])
    with video_col:
        webrtc_streamer(
            key="scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_transformer_factory=lambda: PPEVideoTransformer(wid, wname),
            async_transform=True,
        )

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

        if st.button("Save Inspection"):
            log_inspection(wid, wname, detected)
            log_violation(wid, wname, missing)
            st.success("Inspection saved!")
            if missing:
                st.warning(f"Missing PPE: {', '.join(missing)}")

        # Download buttons
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button("Download PPE Logs CSV", f, file_name="ppe_logs.csv", mime="text/csv")
        if os.path.exists(VIOLATION_LOG):
            with open(VIOLATION_LOG, "rb") as f:
                st.download_button("Download Violation Logs CSV", f, file_name="ppe_violations.csv", mime="text/csv")

def dashboard_page():
    st.title("📊 SiteSafe PPE Compliance Dashboard")
    if px is None:
        st.warning("Plotly not installed. Install plotly for dashboard charts.")
        return

    df = pd.read_csv(LOG_FILE)
    violations_df = pd.read_csv(VIOLATION_LOG)

    st.markdown(f"**Total Inspections:** {len(df)}")
    st.markdown(f"**Total Violations:** {len(violations_df)}")

    # PPE missed counts
    missed_counts = {item: (df[item] == 0).sum() for item in PPE_ITEMS}
    missed_df = pd.DataFrame({"PPE": list(missed_counts.keys()), "Missed": list(missed_counts.values())})

    fig = px.bar(missed_df, x="PPE", y="Missed", color="Missed", color_continuous_scale="Oranges",
                 title="Most Missed PPE Items")
    st.plotly_chart(fig, use_container_width=True)

    # Violations by Worker Pie Chart
    if not violations_df.empty:
        worker_counts = violations_df['worker_name'].value_counts().reset_index()
        worker_counts.columns = ["Worker", "Violations"]
        fig2 = px.pie(worker_counts, names="Worker", values="Violations", title="Violations by Worker",
                      color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig2, use_container_width=True)

def worker_history_page():
    st.title("👷 Worker Violation History")
    worker_id = st.selectbox("Select Worker", list(WORKERS.keys()))
    worker_name = WORKERS[worker_id]

    violations_df = pd.read_csv(VIOLATION_LOG)
    worker_df = violations_df[violations_df['worker_id'] == worker_id]

    if worker_df.empty:
        st.info("No violations recorded for this worker.")
    else:
        st.dataframe(worker_df.sort_values("timestamp", ascending=False))

# ----- Sidebar -----
def sidebar():
    st.sidebar.title("🛠 SiteSafe Navigation")
    if not st.session_state.get("logged_in", False):
        return

    pages = ["Dashboard", "Workers", "Scanner", "Worker History", "Logout"]
    if st.session_state.get("worker_id") is None:
        pages.remove("Scanner")

    choice = st.sidebar.radio("Go to:", pages, index=pages.index(st.session_state.get("page", "Dashboard")))

    if choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.session_state.worker_id = None
        st.session_state.worker_name = None
        st.rerun()
    else:
        st.session_state.page = choice

# ----- Main app control -----
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "Login"
    if "worker_id" not in st.session_state:
        st.session_state.worker_id = None
    if "worker_name" not in st.session_state:
        st.session_state.worker_name = None

    sidebar()

    if not st.session_state.logged_in:
        login_page()
    else:
        page = st.session_state.page
        if page == "Dashboard":
            dashboard_page()
        elif page == "Workers":
            worker_page()
        elif page == "Scanner":
            scanner_page()
        elif page == "Worker History":
            worker_history_page()
        else:
            st.session_state.page = "Dashboard"
            dashboard_page()

if __name__ == "__main__":
    main()







