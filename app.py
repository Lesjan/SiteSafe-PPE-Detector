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
import plotly.express as px
import plotly.graph_objects as go

# ============================
# PAGE SETUP
# ============================
st.set_page_config(
    page_title="SiteSafe PPE Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# FILE PATHS
# ============================
LOG_FILE = "ppe_logs.csv"
VIOLATION_LOG = "violation_logs.csv"
USER_DB_FILE = "user_db.pkl"
MODEL_PATH = "best.pt"

MODEL_URL = "https://raw.githubusercontent.com/lesjan/SiteSafe-PPE-Detector/main/best.pt"

# ============================
# DOWNLOAD MODEL
# ============================
def download_model():
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 1000000:
            os.remove(MODEL_PATH)
        else:
            return
    try:
        st.info("Downloading PPE model...")
        r = requests.get(MODEL_URL, timeout=30)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            if os.path.getsize(MODEL_PATH) < 1000000:
                raise ValueError("Model corrupted")
        else:
            raise RuntimeError(f"HTTP {r.status_code}")
    except Exception as e:
        st.warning(f"⚠ Model download failed: {e}. Using YOLOv8n.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

# ============================
# LOAD MODEL
# ============================
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

# ============================
# USER DATABASE
# ============================
def load_user_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {"admin": "12345"}

def save_user_db(data):
    with open(USER_DB_FILE, "wb") as f:
        pickle.dump(data, f)

USER_DB = load_user_db()

# ============================
# WORKERS
# ============================
WORKERS = {
    "CW01": "Jasmin Romon",
    "CW02": "Cordel Kent Corona",
    "CW03": "Shine Acuña",
    "CW04": "Justin Baculio",
    "CW05": "Alexis Anne Emata",
}

# ============================
# PPE ITEMS
# ============================
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
    "Hardhat": "Hard Hat",
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

# ============================
# LOGGING
# ============================
def init_log_files():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["timestamp", "worker_id", "worker_name"] + PPE_ITEMS)
        df.to_csv(LOG_FILE, index=False)
    if not os.path.exists(VIOLATION_LOG):
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

def save_violation(worker_id, worker_name, missing):
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

# ============================
# HELPER FUNCTIONS
# ============================
def set_page(page_name):
    st.session_state.page = page_name

def get_sidebar_selection():
    if "sidebar_selection" not in st.session_state:
        st.session_state.sidebar_selection = "Dashboard"
    return st.session_state.sidebar_selection

# ============================
# LOGIN PAGE
# ============================
def login_page():
    st.markdown("<h1 style='text-align:center;'>🔐 SiteSafe PPE Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Please sign in to continue</h3>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        user = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", key="login_btn"):
            if user in USER_DB and USER_DB[user] == pw:
                st.session_state.logged_in = True
                set_page("Workers")
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

# ============================
# WORKER SELECTION PAGE
# ============================
def worker_page():
    st.title("👷 Select Worker for PPE Inspection")
    worker_id = st.selectbox("Worker ID", list(WORKERS.keys()))
    worker_name = WORKERS[worker_id]
    st.write(f"Selected: **{worker_name}**")
    
    if st.button("Start Scanner"):
        st.session_state.worker_id = worker_id
        st.session_state.worker_name = worker_name
        set_page("Scanner")

# ============================
# VIDEO TRANSFORMER FOR SCANNER
# ============================
class PPEVideoTransformer(VideoTransformerBase):
    def __init__(self, worker_id, worker_name):
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.model = model
        self.names = self.model.names
        self.HISTORY = 7
        self.smoothing_history = []

        if "detected_live_ppe" not in st.session_state:
            st.session_state.detected_live_ppe = set()

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
        try:
            raw_detect, annotated = self.run_yolo(rgb)
        except:
            raw_detect, annotated = set(), rgb
        stable_detect = self.smooth(raw_detect)
        st.session_state.detected_live_ppe = stable_detect
        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

# ============================
# SCANNER PAGE
# ============================
def scanner_page():
    st.title("📹 PPE Live Scanner")
    
    if "worker_id" not in st.session_state:
        st.warning("Select a worker first!")
        return

    worker_id = st.session_state.worker_id
    worker_name = st.session_state.worker_name

    st.subheader(f"Worker: **{worker_name} ({worker_id})**")

    video_col, status_col = st.columns([2, 1])
    with video_col:
        webrtc_streamer(
            key="scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_transformer_factory=lambda: PPEVideoTransformer(worker_id, worker_name),
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
            save_inspection(worker_id, worker_name, detected)
            save_violation(worker_id, worker_name, missing)
            st.success("Inspection saved!")
            if missing:
                st.warning(f"Missing PPE: {', '.join(missing)}")

        st.download_button(
            label="Download PPE Log CSV",
            data=open(LOG_FILE, "rb"),
            file_name="ppe_logs.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Violation CSV",
            data=open(VIOLATION_LOG, "rb"),
            file_name="ppe_violations.csv",
            mime="text/csv"
        )

# ============================
# DASHBOARD PAGE
# ============================
def dashboard_page():
    st.title("📊 SiteSafe Dashboard")
    st.markdown("### Overview of PPE Compliance")
    
    # Load logs
    df = pd.read_csv(LOG_FILE)
    violations_df = pd.read_csv(VIOLATION_LOG)

    total_inspections = len(df)
    total_violations = len(violations_df)

    st.markdown(f"**Total Inspections:** {total_inspections}")
    st.markdown(f"**Total Violations:** {total_violations}")

    # Most-missed PPE
    missed_counts = {}
    for item in PPE_ITEMS:
        missed_counts[item] = df[item].apply(lambda x: 0 if x==1 else 1).sum()
    missed_df = pd.DataFrame({"PPE": list(missed_counts.keys()), "Missed": list(missed_counts.values())})
    
    fig1 = px.bar(missed_df, x="PPE", y="Missed", color="Missed", color_continuous_scale="Oranges")
    st.plotly_chart(fig1, use_container_width=True)

    # Violations by Worker
    if not violations_df.empty:
        worker_counts = violations_df['worker_name'].value_counts().reset_index()
        worker_counts.columns = ["Worker", "Violations"]
        fig2 = px.pie(worker_counts, names="Worker", values="Violations", color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Trend over time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    trend_df = df.groupby(df['timestamp'].dt.date).apply(lambda x: (x[PPE_ITEMS]==0).any(axis=1).sum()).reset_index()
    trend_df.columns = ["Date", "Violations"]
    fig3 = px.line(trend_df, x="Date", y="Violations", markers=True)
    fig3.update_traces(line=dict(color='orange'))
    st.plotly_chart(fig3, use_container_width=True)

# ============================
# WORKER VIOLATION HISTORY
# ============================
def worker_history_page():
    st.title("👷 Worker Violation History")
    worker_id = st.selectbox("Select Worker", list(WORKERS.keys()))
    worker_name = WORKERS[worker_id]
    
    st.subheader(f"History for: **{worker_name} ({worker_id})**")
    
    violations_df = pd.read_csv(VIOLATION_LOG)
    worker_df = violations_df[violations_df['worker_id'] == worker_id]
    
    if worker_df.empty:
        st.info("No violations recorded for this worker.")
    else:
        st.dataframe(worker_df.sort_values("timestamp", ascending=False))


# ============================
# INITIALIZE SESSION STATE
# ============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "worker_id" not in st.session_state:
    st.session_state.worker_id = None
if "worker_name" not in st.session_state:
    st.session_state.worker_name = None

# ============================
# SIDEBAR NAVIGATION
# ============================
def sidebar():
    st.sidebar.title("🛠 SiteSafe Navigation")
    if not st.session_state.logged_in:
        return

    menu_options = ["Dashboard", "Workers", "Scanner", "Worker History", "Logout"]
    if st.session_state.worker_id is None:
        menu_options.remove("Scanner")  # Disable scanner if no worker selected

    choice = st.sidebar.radio("Go to:", menu_options, index=menu_options.index(st.session_state.page) if st.session_state.page in menu_options else 0)
    if choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.session_state.worker_id = None
        st.session_state.worker_name = None
    else:
        st.session_state.page = choice

# ============================
# RENDER PAGES BASED ON STATE
# ============================
def render_page():
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.page == "Dashboard":
            dashboard_page()
        elif st.session_state.page == "Workers":
            worker_page()
        elif st.session_state.page == "Scanner":
            scanner_page()
        elif st.session_state.page == "Worker History":
            worker_history_page()
        else:
            st.session_state.page = "Dashboard"
            dashboard_page()

# ============================
# MAIN APP
# ============================
sidebar()
render_page()



