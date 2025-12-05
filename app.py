import streamlit as st
import cv2
import time
import os
import pickle
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
import requests
import threading
import queue

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

# Enhanced CLASS_TO_PPE mapping - VERY COMPREHENSIVE
CLASS_TO_PPE = {
    # Hard Hat variations (try everything!)
    "hardhat": "Hard Hat",
    "hard-hat": "Hard Hat",
    "hard_hat": "Hard Hat",
    "helmet": "Hard Hat",
    "hard hat": "Hard Hat",
    "hat": "Hard Hat",
    
    # Safety Vest variations
    "vest": "Safety Vest",
    "safety vest": "Safety Vest",
    "safety-vest": "Safety Vest",
    "safety_vest": "Safety Vest",
    "hi-vis": "Safety Vest",
    "hiviz": "Safety Vest",
    "hi vis": "Safety Vest",
    "reflective vest": "Safety Vest",
    
    # Gloves variations
    "glove": "Gloves",
    "gloves": "Gloves",
    
    # Boots variations
    "boot": "Safety Boots",
    "boots": "Safety Boots",
    "safety boots": "Safety Boots",
    "safety-boots": "Safety Boots",
    "safety_boots": "Safety Boots",
    
    # Eye/Face Protection variations
    "goggles": "Eye/Face Protection",
    "glasses": "Eye/Face Protection",
    "mask": "Eye/Face Protection",
    "face shield": "Eye/Face Protection",
    "eye protection": "Eye/Face Protection",
    "eye/face protection": "Eye/Face Protection",
    "safety glasses": "Eye/Face Protection",
    
    # Hearing Protection variations
    "earmuff": "Hearing Protection",
    "earmuffs": "Hearing Protection",
    "ear protection": "Hearing Protection",
    "ear_protection": "Hearing Protection",
    "hearing protection": "Hearing Protection",
    "ear muff": "Hearing Protection",
    
    # Safety Harness variations
    "harness": "Safety Harness",
    "safety harness": "Safety Harness",
    "safety-harness": "Safety Harness",
    "safety_harness": "Safety Harness",
}

WORKERS = {
    "CW01": "Jasmin Romon",
    "CW02": "Cordel Kent Corona",
    "CW03": "Shine Acuña",
    "CW04": "Justin Baculio",
    "CW05": "Alexis Anne Emata",
}   

# Thread-safe result queue
result_queue = queue.Queue(maxsize=1)

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
            m = YOLO(MODEL_PATH)
            # Print all class names for debugging
            st.write("**Model loaded! Class names:**")
            class_info = []
            for idx, name in m.names.items():
                class_info.append(f"{idx}: '{name}'")
            st.code("\n".join(class_info))
            return m
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

# ----- Video Processor -----
class PPEVideoProcessor(VideoProcessorBase):
    def __init__(self, worker_id, worker_name):
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.model = model
        self.names = self.model.names

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            detected = set()
            result = self.model(rgb, conf=0.5, verbose=False)[0]
            annotated = result.plot()
            
            debug_info = []
            debug_info.append(f"Total detections: {len(result.boxes)}")
            
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                original_label = self.names.get(cls, "unknown")
                
                # Try multiple lowercase variations
                label_lower = original_label.lower().strip()
                label_no_underscore = label_lower.replace("_", " ")
                label_no_dash = label_lower.replace("-", " ")
                label_no_space = label_lower.replace(" ", "")
                
                debug_info.append(f"\n[Detection {cls}]")
                debug_info.append(f"  Original: '{original_label}'")
                debug_info.append(f"  Lowercase: '{label_lower}'")
                debug_info.append(f"  Confidence: {conf:.2f}")
                
                # Try all variations
                mapped_ppe = None
                for variant in [label_lower, label_no_underscore, label_no_dash, label_no_space]:
                    if variant in CLASS_TO_PPE:
                        mapped_ppe = CLASS_TO_PPE[variant]
                        break
                
                if mapped_ppe:
                    detected.add(mapped_ppe)
                    debug_info.append(f"  ✅ MAPPED TO: {mapped_ppe}")
                else:
                    debug_info.append(f"  ❌ NOT MAPPED")
                    debug_info.append(f"  💡 Add this to CLASS_TO_PPE:")
                    debug_info.append(f"     '{label_lower}': 'Hard Hat'  # or appropriate PPE")
            
            # Put result in queue (non-blocking, replace old)
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
            result_queue.put({"detected": detected, "debug": debug_info})
            
        except Exception as e:
            annotated = rgb
            debug_info = [f"ERROR: {str(e)}"]
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
            result_queue.put({"detected": set(), "debug": debug_info})
        
        from av import VideoFrame
        return VideoFrame.from_ndarray(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), format="bgr24")

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
            video_processor_factory=lambda: PPEVideoProcessor(wid, wname),
            async_processing=True,
        )

    with status_col:
        # Try to get latest detection result
        detected = set()
        debug_info = []
        
        try:
            result = result_queue.get_nowait()
            detected = result["detected"]
            debug_info = result["debug"]
        except queue.Empty:
            pass
        
        missing = [it for it in PPE_ITEMS if it not in detected]

        # Display current detections
        st.markdown("### 🔍 Live Detection")
        if detected:
            for item in detected:
                st.success(f"✅ **{item}**")
        else:
            st.info("⏳ Waiting for detection...")

        # Display checklist
        st.markdown("### 📋 PPE Checklist")
        for it in PPE_ITEMS:
            if it in detected:
                st.markdown(f"<span style='color:green; font-size:16px'>✔ **{it}**</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red; font-size:16px'>❌ **{it}**</span>", unsafe_allow_html=True)

        # Show debug information
        if debug_info:
            with st.expander("🐛 Debug Info - What Model Sees", expanded=True):
                st.code("\n".join(debug_info))

        st.markdown("---")
        
        if st.button("💾 Save Inspection", type="primary"):
            log_inspection(wid, wname, detected)
            log_violation(wid, wname, missing)
            st.success("✅ Inspection saved!")
            if missing:
                st.warning(f"⚠️ Missing: {', '.join(missing)}")

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "rb") as f:
                    st.download_button("📥 Logs", f, file_name="ppe_logs.csv", mime="text/csv")
        with col2:
            if os.path.exists(VIOLATION_LOG):
                with open(VIOLATION_LOG, "rb") as f:
                    st.download_button("📥 Violations", f, file_name="ppe_violations.csv", mime="text/csv")
        
        # Refresh button for manual update
        if st.button("🔄 Refresh Detection"):
            st.rerun()

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
