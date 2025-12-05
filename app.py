import streamlit as st
import cv2
import time
from datetime import datetime
import pandas as pd
import os
import pickle
import random
import multiprocessing

# --- GLOBAL PAGE CONFIGURATION (FOR MOBILE/WIDE VIEW) ---
# This must be the first Streamlit command
st.set_page_config(
    page_title="SiteSafe PPE Detector", 
    layout="wide", # Use full screen width
    # 'auto' is the default and hides the sidebar on mobile-sized devices
    initial_sidebar_state="collapsed" 
) 

# ----------------------
# PYINSTALLER FIX (CRITICAL FOR .EXE)
# ----------------------
if __name__ == '__main__':
    multiprocessing.freeze_support() 
    
    # ----------------------
    # CONFIG
    # ----------------------
    LOG_FILE = "ppe_app_logs.csv"
    USER_DB_FILE = "user_credentials.pkl"
    FRAME_SKIP = 3
    AUTO_LOG_INTERVAL = 5 
    
    # ----------------------
    # MODEL (try to load YOLO, otherwise simulated)
    # ----------------------
    MODEL_PATH = "best.pt" # <--- IMPORTANT: Ensure this matches your downloaded model file name
    USE_SIMULATED = False
    model = None

    try:
        from ultralytics import YOLO
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            model = YOLO("yolov8n.pt")
    except Exception as e:
        USE_SIMULATED = True
        
    # --- HANDLERS ---
    def go_back():
        st.session_state.page = "worker"
        st.session_state.run_camera = False
        st.rerun() 

    def start_camera_handler():
        st.session_state.run_camera = True

    def stop_camera_handler():
        st.session_state.run_camera = False
        st.rerun() 

    # ----------------------
    # USER DB helpers (unchanged)
    # ----------------------
    def load_user_db():
        if os.path.exists(USER_DB_FILE):
            with open(USER_DB_FILE, "rb") as f:
                return pickle.load(f)
        return {"admin": "12345"}

    def save_user_db(db):
        with open(USER_DB_FILE, "wb") as f:
            pickle.dump(db, f)

    USER_DB = load_user_db()

    # ----------------------
    # WORKERS & PPE MAPPING (unchanged)
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
    # LOGGING (unchanged)
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
    # DETECTION (CPU Optimized)
    # ----------------------
    def simulated_detect(frame):
        present = set()
        if random.random() > 0.2: present.add("Hard Hat")
        if random.random() > 0.3: present.add("Safety Vest")
        if random.random() > 0.6: present.add("Gloves")
        if random.random() > 0.5: present.add("Safety Boots")
        if random.random() > 0.8: present.add("Eye/Face Protection")
        if random.random() > 0.9: present.add("Hearing Protection")
        if random.random() > 0.95: present.add("Safety Harness")
        return present

    def detect_ppe(frame):
        if USE_SIMULATED or model is None:
            return simulated_detect(frame)

        detected = set()
        try:
            # OPTIMIZATION: Explicitly set device='cpu' for packaged apps
            results = model(
                frame, 
                device='cpu', 
                imgsz=640, 
                conf=0.5 # Confidence threshold for reliable detection
            )[0]
            
            names = model.names if hasattr(model, "names") else {}
            for box in results.boxes:
                try:
                    cls_id = int(box.cls)
                    label = names.get(cls_id, str(cls_id)).lower()
                except Exception:
                    label = str(box.cls)
                if label in CLASS_TO_PPE:
                    detected.add(CLASS_TO_PPE[label])
        except Exception as e:
            return simulated_detect(frame)
        return detected

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
    # UI: Worker selection (Modified to remove history)
    # ----------------------
    def worker_page():
        st.title("👷 SiteSafe - Worker & Supervisor View")

        # Display logged-in user at the top (replaces st.sidebar.write)
        st.subheader(f"👋 Logged In User: {st.session_state.get('user_name', 'UNKNOWN')}")
        
        # Logout button moved to main area
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun() 

        st.markdown("---")
        
        # Worker Selection
        choice = st.selectbox("Select Worker ID to Inspect", list(WORKERS.keys()))
        worker_name = WORKERS[choice]
        st.write(f"**Worker Name:** {worker_name}")

        if st.button("Proceed to PPE Scanner"):
            st.session_state.worker_id = choice
            st.session_state.worker_name = worker_name
            st.session_state.page = "scanner"
            st.session_state.inspection_complete = False
            st.session_state.run_camera = False
            st.rerun() 

        # Compliance history section removed as requested
        # st.subheader("Compliance History") 

    # ----------------------
    # UI: Scanner page (Mobile Refined)
    # ----------------------
    def scanner_page():
        # Hide the sidebar button (even though the sidebar is collapsed by default on mobile)
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {display: none;}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.title("🎥 Live PPE Scanner")

        worker_id = st.session_state.get("worker_id")
        worker_name = st.session_state.get("worker_name")
        st.subheader(f"Target: **{worker_name}** ({worker_id})")

        # Use two columns for the button controls
        col_start, col_stop, col_back = st.columns([1, 1, 1])

        with col_start:
            st.button("🟢 Start Scan", key="start_btn", on_click=start_camera_handler, disabled=st.session_state.get("run_camera", False))
        with col_stop:
            st.button("🟥 Stop Scan", key="stop_btn", on_click=stop_camera_handler, disabled=not st.session_state.get("run_camera", False))
        with col_back:
            st.button("⬅️ Back", key="back_btn", on_click=go_back)

        st.markdown("---")

        # Setup main video and checklist areas
        video_col, status_col = st.columns([2, 1]) # Keep the 2:1 ratio for desktop/laptop stability

        with video_col:
            frame_slot = st.empty()
            if USE_SIMULATED:
                st.warning("⚠️ Running in SIMULATED detection mode. Results are random.")

        with status_col:
            checklist_placeholder = st.empty()
            status_placeholder = st.empty()
            warning_placeholder = st.empty()

        run = st.session_state.get("run_camera", False)

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not open webcam.")
                st.session_state.run_camera = False
                return

            frame_counter = 0
            last_log = st.session_state.get("last_log_time", time.time())
            inspection_complete = st.session_state.get("inspection_complete", False)
            
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Camera error. Stopping feed.")
                    st.session_state.run_camera = False
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame_counter % FRAME_SKIP == 0:
                    detected = detect_ppe(frame_rgb)
                    missing = [it for it in PPE_ITEMS if it not in detected]

                    # --- UI Update (Checklist) ---
                    checklist_text = "### 📋 PPE Checklist\n"
                    for it in PPE_ITEMS:
                        # Displaying items in a compact, clear list
                        if it in detected:
                            checklist_text += f"**<span style='color:green'>✔ {it}</span>**\n"
                        else:
                            checklist_text += f"**<span style='color:red'>❌ {it}</span>**\n"
                    
                    checklist_placeholder.markdown(checklist_text, unsafe_allow_html=True)
                    
                    # --- UI Update (Status) ---
                    if not missing:
                        status_placeholder.success("✅ **FULLY COMPLIANT**")
                        warning_placeholder.empty()
                    else:
                        status_placeholder.error("🚨 **NON-COMPLIANT**")
                        warning_placeholder.warning(f"Missing: {', '.join(missing)}")

                    # --- Auto-log logic ---
                    now = time.time()
                    if now - last_log > AUTO_LOG_INTERVAL:
                        
                        if not missing and not inspection_complete:
                            log_inspection(worker_id, worker_name, detected)
                            st.session_state.inspection_complete = True
                            st.balloons()
                            status_placeholder.success("✅ **LOGGED!** Inspection complete.")
                            st.session_state.run_camera = False
                            st.rerun() 
                        
                        elif missing and not inspection_complete:
                            log_inspection(worker_id, worker_name, detected)
                            warning_placeholder.info(f"Non-compliant status logged at {datetime.now().strftime('%H:%M:%S')}. Still checking...")
                            
                        last_log = now
                        st.session_state.last_log_time = last_log

                # Display frame
                frame_slot.image(frame_rgb, channels="RGB")
                frame_counter += 1
                
            cap.release()
            
            if not st.session_state.get("run_camera"):
                st.info("Scan stopped.")


    # ----------------------
    # Main app flow
    # ----------------------
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

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
