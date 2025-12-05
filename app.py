# ----------------------
    # UI: Scanner page (MODIFIED FOR STREAMLIT CLOUD/FILE UPLOAD)
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

        st.title("📸 Static PPE Scanner (Cloud Ready)")
        st.caption("Upload an image to check worker PPE compliance.")

        worker_id = st.session_state.get("worker_id")
        worker_name = st.session_state.get("worker_name")
        st.subheader(f"Target: **{worker_name}** ({worker_id})")

        # Back button is now the only control
        if st.button("⬅️ Back to Worker Selection", key="back_btn", on_click=go_back):
            pass # Button handler calls go_back()

        st.markdown("---")
        
        # --- FILE UPLOADER REPLACES START/STOP CAMERA BUTTONS ---
        uploaded_file = st.file_uploader(
            "Choose a worker image (JPG/PNG)",
            type=["jpg", "jpeg", "png"]
        )
        
        # Setup main image and checklist areas
        if uploaded_file is not None:
            # Read the uploaded file into a NumPy array format suitable for OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Setup columns for display
            image_col, status_col = st.columns([2, 1])

            with image_col:
                st.image(frame_rgb, caption="Uploaded Image", channels="RGB")
                
            with status_col:
                # Run the detection once
                detected = detect_ppe(frame_rgb)
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
                if not missing:
                    st.success("✅ **FULLY COMPLIANT**")
                    compliance_status = "FULLY COMPLIANT"
                else:
                    st.error("🚨 **NON-COMPLIANT**")
                    st.warning(f"Missing: {', '.join(missing)}")
                    compliance_status = "NON-COMPLIANT"

                # --- Manual Log Button (Replaces Auto-Log) ---
                if st.button("💾 Log Inspection Result", key="log_btn"):
                    log_inspection(worker_id, worker_name, detected)
                    st.success(f"Log recorded: {compliance_status} for {worker_name}")
                    st.balloons() # Celebrate a logged success (or lack of compliance!)
        
        elif st.session_state.get("inspection_complete"):
            st.info("Log recorded. Please upload a new image to continue inspection.")
        else:
            st.info("Upload an image of the worker to begin the PPE compliance scan.")
