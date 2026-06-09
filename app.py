import cv2
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from ultralytics import YOLO

#  CORE PAGE UI STYLING & SETUP
st.set_page_config(layout="wide")
st.title("⚡GridGuard: Edge-AI Substation Thermography & Automated Predictive Maintenance Infrastructure")
st.markdown("""
This production dashboard combines **YOLOv8 Object Localization** with automated 
**Computer Vision Thermal Profiling** to monitor electrical grid health indices in real-time.
""")

# Local Relative Path targeting  weights folder
MODEL_PATH = 'weights/best.pt'

@st.cache_resource
def load_production_model():
    """Cache the model in RAM so it doesn't reload on every browser refresh"""
    return YOLO(MODEL_PATH)

try:
    model = load_production_model()
except Exception as e:
    st.error(f"❌ Failed to load model weights from '{MODEL_PATH}'. Please ensure 'best.pt' is placed inside the 'weights' folder.")
    st.stop()


#  FILE INGESTION SIDEBAR PANEL

st.sidebar.header("Asset Ingestion Panel")
uploaded_file = st.sidebar.file_uploader(
    "Upload Infrared Substation Image", 
    type=["jpg", "jpeg", "png"]
)


# COMPUTE ENGINE AND PIPELINE LOGIC

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    original_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    display_img = original_bgr.copy()
 
    results = model(display_img, verbose=False,iou=0.03,conf=0.50)[0]
    
    ticket_accumulator = []
    
    for box in results.boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        class_name = model.names[cls_id].upper()  
        
        cv2.rectangle(display_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
        
        cropped_asset = original_bgr[ymin:ymax, xmin:xmax]
        if cropped_asset.size == 0: continue
        
        gray_crop = cv2.cvtColor(cropped_asset, cv2.COLOR_BGR2GRAY)
        I_ambient = np.mean(gray_crop) or 1
        blurred = cv2.GaussianBlur(gray_crop, (3, 3), 0)
        _, I_max, _, max_loc = cv2.minMaxLoc(blurred)
        
        severity_index = I_max / I_ambient
        
        urgency = "NORMAL"
        if severity_index >= 1.95: urgency = "CRITICAL"
        elif severity_index >= 1.65: urgency = "MEDIUM"
        elif severity_index >= 1.30: urgency = "LOW"
            
        #  DYNAMIC LABEL & PLACEMENT ENGINE
        
    
        if urgency != "NORMAL":
            label_text = f"{class_name} [{urgency}] S={severity_index:.2f}"
            label_color = (0, 0, 255) 
            
            
            global_hotspot_x = xmin + max_loc[0]
            global_hotspot_y = ymin + max_loc[1]
            cv2.circle(display_img, (global_hotspot_x, global_hotspot_y), 12, (0, 0, 255), -1)
            
           
            ticket_data = {
                "id": f"TICK-{datetime.now().strftime('%M%S')}-{class_name[:3]}",
                "asset": class_name,
                "score": severity_index,
                "urgency": urgency,
                "bounds": f"[{xmin}, {ymin}, {xmax}, {ymax}]",
                "pixel": f"({global_hotspot_x}, {global_hotspot_y})"
            }
            ticket_accumulator.append(ticket_data)
        else:
            label_text = f"{class_name} [OK]"
            label_color = (0, 255, 0) 

        if ymin - 15 < 15:
            text_y_position = ymin + 25 
        else:
            text_y_position = ymin - 10 

        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            display_img, 
            (xmin, text_y_position - text_height - 4), 
            (xmin + text_width + 6, text_y_position + 4), 
            (20, 20, 20), 
            -1
        )

        cv2.putText(
            display_img, 
            label_text, 
            (xmin + 3, text_y_position),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            label_color, 
            2, 
            cv2.LINE_AA
        )

    #  STREAMLIT LAYOUT DISPLAY
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Computer Vision Tracking Output")
        rgb_render = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_render, use_container_width=True)
        
    with col2:
        st.subheader("🎫 Automated Fleet Dispatch System")
        
        if not ticket_accumulator:
            st.success("✅ Diagnostic Report: All components operating within normal thermal ranges.")
        else:
            for ticket in ticket_accumulator:
                with st.expander(f"🛑 {ticket['urgency']} ALERT - {ticket['asset']} ({ticket['id']})", expanded=True):
                    st.markdown(f"""
                    * **System Severity Index ($S$):** `{ticket['score']:.4f}`
                    * **Asset Class Detection Box:** `{ticket['bounds']}`
                    * **Thermal Hotspot Core Intersection:** `{ticket['pixel']}`
                    """)
                    if ticket["urgency"] == "CRITICAL":
                        st.error("🚨 EMERGENCY ACTION: Dispatch immediate grid maintenance response. Asset localized failure probability is high.")
                    else:
                        st.warning("⚠️ SCHEDULE ROUTINE AUDIT: Plan component field inspection window within 48 operating hours.")
else:
    st.info("💡 Dashboard Active: Upload an uncalibrated thermal infrared frame from the sidebar menu to begin automated profiling.")