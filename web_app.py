import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import collections
import copy
import itertools
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 1. ตั้งค่าสถานะเริ่มต้น ---
if "last_msg" not in st.session_state:
    st.session_state.last_msg = "รอตรวจจับ..."
if "history" not in st.session_state:
    st.session_state.history = collections.deque(maxlen=10)

# --- 2. โหลดทรัพยากร (เพิ่ม error handling) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_resources():
    try:
        with open(model_path, 'rb') as f:
            m = pickle.load(f)
            model_obj = m['model'] if isinstance(m, dict) else m
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, -1].astype(str).tolist()
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None, [], None, None, None

    mp_hands = mp.solutions.hands
    hands_engine = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.4, # ปรับค่าให้เหมาะสมกับสภาพแสง
        min_tracking_confidence=0.4
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 3. ฟังก์ชันเตรียมข้อมูล (Pre-process) ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_val = max(list(map(abs, temp_landmark_list)))
    return [n / max_val if max_val != 0 else 0 for n in temp_landmark_list]

# --- 4. ฟังก์ชันจัดการวิดีโอ (Callback) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # ลดความละเอียดภาพก่อนส่งเข้า Mediapipe เพื่อแก้การค้าง
    img_small = cv2.resize(img, (320, 240))
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hl = results.multi_hand_landmarks[0]
        # วาดจุดบนภาพจริงเพื่อให้ผู้ใช้เห็นว่าระบบ "เจอมือแล้ว"
        mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
        
        # เก็บพิกัด Motion
        p9 = hl.landmark[9]
        st.session_state.history.append((p9.x, p9.y))

        # ตรวจจับการส่ายมือ (ไม่)
        if len(st.session_state.history) == 10:
            dx = st.session_state.history[-1][0] - st.session_state.history[0][0]
            if abs(dx) > 0.12:
                st.session_state.last_msg = "ไม่"
                return frame.from_ndarray(img, format="bgr24")

        # ตรวจจับท่าทางนิ่งด้วย AI
        landmark_list = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
        processed_data = pre_process_landmark(landmark_list)
        prediction = model.predict(np.array([processed_data]))[0]
        conf = model.predict_proba(np.array([processed_data])).max()
        
        if conf > 0.7:
            st.session_state.last_msg = labels[int(prediction)]

    return frame.from_ndarray(img, format="bgr24")

# --- 5. หน้าตา UI ---
st.title("🖐️ ระบบแปลภาษามือไทย")

placeholder = st.empty()
placeholder.markdown(
    f"""
    <div style="background-color: #1e1e1e; color: #00ff00; padding: 20px; border-radius: 15px; border: 2px solid #00ff00; text-align: center; margin-bottom: 10px;">
        <h1 style="margin: 0; font-size: 60px;">{st.session_state.last_msg}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

webrtc_streamer(
    key="fix-freeze-v5", # เปลี่ยน Key ทุกครั้งที่แก้โค้ดเพื่อ Reset Browser
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": 480, # ลดความละเอียดลงเพื่อความลื่นไหล
            "height": 360,
            "frameRate": 10
        },
        "audio": False
    },
    async_processing=True,
)
