import streamlit as st

# --- 1. ตั้งค่าหน้าเว็บเพื่อ SEO (ต้องเป็นคำสั่งแรกสุดของ Streamlit) ---
st.set_page_config(
    page_title="แปลภาษามือไทยออนไลน์ - AI Sign Language Translator",
    page_icon="🖐️",
    layout="centered"
)

import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 2. ข้อความอธิบายสำหรับ Google (SEO Section) ---
st.title("🖐️ ระบบแปลภาษามือไทยแบบ Real-time")
st.markdown("""
### เครื่องมือช่วยแปลภาษามือไทยเป็นตัวอักษรด้วย AI
แอปพลิเคชันนี้ใช้เทคโนโลยี **Machine Learning** และ **Mediapipe** เพื่อตรวจจับท่าทางมือและแปลเป็นภาษาไทยได้ทันทีผ่านกล้องเว็บแคม 
เหมาะสำหรับการเรียนรู้ภาษามือเบื้องต้นและช่วยในการสื่อสาร
""")
st.markdown("---")

# --- 3. โหลดทรัพยากร ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

# Queue สำหรับส่งข้อความจากกล้องมาที่ UI (ใช้ non-blocking หรือ timeout สั้นๆ)
result_queue = queue.Queue()

@st.cache_resource
def load_resources():
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
        model_obj = m['model'] if isinstance(m, dict) else m
    
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None, encoding='utf-8')
        labels_list = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
    else:
        labels_list = ["Error: No Label File"]
    
    mp_hands = mp.solutions.hands
    # ปรับ min_detection_confidence และ min_tracking_confidence ให้เสถียรขึ้น
    hands_engine = mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 4. ฟังก์ชันประมวลผล ---
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_val = max(list(map(abs, temp_landmark_list)))
    return [n / max_val if max_val != 0 else 0 for n in temp_landmark_list]

def flip_keypoint_x(keypoint_list):
    flipped = list(keypoint_list)
    for i in range(0, 42, 2): 
        flipped[i] *= -1
    return flipped

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hl, mp_hands_module.HAND_CONNECTIONS)
            
            data_aux = []
            sorted_hands = sorted(
                zip(results.multi_hand_landmarks, results.multi_handedness),
                key=lambda x: x[0].landmark[0].x
            )
            
            if len(sorted_hands) == 1:
                hl, hn = sorted_hands[0]
                pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
                processed = pre_process_landmark(pts)
                if hn.classification[0].label == 'Right':
                    processed = flip_keypoint_x(processed)
                data_aux.extend(processed)
                data_aux.extend([0.0] * 42)
            elif len(sorted_hands) >= 2:
                for i in range(2):
                    hl = sorted_hands[i][0]
                    pts = [[int(l.x * w), int(l.y * h)] for l in hl.landmark]
                    data_aux.extend(pre_process_landmark(pts))
            
            if len(data_aux) == 84:
                input_data = np.array([data_aux])
                prediction = model.predict(input_data)[0]
                
                # เช็คว่าโมเดลมี predict_proba หรือไม่ (ป้องกัน Error กรณีโมเดลบางประเภทไม่มี)
                if hasattr(model, "predict_proba"):
                    conf = model.predict_proba(input_data).max()
                else:
                    conf = 1.0  # ถ้าไม่มี ให้ผ่านไปเลย
                
                if conf > 0.7:
                    res_thai = labels[int(prediction)]
                    # ใส่ค่าลง Queue (เคลียร์ของเก่าทิ้งก่อน เพื่อให้ได้ค่าล่าสุดทันที ไม่ดีเลย์)
                    if not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    result_queue.put(res_thai)
                    
    except Exception as e:
        print(f"Error in video processing: {e}")

    return frame.from_ndarray(img, format="bgr24")

# --- 5. เซิร์ฟเวอร์ช่วยเชื่อมต่อ WebRTC (STUN + TURN) ---
def get_ice_servers():
    fallback = [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
    try:
        from twilio.rest import Client
        sid = st.secrets["TWILIO_ACCOUNT_SID"]
        token = st.secrets["TWILIO_AUTH_TOKEN"]
        return Client(sid, token).tokens.create().ice_servers
    except Exception:
        return fallback

# --- 6. หน้าตาเว็บ ---
output_container = st.empty()
output_container.success("💡 ท่าทางที่พบ: กำลังรอการตรวจจับ...")

ctx = webrtc_streamer(
    key="thai-sign-online",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- ส่วนดึงข้อมูลมาแสดงผลตัวโตๆ (ปรับปรุงให้ไม่บล็อก UI นานเกินไป) ---
while ctx.state.playing:
    try:
        # ใช้ timeout สั้นๆ (เช่น 0.1 วินาที) เพื่อให้ลูปเช็คสถานะกล้องทำงานได้ลื่นไหลขึ้น
        msg = result_queue.get(timeout=0.1)
        output_container.markdown(
            f"""
            <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center;">
                <p style="margin: 0; font-size: 24px;">✅ ท่าทางที่พบ:</p>
                <h1 style="margin: 0; font-size: 100px; font-weight: bold;">{msg}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    except queue.Empty:
        pass
