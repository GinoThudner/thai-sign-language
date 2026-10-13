import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools

# --- 1. ตั้งค่าหน้าเว็บเพื่อ SEO ---
st.set_page_config(
    page_title="แปลภาษามือไทยออนไลน์ - AI Sign Language Translator",
    page_icon="🖐️",
    layout="centered"
)

# --- 2. ส่วนหัวข้อเว็บไซต์ ---
st.title("🖐️ ระบบแปลภาษามือไทยด้วย AI (โหมดถ่ายภาพเสถียรพิเศษ)")
st.markdown("""
### เครื่องมือช่วยแปลภาษามือไทยเป็นตัวอักษรด้วย AI
แอปพลิเคชันนี้ออกแบบมาให้ใช้งานบนคลาวด์ได้อย่างเสถียร 
เพียงกดถ่ายภาพท่าทางมือ ระบบจะทำการประมวลผลโครงสร้างแลนด์มาร์กและแปลผลทันที
""")
st.markdown("---")

# --- 3. โหลดทรัพยากร (โมเดลและป้ายกำกับ) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

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
    hands_engine = mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return model_obj, labels_list, hands_engine, mp.solutions.drawing_utils, mp_hands

model, labels, hands, mp_draw, mp_hands_module = load_resources()

# --- 4. ฟังก์ชันประมวลผลโครงสร้างมือ ---
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

# --- 5. ช่องทางรับภาพจากกล้องเว็บแคมผ่าน st.camera_input ---
st.info("💡 วิธีใช้งาน: จัดท่าทางมือให้ตรงกับกล้อง แล้วกดปุ่ม **'Take Photo'** เพื่อแปลผล")
image_file = st.camera_input("📸 ถ่ายภาพท่าทางภาษามือ")

if image_file is not None:
    # แปลงไฟล์ภาพที่ได้เป็นรูปแบบ OpenCV
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # วาดเส้นโครงสร้างมือลงบนรูปภาพ
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
            
            if hasattr(model, "predict_proba"):
                conf = model.predict_proba(input_data).max()
            else:
                conf = 1.0
            
            if conf > 0.7:
                res_thai = labels[int(prediction)]
                st.success("✅ แปลผลสำเร็จ:")
                st.markdown(
                    f"""
                    <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center;">
                        <h1 style="margin: 0; font-size: 80px; font-weight: bold;">{res_thai}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ ความมั่นใจต่ำกว่า 70% กรุณาลองจัดท่าทางให้ชัดเจนแล้วถ่ายใหม่อีกครั้ง")
        else:
            st.warning("⚠️ ข้อมูลแลนด์มาร์กมือไม่ครบถ้วน")
    else:
        st.warning("⚠️ ไม่พบมือในภาพ กรุณาวางตำแหน่งมือให้อยู่หน้ากล้องแล้วถ่ายใหม่")
    
    # แสดงรูปภาพที่ตรวจจับโครงสร้างมือแล้ว
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="ภาพตัวอย่างพร้อมโครงสร้างแลนด์มาร์ก", use_container_width=True)
