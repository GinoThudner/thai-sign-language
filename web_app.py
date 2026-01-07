import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd

# --- 1. การจัดการ Path และโหลดโมเดลแบบปลอดภัย ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

@st.cache_resource
def load_my_model():
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            # รองรับทั้งแบบเก็บเป็น dict {'model': ...} หรือเก็บตัว model ตรงๆ
            model = data['model'] if isinstance(data, dict) else data
        
        labels = pd.read_csv(label_path, header=None).iloc[:, 0].tolist()
        return model, labels
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None, None

model, labels = load_my_model()

# --- 2. ตั้งค่าหน้าเว็บ Streamlit ---
st.set_page_config(page_title="Thai Sign Language Translator", layout="wide")
st.title("🖐️ เครื่องแปลภาษามือไทยแบบ Real-time (On Web)")
st.write(f"สถานะโมเดล: {'✅ พร้อมใช้งาน' if model else '❌ ไม่พบไฟล์โมเดล'}")

# --- 3. ส่วนประมวลผลวิดีโอ ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # กลับด้านภาพให้เหมือนกระจก
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks and model:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- การเตรียมข้อมูลพิกัด (Logic เดิมของคุณ) ---
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # --- การทำนายผล ---
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels[int(prediction[0])]
                    
                    # วาดผลลัพธ์ลงบนหน้าจอ
                    cv2.putText(img, f"Translate: {predicted_character}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                except:
                    pass

                # วาดเส้นจุดเชื่อมมือ
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

# --- 4. ปุ่มเปิดกล้องบนเว็บ ---
webrtc_streamer(
    key="sign-language-translator", 
    video_processor_factory=VideoProcessor,
    rtc_configuration={ # ส่วนนี้ช่วยให้รันบนเครือข่ายอินเทอร์เน็ตได้เสถียรขึ้น
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
