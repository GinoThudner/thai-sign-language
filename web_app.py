import gradio as gr
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import copy
import itertools

# --- 1. โหลดทรัพยากร ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'keypoint_classifier_model.pkl')
label_path = os.path.join(BASE_DIR, 'keypoint_classifier_label.csv')

with open(model_path, 'rb') as f:
    m = pickle.load(f)
    model = m['model'] if isinstance(m, dict) else m

if os.path.exists(label_path):
    df = pd.read_csv(label_path, header=None, encoding='utf-8')
    labels = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()
else:
    labels = ["Error: No Label File"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- 2. ฟังก์ชันประมวลผล ---
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

# --- 3. ฟังก์ชันประมวลผลเฟรมวิดีโอ (รับภาพจาก Gradio Webcam และส่งค่ากลับ) ---
def video_frame_callback(frame):
    if frame is None:
        return frame, """
        <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border: 1px solid #f5c6cb; text-align: center;">
            <p style="margin: 0; font-size: 24px;">💡 ท่าทางที่พบ:</p>
            <h1 style="margin: 0; font-size: 80px; font-weight: bold;">กำลังรอการเปิดกล้อง...</h1>
        </div>
        """
    
    # Gradio ส่งภาพมาเป็น RGB ต้องแปลงเป็น BGR เพื่อใช้กับ OpenCV
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    res_thai = "กำลังรอการตรวจจับ..."
    bg_color = "#d4edda"
    text_color = "#155724"
    border_color = "#c3e6cb"

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)

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
            input_array = np.array([data_aux])
            prediction = model.predict(input_array)[0]
            
            if hasattr(model, "predict_proba"):
                conf = model.predict_proba(input_array).max()
            else:
                conf = 1.0

            if conf > 0.7:
                res_thai = labels[int(prediction)]

    # สร้าง HTML แสดงผลตัวหนังสือตัวโตๆ แบบที่คุณต้องการ
    html_output = f"""
    <div style="background-color: {bg_color}; color: {text_color}; padding: 20px; border-radius: 10px; border: 1px solid {border_color}; text-align: center;">
        <p style="margin: 0; font-size: 24px;">✅ ท่าทางที่พบ:</p>
        <h1 style="margin: 0; font-size: 80px; font-weight: bold;">{res_thai}</h1>
    </div>
    """

    # แปลงภาพกลับเป็น RGB สำหรับแสดงผลใน Gradio
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), html_output

# --- 4. หน้าตาเว็บด้วย Gradio Blocks ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖐️ ระบบแปลภาษามือไทยแบบ Real-time โดย นายอชิรวิทย์ กิติสุข")
    gr.Markdown("""
    ### เครื่องมือช่วยแปลภาษามือไทยเป็นตัวอักษรด้วย AI
    แอปพลิเคชันนี้ใช้เทคโนโลยี **Machine Learning** และ **Mediapipe** เพื่อตรวจจับท่าทางมือและแปลเป็นภาษาไทยได้ทันทีผ่านกล้องเว็บแคม  
    เหมาะสำหรับการเรียนรู้ภาษามือเบื้องต้นและช่วยในการสื่อสาร
    """)
    gr.markdown("---")

    with gr.Row():
        with gr.Column(scale=2):
            # ใช้ gr.Image เปิด streaming mode สำหรับดึงภาพจากกล้องแบบเรียลไทม์
            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="กล้องเว็บแคม")
        with gr.Column(scale=1):
            output_html = gr.HTML("""
            <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; text-align: center;">
                <p style="margin: 0; font-size: 24px;">💡 ท่าทางที่พบ:</p>
                <h1 style="margin: 0; font-size: 80px; font-weight: bold;">กำลังรอการตรวจจับ...</h1>
            </div>
            """)

    # ผูกฟังก์ชันเข้ากับสตรีมของกล้อง
    webcam_input.stream(
        fn=video_frame_callback,
        inputs=webcam_input,
        outputs=[webcam_input, output_html],
        time_limit=None,
        concurrency_limit=20
    )

if __name__ == "__main__":
    demo.launch()
