import torch
import cv2
import pyttsx3
import speech_recognition as sr
import threading
import time

# 初始化语音播报引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 加载 YOLOv5 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device)

# 路标说明字典
road_sign_info = {
    "stop sign": "This is a stop sign, please stop and check if the surroundings are safe.",
    "speed limit": "This is a speed limit sign, recommended speed is 50 kilometers per hour."
}

# 摄像头设置
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 异步语音识别线程
user_command = None
def listen_and_recognize():
    global user_command
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            try:
                print("Listening for commands...")
                audio = recognizer.listen(source, timeout=5)
                user_command = recognizer.recognize_google(audio, language="zh-CN")
                print(f"Recognized command: {user_command}")
            except Exception as e:
                print(f"Speech recognition error: {e}")

threading.Thread(target=listen_and_recognize, daemon=True).start()

# 主循环
frame_count = 0
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 限制帧率为30FPS
    if time.time() - start_time >= 0.033:
        start_time = time.time()

        # YOLO 推理（每5帧进行一次）
        if frame_count % 5 == 0:
            results = model(frame)
            results.render()
            detected_signs = set()
            for *box, conf, cls in results.xywh[0]:
                detected_sign = results.names[int(cls)]
                detected_signs.add(detected_sign)
                if detected_sign in road_sign_info:
                    print(f"Detected road sign: {detected_sign}")
                    engine.say(road_sign_info[detected_sign])
                    engine.runAndWait()

        # 显示视频帧
        if 'results' in locals():  # 确保 results 已定义
            cv2.imshow('Road Sign Detection', results.ims[0])
        else:
            cv2.imshow('Road Sign Detection', frame)

        # 检查用户语音命令
        if user_command:
            if "前方是什么" in user_command:
                if detected_signs:
                    engine.say(f"The road signs detected are: {', '.join(detected_signs)}")
                else:
                    engine.say("No road signs detected")
                engine.runAndWait()
                user_command = None  # 重置命令

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

