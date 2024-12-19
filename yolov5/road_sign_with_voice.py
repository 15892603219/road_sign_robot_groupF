import torch
import cv2
import pyttsx3
import queue
import threading

# 初始化语音播报引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 设置语速

# 初始化队列用于处理检测结果
result_queue = queue.Queue()

# 加载预训练的 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', force_reload=True)

# 摄像头视频捕获
cap = cv2.VideoCapture(0)  # 0 是默认摄像头设备

def voice_alert():
    """语音播报线程函数"""
    while True:
        result = result_queue.get()  # 从队列中获取检测结果
        if result == "STOP":  # 停止信号
            break
        engine.say(result)
        engine.runAndWait()

# 启动语音播报线程
voice_thread = threading.Thread(target=voice_alert, daemon=True)
voice_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 推理
    results = model(frame)
    results.render()  # 绘制检测框到图像

    # 显示检测结果
    cv2.imshow('Road Sign Detection', results.ims[0])

    # 解析检测结果
    detected_objects = []
    for *box, conf, cls in results.xywh[0]:
        detected_objects.append(f"{results.names[int(cls)]} with confidence {conf:.2f}")

    # 如果有检测到的目标，将结果放入队列进行语音播报
    if detected_objects:
        for obj in detected_objects:
            result_queue.put(obj)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        result_queue.put("STOP")  # 向队列发送停止信号
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

