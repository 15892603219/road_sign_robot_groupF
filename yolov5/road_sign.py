import torch
import cv2

# 加载预训练的YOLOv5s模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', force_reload=True)  # 使用预训练的YOLOv5s模型

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 是默认摄像头设备

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 进行推理
    results = model(frame)

    # 使用 render() 绘制检测框
    results.render()  # 在图像上绘制检测框

    # 获取渲染后的图像
    rendered_img = results.ims[0]

    # 显示带有检测框的图像
    cv2.imshow('Road Sign Detection', rendered_img)  # 使用 OpenCV 显示图像

    # 打印推理结果（检测的类别和置信度）
    for *box, conf, cls in results.xywh[0]:
        print(f"Detected: {results.names[int(cls)]}, Confidence: {conf:.2f}")

    # 如果按了 'q' 键，退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()

