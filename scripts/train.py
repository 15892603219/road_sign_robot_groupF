import os
import sys
from yolov5.train import run

if __name__ == "__main__":
    # 设置训练参数
    run(
        sys.path.append('/home/tian/Traffic_Sign_Recognition/yolov5'),  # 替换为你的 yolov5 目录路径
        data='data/dataset.yaml',          # 数据集配置文件路径
        weights='yolov5n.pt',             # 使用 YOLOv5 Nano 预训练权重
        imgsz=640,                        # 输入图片大小
        batch_size=16,                    # 批次大小
        epochs=50,                        # 训练轮次
        project='outputs',                # 输出文件夹
        name='Traffic_Sign_Detection',    # 项目名称
        device='cpu'                         # 使用 GPU (若无 GPU，则设置为 'cpu')
    )

