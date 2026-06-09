#Your automated script to load pretrained weights and execute deep learning training

import os
from ultralytics import YOLO

def train_model():
    HOME = os.getcwd()
    
    # Initialize lightweight YOLOv8n (Nano) architecture
    model = YOLO(f"{HOME}/weights/yolov8n.pt")
    
    # Execute training workflow across custom dataset split
    model.train(
        task='detect',
        mode='train',
        epochs=90,
        batch=32,
        plots=True,
        patience=20,
        verbose=True,
        data=f"{HOME}/datasets/stray-dog-detection/data.yaml"
    )

if __name__ == "__main__":
    train_model()
