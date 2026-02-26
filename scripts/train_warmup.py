from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Load pre-trained YOLOv8 nano model
    model = YOLO('yolov8n.pt')

    # Train on your dataset
    results = model.train(
        data=os.path.join(BASE_DIR, 'data/dataset.yaml'),
        project=os.path.join(BASE_DIR, 'models'),
        epochs=50,
        imgsz=640,
        batch=16,
        name='warmup_finetune',,
        device=0  # GPU
    )

    # Export to ONNX
    model.export(format='onnx')