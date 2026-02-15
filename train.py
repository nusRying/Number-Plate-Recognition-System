from ultralytics import YOLO
import torch

def train():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model with optimized memory settings for Windows
    results = model.train(
        data=r"C:\Users\umair\Videos\Number Plate Recognition System\data\data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,           # Reduced from 16 to 4 to save memory
        workers=0,         # Prevents multiple processes from locking memory
        device=0 if torch.cuda.is_available() else 'cpu',
        name="license_plate_yolov8n"
    )

    # Evaluate performance
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

    # Export the model
    success = model.export(format="onnx")
    print("Model exported successfully.")

if __name__ == "__main__":
    train()
