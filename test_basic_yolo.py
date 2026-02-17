
import cv2
from ultralytics import YOLO
import os



def test():
    v_model = YOLO("yolov8n.pt")
    p_model = YOLO("models/license_plate_detector.pt")
    
    # Try a training image
    training_img_path = r"data\yolo_data\train\images\00a7b45a-6d6f4699.jpg" # Example name from previous listing
    if not os.path.exists(training_img_path):
        # Find first image in training set
        imgs = os.listdir(r"data\yolo_data\train\images")
        if imgs:
            training_img_path = os.path.join(r"data\yolo_data\train\images", imgs[0])
    
    print(f"Testing on image: {training_img_path}")
    frame = cv2.imread(training_img_path)
    if frame is None:
        print("Could not read image")
        return
    
    p_results = p_model(frame, conf=0.1, verbose=False)[0]
    print(f"Plates detected: {len(p_results.boxes)}")
    for box in p_results.boxes:
        print(f"  P PLATE Conf: {float(box.conf[0]):.2f} Bbox: {box.xyxy[0].tolist()}")

if __name__ == "__main__":
    test()
