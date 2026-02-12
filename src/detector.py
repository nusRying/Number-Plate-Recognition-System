import cv2
import logging
from typing import List, Dict, Any
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ANPRDetector:
    def __init__(self, vehicle_model: str = "yolov8n.pt", plate_model: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """
        Initialize the ANPR Detector with YOLOv8 models.
        :param vehicle_model: Path/alias for vehicle detection model.
        :param plate_model: Path/alias for license plate detection model.
        :param conf_threshold: Minimum confidence score for detections.
        """
        try:
            self.vehicle_model = YOLO(vehicle_model)
            self.plate_model = YOLO(plate_model)
            logger.info(f"Models loaded: Vehicle({vehicle_model}), Plate({plate_model})")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

        self.conf_threshold = conf_threshold
        # COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]

    def detect_vehicles(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the frame and return filtered results.
        """
        results = self.vehicle_model(frame, classes=self.vehicle_classes, conf=self.conf_threshold, verbose=False)[0]
        vehicles = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            vehicles.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class_id": int(class_id),
                "label": self.vehicle_model.names[int(class_id)]
            })
        return vehicles

    def detect_plates(self, vehicle_crop: Any) -> List[Dict[str, Any]]:
        """
        Detect license plates within a cropped vehicle image.
        """
        # We lower the confidence slightly for plates as they are often smaller
        results = self.plate_model(vehicle_crop, conf=self.conf_threshold * 0.8, verbose=False)[0]
        plates = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            plates.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score)
            })
        return plates

if __name__ == "__main__":
    detector = ANPRDetector()
    print("Detector refinement complete.")
