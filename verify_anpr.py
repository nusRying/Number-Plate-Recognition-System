
import cv2
import os
import logging
from src.detector import ANPRDetector
from src.ocr import PlateOCR
from src.utils import draw_info, save_crop, perspective_correction, get_plate_corners

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyANPR")

def verify_on_video(video_path, num_frames=10):
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    detector = ANPRDetector()
    ocr = PlateOCR(gpu=False) # Use CPU for verification to avoid GPU issues

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detections_found = 0

    print(f"--- Starting Verification on {video_path} ---")

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        logger.info(f"Processing frame {frame_count}...")

        # Detect Vehicles
        vehicles = detector.detect_vehicles(frame)
        logger.info(f"  Found {len(vehicles)} vehicles")

        for v in vehicles:
            x1, y1, x2, y2 = v['bbox']
            vehicle_crop = frame[y1:y2, x1:x2]

            if vehicle_crop.size > 0:
                # Detect Plates
                plates = detector.detect_plates(vehicle_crop)
                logger.info(f"    Found {len(plates)} plates in vehicle {v['label']}")

                for p in plates:
                    detections_found += 1
                    px1, py1, px2, py2 = p['bbox']
                    plate_crop = vehicle_crop[py1:py2, px1:px2]

                    if plate_crop.size > 0:
                        # Perspective Correction
                        corners = get_plate_corners(plate_crop)
                        corrected_plate = perspective_correction(plate_crop, corners)
                        
                        # OCR
                        plate_text, ocr_conf = ocr.extract_text(corrected_plate)
                        logger.info(f"      OCR Result: {plate_text} (Conf: {ocr_conf:.2f})")

                        # Save results for manual check
                        abs_px1, abs_py1 = x1 + px1, y1 + py1
                        abs_px2, abs_py2 = x1 + px2, y1 + py2
                        
                        annotated_frame = draw_info(frame.copy(), [abs_px1, abs_py1, abs_px2, abs_py2], plate_text, ocr_conf)
                        
                        output_dir = "debug_results"
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            
                        cv2.imwrite(f"{output_dir}/frame_{frame_count}_plate_{detections_found}.jpg", annotated_frame)
                        cv2.imwrite(f"{output_dir}/crop_{frame_count}_plate_{detections_found}.jpg", plate_crop)
                        cv2.imwrite(f"{output_dir}/corrected_{frame_count}_plate_{detections_found}.jpg", corrected_plate)

    cap.release()
    print(f"--- Verification Finished. Total plates found: {detections_found} ---")
    print(f"Check the 'debug_results' folder for output images.")

if __name__ == "__main__":
    verify_on_video("data/sample_video.mp4", num_frames=50)
