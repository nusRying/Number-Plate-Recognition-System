import cv2
import logging
import time
import queue
import threading
from collections import Counter
from src.detector import ANPRDetector
from src.ocr import PlateOCR
from src.database import DatabaseManager
from src.notifier import TelegramNotifier
from src.utils import draw_info, save_crop, apply_clahe, perspective_correction, get_plate_corners
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ANPR_System")

class ANPRSystem:
    def __init__(self, use_gpu=False, telegram_token=None, telegram_chat_id=None):
        logger.info("Initializing Advanced ANPR System...")
        self.detector = ANPRDetector()
        self.ocr = PlateOCR(gpu=use_gpu)
        self.db = DatabaseManager()
        self.notifier = TelegramNotifier(token=telegram_token, chat_id=telegram_chat_id)
        self.tracker = DeepSort(max_age=30, n_init=3)
        
        # Advanced State Management
        self.processed_track_ids = set()
        self.voting_buffer = {} # {track_id: [list of ocr results]}
        self.vote_threshold = 5 # Minimum frames to vote
        self.blacklist = ["ABC-123", "XYZ-786"] # Dummy blacklist

        # Async Processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.stopped = False

    def process_frame_async(self):
        """
        Background thread for frame processing.
        """
        while not self.stopped:
            try:
                frame, night_mode = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            processed_frame = self.process_frame(frame, night_mode)
            self.output_queue.put(processed_frame)

    def process_frame(self, frame, night_mode=False):
        if night_mode:
            frame = apply_clahe(frame)

        # 1. Detect Vehicles
        vehicles = self.detector.detect_vehicles(frame)
        
        detections_for_tracker = []
        for v in vehicles:
            x1, y1, x2, y2 = v['bbox']
            w, h = x2 - x1, y2 - y1
            detections_for_tracker.append([[x1, y1, w, h], v['score'], v['class_id']])

        # 2. Update Tracks
        tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Draw vehicle tracking info
            frame = draw_info(frame, [x1, y1, x2, y2], f"ID:{track_id}", 1.0, color=(255, 0, 0))

            # 3. Plate Analysis
            h_f, w_f = frame.shape[:2]
            y1_p, y2_p = max(0, y1), min(h_f, y2)
            x1_p, x2_p = max(0, x1), min(w_f, x2)
            vehicle_crop = frame[y1_p:y2_p, x1_p:x2_p]

            if vehicle_crop.size > 0:
                plates = self.detector.detect_plates(vehicle_crop)
                for p in plates:
                    px1, py1, px2, py2 = p['bbox']
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    
                    if plate_crop.size > 0:
                        # --- Advanced Step: Perspective Correction ---
                        corners = get_plate_corners(plate_crop)
                        corrected_plate = perspective_correction(plate_crop, corners)
                        
                        plate_text, ocr_conf = self.ocr.extract_text(corrected_plate)
                        
                        if plate_text and self.ocr.is_valid(plate_text):
                            # --- Advanced Step: Multi-frame Voting ---
                            if track_id not in self.voting_buffer:
                                self.voting_buffer[track_id] = []
                            self.voting_buffer[track_id].append(plate_text)

                            # Determine final plate via majority vote after threshold
                            if len(self.voting_buffer[track_id]) >= self.vote_threshold:
                                most_common_plate, count = Counter(self.voting_buffer[track_id]).most_common(1)[0]
                                
                                # Visualize and Log once
                                abs_px1, abs_py1 = x1_p + px1, y1_p + py1
                                abs_px2, abs_py2 = x1_p + px2, y1_p + py2
                                frame = draw_info(frame, [abs_px1, abs_py1, abs_px2, abs_py2], most_common_plate, ocr_conf, color=(0, 255, 255))

                                if track_id not in self.processed_track_ids:
                                    crop_path = save_crop(corrected_plate)
                                    self.db.log_detection(most_common_plate, ocr_conf, vehicle_id=track_id, image_path=crop_path)
                                    self.processed_track_ids.add(track_id)
                                    
                                    # --- Advanced Step: Alerting ---
                                    if most_common_plate in self.blacklist:
                                        self.notifier.send_alert(most_common_plate, track_id)
                                    
                                    logger.info(f"Finalized Plate (Voted): {most_common_plate} (ID: {track_id})")

        return frame

def main(source=0):
    system = ANPRSystem(use_gpu=False)
    
    # Start background processing thread
    thread = threading.Thread(target=system.process_frame_async, daemon=True)
    thread.start()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Could not open video source.")
        return

    logger.info("Starting processing loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Push to processing queue
        if not system.frame_queue.full():
            system.frame_queue.put((frame, False)) # False = night_mode off

        # Get processed frame if available
        try:
            output_frame = system.output_queue.get_nowait()
            cv2.imshow("Advanced Number Plate Recognition", output_frame)
        except queue.Empty:
            # If processing is slower than capture, we show the raw frame or wait
            cv2.imshow("Advanced Number Plate Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            system.stopped = True
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
