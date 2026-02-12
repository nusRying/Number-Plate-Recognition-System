import cv2
import easyocr
import re
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class PlateOCR:
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        """
        Initialize the OCR engine.
        :param languages: List of languages for EasyOCR.
        :param gpu: Whether to use GPU acceleration.
        """
        try:
            self.reader = easyocr.Reader(languages, gpu=gpu)
            logger.info(f"EasyOCR initialized with languages: {languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

        # Generic pattern for plates, can be customized via config
        self.plate_pattern = re.compile(r'^[A-Z0-9-]{4,10}$')

    def preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Multi-stage preprocessing to handle various lighting conditions.
        """
        if plate_img is None or plate_img.size == 0:
            return None

        # 1. Resize for better OCR visibility if too small
        height, width = plate_img.shape[:2]
        if width < 200:
            plate_img = cv2.resize(plate_img, (None, None), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 2. Grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # 3. Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 4. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 5. Morphological operations (Optional dilation to thicken characters)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cv2.bitwise_not(morphed) # Flip back to black text on white background

    def extract_text(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from plate image and return (text, confidence).
        """
        processed_img = self.preprocess(plate_img)
        if processed_img is None:
            return "", 0.0

        results = self.reader.readtext(processed_img)
        
        if not results:
            return "", 0.0
            
        # Filter results: sometimes OCR breaks one plate into multiple chunks
        # We'll join them and take the average confidence
        full_text = "".join([res[1] for res in results]).upper().strip()
        # Clean non-alphanumeric noise except hyphen
        clean_text = re.sub(r'[^A-Z0-9-]', '', full_text)
        
        avg_confidence = sum([res[2] for res in results]) / len(results)
            
        return clean_text, float(avg_confidence)

    def is_valid(self, text: str) -> bool:
        """
        Validate plate text using Regex.
        """
        return bool(self.plate_pattern.match(text))

if __name__ == "__main__":
    ocr = PlateOCR(gpu=False)
    print("OCR refinement complete.")
