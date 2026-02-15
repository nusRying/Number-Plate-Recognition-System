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
        Enhanced multi-stage preprocessing with CLAHE, bilateral filtering, and sharpening.
        """
        if plate_img is None or plate_img.size == 0:
            return None

        # 1. Resize for better OCR visibility if too small
        height, width = plate_img.shape[:2]
        if width < 200:
            plate_img = cv2.resize(plate_img, (None, None), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 2. Grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 4. Bilateral Filter (preserves edges while reducing noise)
        bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 5. Unsharp Masking (sharpening)
        gaussian = cv2.GaussianBlur(bilateral, (0, 0), 3.0)
        sharpened = cv2.addWeighted(bilateral, 1.5, gaussian, -0.5, 0)
        
        # 6. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            sharpened, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 7. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return morphed

    def preprocess_multi_strategy(self, plate_img: np.ndarray) -> list:
        """
        Apply multiple preprocessing strategies and return all variants.
        Use this for multi-strategy OCR where we try different approaches.
        """
        if plate_img is None or plate_img.size == 0:
            return []

        strategies = []
        height, width = plate_img.shape[:2]
        if width < 200:
            plate_img = cv2.resize(plate_img, (None, None), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Strategy 1: CLAHE + Bilateral + Adaptive Threshold
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        strategies.append(thresh1)

        # Strategy 2: Otsu's Thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strategies.append(otsu)

        # Strategy 3: Simple Denoising + Adaptive Threshold
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        thresh2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        strategies.append(cv2.bitwise_not(thresh2))

        return strategies

    def extract_text(self, plate_img: np.ndarray, use_multi_strategy: bool = True) -> Tuple[str, float]:
        """
        Extract text from plate image and return (text, confidence).
        If use_multi_strategy=True, tries multiple preprocessing approaches and picks the best.
        """
        if use_multi_strategy:
            # Try multiple preprocessing strategies
            strategies = self.preprocess_multi_strategy(plate_img)
            best_text = ""
            best_confidence = 0.0

            for processed_img in strategies:
                if processed_img is None:
                    continue

                results = self.reader.readtext(processed_img)
                
                if not results:
                    continue
                    
                full_text = "".join([res[1] for res in results]).upper().strip()
                clean_text = re.sub(r'[^A-Z0-9-]', '', full_text)
                avg_confidence = sum([res[2] for res in results]) / len(results)

                # Keep the result with highest confidence that passes validation
                if avg_confidence > best_confidence and self.is_valid(clean_text):
                    best_text = clean_text
                    best_confidence = avg_confidence

            # If no valid result found, return the highest confidence anyway
            if not best_text:
                for processed_img in strategies:
                    if processed_img is None:
                        continue
                    results = self.reader.readtext(processed_img)
                    if not results:
                        continue
                    full_text = "".join([res[1] for res in results]).upper().strip()
                    clean_text = re.sub(r'[^A-Z0-9-]', '', full_text)
                    avg_confidence = sum([res[2] for res in results]) / len(results)
                    if avg_confidence > best_confidence:
                        best_text = clean_text
                        best_confidence = avg_confidence

            return best_text, float(best_confidence)
        else:
            # Single strategy mode
            processed_img = self.preprocess(plate_img)
            if processed_img is None:
                return "", 0.0

            results = self.reader.readtext(processed_img)
            
            if not results:
                return "", 0.0
                
            full_text = "".join([res[1] for res in results]).upper().strip()
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
