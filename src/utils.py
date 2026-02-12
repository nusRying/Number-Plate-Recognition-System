import cv2
import numpy as np
import os
from datetime import datetime

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization for night vision/low light.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def draw_info(frame: np.ndarray, bbox: list, text: str, score: float, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Utility to draw bounding box and text on frame.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label = f"{text} ({score:.2f})"
    (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_w, y1), color, -1)
    
    cv2.putText(frame, label, (x1, y1 - 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame

def get_plate_corners(plate_img: np.ndarray) -> np.ndarray:
    """
    Attempt to find 4 corners of the plate using contour detection.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
            
    # Fallback to bbox corners if no 4-point contour found
    h, w = plate_img.shape[:2]
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")

def perspective_correction(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective correction to a plate image given 4 corners.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def save_crop(img: np.ndarray, folder: str = "data/crops") -> str:
    """
    Save cropped image to folder and return path.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, img)
    return path
