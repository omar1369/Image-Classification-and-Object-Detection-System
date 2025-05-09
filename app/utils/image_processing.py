import cv2
import numpy as np
from PIL import Image
import io

def bytes_to_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded file bytes to OpenCV image"""
    image = Image.open(io.BytesIO(file_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def image_to_bytes(image: np.ndarray) -> bytes:
    """Convert OpenCV image to bytes"""
    _, buffer = cv2.imencode('.jpg', image)
    return io.BytesIO(buffer).getvalue()

def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes with class labels"""
    for label, confidence, box in detections:
        x1, y1, x2, y2 = map(int, box)
        
        # Choose color based on label
        color = (0, 0, 255) if label == 'fire' else (0, 255, 255)  # Red for fire, yellow for smoke
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label_text = f"{label}: {confidence:.2f}"
        
        # Calculate text size
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Draw label background
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return image
