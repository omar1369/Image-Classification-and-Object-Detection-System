import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, List

class YOLOv8FireDetector:
    def __init__(self, model_path: str, conf_thres=0.3, iou_thres=0.4):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]  # (height, width)
        self.class_names = ['fire', 'smoke']  # Update with your actual class names

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """Main prediction method with proper aspect ratio handling"""
        # Store original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess with padding info
        input_tensor, scale_pad = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # Postprocess with correct scaling
        boxes, scores, class_ids = self._postprocess(outputs, (orig_h, orig_w), scale_pad)
        
        # Format results
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append((
                self.class_names[class_id],
                float(score),
                box.astype(int).tolist()
            ))
        
        return image, detections

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, Tuple[int, int]]]:
        """Prepares image for inference with proper scaling and normalization"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate padding to maintain aspect ratio
        h, w = image.shape[:2]
        scale = min(self.input_shape[0] / h, self.input_shape[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize with aspect ratio preservation
        img = cv2.resize(img, (new_w, new_h))
        
        # Add padding to make it square
        top = (self.input_shape[0] - new_h) // 2
        bottom = self.input_shape[0] - new_h - top
        left = (self.input_shape[1] - new_w) // 2
        right = self.input_shape[1] - new_w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Normalize and transpose
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        return np.expand_dims(img, axis=0), (scale, (left, top))

    def _postprocess(self, outputs: List[np.ndarray], image_shape: Tuple[int, int], 
                   scale_pad: Tuple[float, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes model outputs with correct box scaling"""
        scale, (pad_x, pad_y) = scale_pad
        predictions = np.squeeze(outputs[0]).T
        
        # Filter by confidence
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        
        if len(scores) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
        
        # Convert from center_x, center_y, width, height to x1, y1, x2, y2
        boxes = predictions[:, :4].copy()
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2)  # x1
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2)  # y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]        # x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]        # y2
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                 self.conf_threshold, self.iou_threshold)
        
        # Process detections
        boxes_out = []
        scores_out = []
        class_ids_out = []
        
        for i in indices:
            idx = i if isinstance(indices, list) else i.item()
            class_id = np.argmax(predictions[idx, 4:])
            
            # Rescale boxes to original image coordinates
            box = boxes[idx]
            box = [
                (box[0] - pad_x) / scale,  # x1
                (box[1] - pad_y) / scale,  # y1
                (box[2] - pad_x) / scale,  # x2
                (box[3] - pad_y) / scale   # y2
            ]
            
            # Clip to image bounds
            h, w = image_shape
            box[0] = max(0, min(box[0], w))
            box[1] = max(0, min(box[1], h))
            box[2] = max(0, min(box[2], w))
            box[3] = max(0, min(box[3], h))
            
            boxes_out.append(box)
            scores_out.append(scores[idx])
            class_ids_out.append(class_id)
            
        return np.array(boxes_out), np.array(scores_out), np.array(class_ids_out)