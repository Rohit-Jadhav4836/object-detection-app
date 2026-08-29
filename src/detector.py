import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

@st.cache_resource(show_spinner=False)
def load_model(model_size: str = "yolov8n.pt"):
    """Load once and cache — ultralytics auto-downloads weights on first run."""
    return YOLO(model_size)


def run_detection(model: YOLO, image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45, color_by_confidence: bool = False):
    import time
    start_time = time.time()

    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    result = results[0]

    inference_time = time.time() - start_time

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class": model.names[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),
        })

    annotated = annotate_hud_style(image, detections, color_by_confidence)
    return annotated, detections, inference_time


CLASS_COLORS = [
    (51, 255, 102), (255, 51, 153), (51, 204, 255), (255, 204, 51),
    (204, 51, 255), (255, 102, 51), (102, 255, 204), (255, 255, 102),
    (153, 255, 51), (255, 153, 51), (51, 153, 255), (255, 51, 204),
]

_class_color_cache = {}

def get_class_color(class_name: str):
    """Get consistent color for a class name."""
    if class_name not in _class_color_cache:
        color_idx = len(_class_color_cache) % len(CLASS_COLORS)
        _class_color_cache[class_name] = CLASS_COLORS[color_idx]
    return _class_color_cache[class_name]


def get_confidence_color(confidence: float):
    """
    Get color based on confidence level.
    Red = low (<60%), Yellow = medium (60-80%), Green = high (>80%)
    """
    if confidence < 0.6:
        return (0, 0, 255)  # Red (BGR)
    elif confidence < 0.8:
        return (0, 255, 255)  # Yellow (BGR)
    else:
        return (0, 255, 0)  # Green (BGR)


def annotate_hud_style(image, detections, color_by_confidence=False):
    """
    Draw bounding boxes with labels.
    Args:
        image: Input image
        detections: List of detection dicts
        color_by_confidence: If True, color boxes by confidence level instead of class
    """
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        # Choose color scheme
        if color_by_confidence:
            color = get_confidence_color(det["confidence"])
        else:
            color = get_class_color(det["class"])

        # Draw bounding box with thickness based on confidence
        thickness = 3 if det["confidence"] > 0.7 else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # Create label with confidence indicator
        conf_pct = det["confidence"] * 100
        if conf_pct < 60:
            conf_indicator = "⚠"
        elif conf_pct < 80:
            conf_indicator = "●"
        else:
            conf_indicator = "✓"

        label = f'{conf_indicator} {det["class"].upper()} {conf_pct:.0f}%'

        # Draw label background and text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(output, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return output