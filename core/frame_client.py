import requests
import cv2
import time
from datetime import datetime
import base64
from loguru import logger

THINKPAD_URL = "http://192.168.68.109:8000"

def send_frame_for_description(frame_bgr, prompt="<MORE_DETAILED_CAPTION>"):
    _, jpeg_bytes = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    data = {"prompt": prompt}

    try:
        response = requests.post(
            f"{THINKPAD_URL}/describe",
            files={"frame": ("frame.jpg", jpeg_bytes.tobytes(), "image/jpeg")},
            data=data,
            timeout=30
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "ThinkPad server not reachable"}
    except requests.exceptions.Timeout:
        return {"error": "Florence-2 inference timed out"}


def queue_pending(frame, description: str, event_type: str, score: float):
    _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    data = {
        "description": description,
        "event_type": event_type,
        "score": score,
    }
    try:
        response = requests.post(
            f"{THINKPAD_URL}/queue-pending",
            files={"frame": ("frame.jpg", jpeg_bytes.tobytes(), "image/jpeg")},
            data=data,
            timeout=15
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def save_frame_remote(frame, rel_path: str):
    _, buf = cv2.imencode(".jpg", frame)
    img_b64 = base64.b64encode(buf).decode()
    try:
        requests.post(
            f"{THINKPAD_URL}/save_frame",
            json={"path": rel_path, "image": img_b64},
            timeout=15
        )
    except Exception as e:
        logger.warning(f"[REMOTE SAVE] Failed: {e}")