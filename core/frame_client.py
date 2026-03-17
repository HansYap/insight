import requests
import cv2
import time
from datetime import datetime

THINKPAD_URL = "http://192.168.68.109:8000"

def send_frame_for_description(frame_bgr, prompt="<MORE_DETAILED_CAPTION>", context: dict | None = None):
    _, jpeg_bytes = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    data = {"prompt": prompt}
    if context:
        data["scene"] = context.get("scene", "")
        data["hour"] = str(context.get("hour", datetime.now().hour))

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


def queue_pending(frame, description: str, event_type: str, score: float,
                  context: dict | None = None):
    _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    data = {
        "description": description,
        "event_type": event_type,
        "score": score,
        "scene": context.get("scene", "") if context else "",
        "hour": str(context.get("hour", datetime.now().hour)) if context else str(datetime.now().hour),
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