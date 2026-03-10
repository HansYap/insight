# pi/frame_client.py
import requests
import cv2
import time

THINKPAD_URL = "http://192.168.68.109:8000/describe"
# Find ThinkPad IP: ip addr show on ThinkPad, look for 192.168.x.x

def send_frame_for_description(frame_bgr, prompt="<MORE_DETAILED_CAPTION>"):
    """
    Takes a CV2 frame (BGR numpy array), sends to ThinkPad, returns description.
    Used when YOLO triggers an interesting event.
    """
    # Encode to JPEG — smaller than raw, fast enough
    _, jpeg_bytes = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    try:
        response = requests.post(
            THINKPAD_URL,
            files={"frame": ("frame.jpg", jpeg_bytes.tobytes(), "image/jpeg")},
            data={"prompt": prompt},
            timeout=30  # Florence-2 on CPU can take 3-5s
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "ThinkPad server not reachable"}
    except requests.exceptions.Timeout:
        return {"error": "Florence-2 inference timed out"}