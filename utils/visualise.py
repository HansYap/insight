import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yaml
from core.detector import ObjectDetector



def visualise(video_path, output_path="data/annotated_output.mp4"):
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    
    detector = ObjectDetector(config["models"]["yolo"])
    detector.load()
    
    cap = cv2.VideoCapture(video_path)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        detections = detector.detect(frame)
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    visualise(sys.argv[1])