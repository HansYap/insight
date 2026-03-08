from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import yaml

class ObjectDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = Path(config["path"])
        self.conf = config["confidence_threshold"]
        self.iou = config["iou_threshold"]
        self.device = config["device"]
        self.imgsz = config["imgsz"]
        self.model = None

    def load(self):
        if not self.model_path.exists():
            logger.info(f"Model not found at {self.model_path}, downloading...")
            self.model = YOLO("yolov8n.pt")  # auto-downloads
            exported_folder = "yolov8n_ncnn_model"

            self.model.export(format="openvino", half=True) 
            Path(exported_folder).rename(self.model_path)
            #self.model_path.parent.mkdir(parents=True, exist_ok=True)
        #     self.model.save(str(self.model_path))
        # else:
        #     self.model = YOLO(str(self.model_path))
        # logger.info(f"YOLOv8n loaded from {self.model_path}")
        self.model = YOLO(str(self.model_path), task="detect")
        logger.info(f"YOLOv8n NCNN loaded from {self.model_path}")

    def detect(self, frame):
        """
        Returns list of dicts: [{class_name, confidence, bbox_xyxy}, ...]
        """
        if self.model is None:
            raise RuntimeError("Detector not loaded. Call .load() first.")
        
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            #device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_name": r.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })
        return detections