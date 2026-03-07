import cv2
import yaml
import time
from pathlib import Path
from loguru import logger

from core.detector import ObjectDetector
from core.database import EventDatabase
from utils.fps_counter import FPSCounter
from utils.memory_monitor import MemoryMonitor


def load_config(path="config/settings.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run(source=None):
    config = load_config()
    cam_cfg = config["camera"]
    
    # source=None → use config, source="path/to/video" → override for testing
    video_source = source if source is not None else cam_cfg["source"]
    
    # Initialize components
    detector = ObjectDetector(config["models"]["yolo"])
    detector.load()
    
    db = EventDatabase(config["database"]["path"])
    fps_counter = FPSCounter(window=30)
    mem_monitor = MemoryMonitor()
    
    # Open capture
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, cam_cfg["buffer_size"])  # prevent lag
    
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {video_source}")
        return
    
    logger.info(f"Pipeline started | Source: {video_source}")
    logger.info(f"Initial RAM: {mem_monitor.report()}")
    
    trigger_classes = set(config["detection"]["trigger_classes"])
    last_stats_log = time.time()
    stats_interval = config["database"]["log_fps_interval"]
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed — end of stream or disconnected camera")
                break
            
            detections = detector.detect(frame)
            fps_counter.tick()
            
            # Log interesting detections to DB
            for det in detections:
                if det["class_name"] in trigger_classes:
                    db.log_detection(
                        scene=config["scene"]["active"],
                        detection=det
                    )
            
            # Periodic stats logging
            now = time.time()
            if now - last_stats_log >= stats_interval:
                fps = fps_counter.fps
                ram_mb = mem_monitor.process_mb()
                ram_pct = mem_monitor.system_percent()
                db.log_stats(fps, ram_mb, ram_pct)
                logger.info(f"FPS: {fps:.1f} | {mem_monitor.report()}")
                last_stats_log = now

    except KeyboardInterrupt:
        logger.info("Shutting down pipeline...")
    finally:
        cap.release()
        logger.info("Pipeline stopped cleanly.")


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run(source=src)