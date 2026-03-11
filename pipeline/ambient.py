import cv2
import yaml
import time
import threading
from pathlib import Path
from loguru import logger

from core.detector import ObjectDetector
from core.database import EventDatabase
from utils.fps_counter import FPSCounter
from utils.memory_monitor import MemoryMonitor
from core.room_state import RoomStateTracker
from core.frame_client import send_frame_for_description


def load_config(path="config/settings.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def on_event_triggered(event: dict, frame):
    logger.info(f"[FLORENCE] Querying for event: {event['type']}")
    ROOT = Path(__file__).parent.parent
    save_dir = ROOT / "debug_frames"
    save_dir.mkdir(exist_ok=True)

    timestamp = int(event["time stamp"])
    event_type = event["type"]
    save_path = save_dir / f"{event_type}_{timestamp}.jpg"

    cv2.imwrite(save_path, frame)

    result = send_frame_for_description(frame)
    
    if "error" in result:
        logger.warning(f"[FLORENCE] Failed: {result['error']}")
        return
    
    description = result.get("description", "")
    logger.info(f"[FLORENCE] {description}")


def run(source=None, loop=False):
    
    config = load_config()
    cam_cfg = config["camera"]
    state_tracker = RoomStateTracker(empty_grace_seconds=3.0)

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

    PROCESS_EVERY_N_FRAMES = 3
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning("Frame read failed — end of stream or disconnected camera")
                    break

            frame_count += 1
            fps_counter.tick()

            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue

            small_frame = cv2.resize(frame, (320, 240))
            detections = detector.detect(small_frame)
            room_state, event = state_tracker.update(detections)

            if event:
                db.log_room_event(event)
                threading.Thread(target=on_event_triggered, args=(event, frame.copy()), daemon=True).start()
            
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    run(source=args.source, loop=args.loop)