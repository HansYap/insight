import cv2
import yaml
import time
import threading
from pathlib import Path
from datetime import datetime
from loguru import logger

from core.detector import ObjectDetector
from core.sql_database import EventDatabase
from utils.fps_counter import FPSCounter
from utils.memory_monitor import MemoryMonitor
from core.room_state import RoomStateTracker, RoomState
from core.frame_client import send_frame_for_description, queue_pending

ROOT = Path(__file__).parent.parent
# Motion delta config
MOTION_THRESHOLD    = 500_000   # tune: lower = more sensitive (320x240 frame)
MOTION_COOLDOWN_SEC = 45        # min seconds between Florence re-triggers while occupied
CONFIDENT_COOLDOWN_SEC = 90     # longer cooldown after a confident recognition
REFERENCE_UPDATE_SECS = 5       # how often to refresh the reference frame


def load_config(path="config/settings.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def on_event_triggered(event: dict, latest_frame: dict, florence_state: dict):
    logger.info(f"[FLORENCE] Querying for event: {event['type']}")

    if event["type"] == "person_entered":
        time.sleep(3.0)

    frame = latest_frame["frame"]
    if frame is None:
        return

    save_dir = ROOT / "debug_frames"
    save_dir.mkdir(exist_ok=True)
    timestamp = int(event["timestamp"])
    save_path = save_dir / f"{event['type']}_{timestamp}.jpg"
    cv2.imwrite(str(save_path), frame)

    result = send_frame_for_description(frame)

    if "error" in result:
        logger.warning(f"[FLORENCE] Failed: {result['error']}")
        return

    description = result.get("description", "")
    confident   = result.get("confident", False)
    label       = result.get("label", "")
    score       = result.get("score", 0.0)

    if confident:
        logger.info(f"[INSIGHT] {label} (confidence: {score})")
        # Extend cooldown — confident recognition, stay quiet longer (MAY ENCOUNTER BUG IN FUTURE)
        florence_state["last_confident_at"] = time.time()
    else:
        queue_result = queue_pending(frame, description, event["type"], score)
        logger.info(
            f"[INSIGHT] Uncertain ({score}) — queued for inbox. "
            f"{queue_result.get('total_pending', '?')} pending."
        )


def _cooldown_expired(florence_state: dict) -> bool:
    now = time.time()
    since_trigger  = now - florence_state["last_trigger_at"]
    since_confident = now - florence_state["last_confident_at"]
    # Respect the longer cooldown if we recently had a confident hit
    if since_confident < CONFIDENT_COOLDOWN_SEC:
        return False
    return since_trigger >= MOTION_COOLDOWN_SEC


def run(source=None, loop=False):
    config = load_config()
    cam_cfg = config["camera"]

    video_source = source if source is not None else cam_cfg["source"]

    detector    = ObjectDetector(config["models"]["yolo"])
    detector.load()
    db          = EventDatabase(config["database"]["path"])
    fps_counter = FPSCounter(window=30)
    mem_monitor = MemoryMonitor()
    state_tracker = RoomStateTracker(empty_grace_seconds=3.0)

    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   cam_cfg["buffer_size"])

    if not cap.isOpened():
        logger.error(f"Cannot open video source: {video_source}")
        return

    logger.info(f"Pipeline started | Source: {video_source}")
    logger.info(f"Initial RAM: {mem_monitor.report()}")

    trigger_classes  = set(config["detection"]["trigger_classes"])
    last_stats_log   = time.time()
    stats_interval   = config["database"]["log_fps_interval"]
    PROCESS_EVERY_N  = 3
    frame_count      = 0

    # Shared state between main loop and Florence threads
    latest_frame  = {"frame": None}
    florence_state = {
        "last_trigger_at":   0.0,
        "last_confident_at": 0.0,
    }

    # Motion delta state
    ref_frame = {"frame": None, "updated_at": 0.0}

    motion_debug_count = {"n": 0}

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

            if frame_count % PROCESS_EVERY_N != 0:
                continue

            small_frame = cv2.resize(frame, (320, 240))
            detections  = detector.detect(small_frame)
            latest_frame["frame"] = frame.copy()

            room_state, event = state_tracker.update(detections)

            # --- State-machine triggered events (enter / leave) ---
            if event:
                db.log_room_event(event)
                florence_state["last_trigger_at"] = time.time()
                threading.Thread(
                    target=on_event_triggered,
                    args=(event, latest_frame, florence_state),
                    daemon=True
                ).start()
                # Fresh reference after a state-change trigger
                ref_frame["frame"]      = small_frame.copy()
                ref_frame["updated_at"] = time.time()

            # --- Motion delta re-trigger while occupied ---
            elif room_state in (RoomState.OCCUPIED, RoomState.JUST_ENTERED):
                now = time.time()

                if now - ref_frame["updated_at"] > REFERENCE_UPDATE_SECS:
                    if ref_frame["frame"] is None:
                        ref_frame["frame"]      = small_frame.copy()
                        ref_frame["updated_at"] = now
                        logger.debug("[MOTION] Reference seeded (first frame in occupied state)")
                    else:
                        diff         = cv2.absdiff(small_frame, ref_frame["frame"])
                        motion_score = int(diff.sum())
                        cooldown_ok  = _cooldown_expired(florence_state)

                        # ── DEBUG: always log the score and cooldown state ──────────
                        since_trigger   = now - florence_state["last_trigger_at"]
                        since_confident = now - florence_state["last_confident_at"]
                        logger.debug(
                            f"[MOTION] score={motion_score:,} | threshold={MOTION_THRESHOLD:,} | "
                            f"score_ok={motion_score > MOTION_THRESHOLD} | cooldown_ok={cooldown_ok} | "
                            f"since_trigger={since_trigger:.1f}s | since_confident={since_confident:.1f}s"
                        )

                        # ── DEBUG: save diff visualisation every check ───────────────
                        n = motion_debug_count["n"]
                        motion_debug_count["n"] += 1
                        debug_dir = ROOT / "debug_frames" / "motion"
                        debug_dir.mkdir(parents=True, exist_ok=True)

                        # side-by-side: ref | current | diff (amplified)
                        diff_amplified = cv2.convertScaleAbs(diff, alpha=5)
                        diff_color     = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_HOT)
                        composite      = cv2.hconcat([ref_frame["frame"], small_frame, diff_color])
                        label_text     = (
                            f"score={motion_score:,} thresh={MOTION_THRESHOLD:,} "
                            f"cooldown={'ok' if cooldown_ok else 'BLOCKED'}"
                        )
                        cv2.putText(composite, label_text, (5, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                        cv2.imwrite(str(debug_dir / f"motion_{n:04d}.jpg"), composite)
                        # ─────────────────────────────────────────────────────────────

                        if motion_score > MOTION_THRESHOLD and cooldown_ok:
                            logger.info(
                                f"[MOTION] TRIGGERED — score {motion_score:,} > threshold, "
                                f"saving trigger frame"
                            )
                            cv2.imwrite(str(debug_dir / f"TRIGGER_{n:04d}.jpg"), composite)
                            activity_event = {
                                "type":      "activity_change",
                                "timestamp": now,
                            }
                            florence_state["last_trigger_at"] = now
                            threading.Thread(
                                target=on_event_triggered,
                                args=(activity_event, latest_frame, florence_state),
                                daemon=True
                            ).start()
                        else:
                            reason = []
                            if motion_score <= MOTION_THRESHOLD:
                                reason.append(f"score {motion_score:,} <= threshold {MOTION_THRESHOLD:,}")
                            if not cooldown_ok:
                                reason.append(f"cooldown not expired (need {MOTION_COOLDOWN_SEC}s since trigger, {since_trigger:.1f}s elapsed)")
                            logger.debug(f"[MOTION] No trigger — {' AND '.join(reason)}")

                        ref_frame["frame"]      = small_frame.copy()
                        ref_frame["updated_at"] = now

            else:
                ref_frame["frame"]      = None
                ref_frame["updated_at"] = 0.0

            # --- Per-detection logging ---
            for det in detections:
                if det["class_name"] in trigger_classes:
                    db.log_detection(detection=det)

            # --- Periodic stats ---
            now = time.time()
            if now - last_stats_log >= stats_interval:
                fps     = fps_counter.fps
                ram_mb  = mem_monitor.process_mb()
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