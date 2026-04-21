import cv2
import yaml
import time
import threading
import collections
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from core.detector import ObjectDetector
from core.sql_database import EventDatabase
from utils.fps_counter import FPSCounter
from utils.memory_monitor import MemoryMonitor
from core.room_state import RoomStateTracker, RoomState
from core.frame_client import send_frame_for_description, queue_pending, save_frame_remote

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data classes — no more raw dicts with string keys
# ---------------------------------------------------------------------------

@dataclass
class MotionConfig:
    threshold: int   = 500_000  # lower = more sensitive (320×240 frame)
    reference_update_secs: int = 5


@dataclass
class FlorenceConfig:
    motion_cooldown_sec: int    = 45
    confident_cooldown_sec: int = 90
    person_entry_delay_sec: float = 3.0


@dataclass
class FlorenceState:
    last_trigger_at:        float = 0.0  # entry/exit events
    last_motion_trigger_at: float = 0.0  # motion delta re-triggers only
    last_confident_at:      float = 0.0


# ---------------------------------------------------------------------------
# MotionDetector — owns reference frame, scoring, cooldown, debug saves
# ---------------------------------------------------------------------------

class MotionDetector:
    """
    Compares successive small frames to detect significant activity change
    while a person is present. Owns all motion-delta state.
    """

    def __init__(self, config: MotionConfig):
        self.config = config
        self._ref_frame: cv2.typing.MatLike | None = None
        self._ref_updated_at: float = 0.0
        self._debug_count: int = 0

    def reset(self) -> None:
        """Call when room becomes empty — clears stale reference."""
        self._ref_frame = None
        self._ref_updated_at = 0.0

    def seed(self, small_frame) -> None:
        """Call after an entry/exit event to start fresh."""
        self._ref_frame = small_frame.copy()
        self._ref_updated_at = time.time()

    def check(self, small_frame, cooldown_ok: bool) -> tuple[bool, int]:
        """
        Returns (triggered: bool, motion_score: int).
        Should be called every frame while room is occupied.
        Only does the expensive diff when the reference update interval has elapsed.
        """
        now = time.time()

        if now - self._ref_updated_at < self.config.reference_update_secs:
            return False, 0

        if self._ref_frame is None:
            self._seed_first_reference(small_frame, now)
            return False, 0

        motion_score = self._compute_score(small_frame)
        triggered    = motion_score > self.config.threshold and cooldown_ok

        self._log_debug(small_frame, motion_score, cooldown_ok, triggered)

        self._ref_frame     = small_frame.copy()
        self._ref_updated_at = now

        return triggered, motion_score

    # --- private helpers ----------------------------------------------------

    def _seed_first_reference(self, small_frame, now: float) -> None:
        self._ref_frame      = small_frame.copy()
        self._ref_updated_at = now
        logger.debug("[MOTION] Reference seeded (first frame in occupied state)")

    def _compute_score(self, small_frame) -> int:
        diff = cv2.absdiff(small_frame, self._ref_frame)
        return int(diff.sum())

    def _log_debug(self, small_frame, motion_score: int, cooldown_ok: bool, triggered: bool) -> None:
        n = self._debug_count
        self._debug_count += 1

        logger.debug(
            f"[MOTION] score={motion_score:,} | threshold={self.config.threshold:,} | "
            f"score_ok={motion_score > self.config.threshold} | cooldown_ok={cooldown_ok}"
        )

        diff           = cv2.absdiff(small_frame, self._ref_frame)
        diff_amplified = cv2.convertScaleAbs(diff, alpha=5)
        diff_color     = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_HOT)
        composite      = cv2.hconcat([self._ref_frame, small_frame, diff_color])
        label_text     = (
            f"score={motion_score:,} thresh={self.config.threshold:,} "
            f"cooldown={'ok' if cooldown_ok else 'BLOCKED'}"
        )
        cv2.putText(composite, label_text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        prefix = "TRIGGER" if triggered else "motion"
        threading.Thread(
            target=save_frame_remote,
            args=(composite, f"motion/{prefix}_{n:04d}.jpg"),
            daemon=True,
        ).start()


# ---------------------------------------------------------------------------
# EventDispatcher — owns Florence state, threading, inbox queuing
# ---------------------------------------------------------------------------

class EventDispatcher:
    """
    Dispatches triggered events to Florence-2 in background threads.
    Owns cooldown state and the human-in-the-loop inbox queue logic.
    """

    def __init__(self, config: FlorenceConfig):
        self.config = config
        self._state = FlorenceState()

    # Public interface -------------------------------------------------------

    def dispatch(self, event: dict, latest_frame: dict) -> None:
        """Fire-and-forget: spawns a daemon thread per event."""
        threading.Thread(
            target=self._handle_event,
            args=(event, latest_frame),
            daemon=True,
        ).start()

    def motion_cooldown_ok(self) -> bool:
        now = time.time()
        if now - self._state.last_confident_at < self.config.confident_cooldown_sec:
            return False
        return (now - self._state.last_motion_trigger_at) >= self.config.motion_cooldown_sec

    def mark_entry_exit_trigger(self) -> None:
        self._state.last_trigger_at = time.time()

    def mark_motion_trigger(self) -> None:
        self._state.last_motion_trigger_at = time.time()

    # Private — runs in background thread ------------------------------------

    def _handle_event(self, event: dict, latest_frame: dict) -> None:
        logger.info(f"[FLORENCE] Querying for event: {event['type']}")

        if event["type"] == "person_entered":
            time.sleep(self.config.person_entry_delay_sec)

        frame = latest_frame["frame"]
        if frame is None:
            return

        self._save_trigger_frame(frame, event)
        self._query_florence(frame, event)

    def _save_trigger_frame(self, frame, event: dict) -> None:
        timestamp = int(event["timestamp"])
        threading.Thread(
            target=save_frame_remote,
            args=(frame, f"{event['type']}_{timestamp}.jpg"),
            daemon=True,
        ).start()

    def _query_florence(self, frame, event: dict) -> None:
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
            self._state.last_confident_at = time.time()
        else:
            queue_result = queue_pending(frame, description, event["type"], score)
            logger.info(
                f"[INSIGHT] Uncertain ({score:.2f}) — queued for inbox. "
                f"{queue_result.get('total_pending', '?')} pending."
            )


# ---------------------------------------------------------------------------
# Farneback (placeholder — implement when ready)
# ---------------------------------------------------------------------------

def farneback_calculate(frames: list) -> dict | None:
    """
    Compute optical flow statistics across a buffer of full-res frames.
    Returns a dict of scalar stats to be attached to the event record,
    or None if the buffer is too short.
    """
    
    if len(frames) < 2:
        return None
    
    flow_store = []
    COVERAGE_THRESHOLD = 0.5

    for f in range(1, len(frames)):
        prev_frame = cv2.cvtColor(frames[f-1], cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)  
        flow_store.append(flow)
    
    avg_flow = np.mean(flow_store, axis=0) # (H,W,2)
    # extract horizontal and vertical displacement per pixel
    dx = avg_flow[..., 0]
    dy = avg_flow[..., 1]

    magnitude_pixel = np.hypot(dx, dy)
    mean_magnitude = np.mean(magnitude_pixel)

    std_magnitude = np.std(magnitude_pixel)

    # calculate directionality
    angle_pixel = np.arctan2(dy, dx)
    circular_variance = 1 - np.abs(np.mean(np.exp(1j * angle_pixel)))

    coverage_ratio = np.mean(magnitude_pixel > COVERAGE_THRESHOLD)

    dominant_angle = np.arctan2(np.mean(np.sin(angle_pixel)), np.mean(np.cos(angle_pixel)))
    dominant_sin = np.sin(dominant_angle)
    dominant_cos = np.cos(dominant_angle)

    v_motion = {
        "mean_magnitude": mean_magnitude,
        "std_magnitude": std_magnitude,
        "directionality": circular_variance,
        "coverage_ratio": coverage_ratio,
        "dominant_sin": dominant_sin,
        "dominant_cos": dominant_cos,
    }

    logger.debug(
                f"[V_MOTION] mean_magnitude ({mean_magnitude:.2f}),"
                f"[V_MOTION] std_magnitude ({std_magnitude:.2f}),"
                f"[V_MOTION] directionality ({circular_variance:.2f}),"
                f"[V_MOTION] coverage_ratio ({coverage_ratio:.2f}),"
                f"[V_MOTION] dominant_sin ({dominant_sin:.2f}),"
                f"[V_MOTION] dominant_cos ({dominant_cos:.2f}),"
            )

    return v_motion


# ---------------------------------------------------------------------------
# Main pipeline loop — thin orchestrator, delegates everything
# ---------------------------------------------------------------------------

def run(source=None, loop=False) -> None:
    config  = load_config()
    cam_cfg = config["camera"]

    video_source = source if source is not None else cam_cfg["source"]

    detector      = ObjectDetector(config["models"]["yolo"])
    detector.load()
    db            = EventDatabase(config["database"]["path"])
    # for debug purpose
    fps_counter   = FPSCounter(window=30)
    mem_monitor   = MemoryMonitor()

    state_tracker = RoomStateTracker(empty_grace_seconds=5.0)

    motion    = MotionDetector(MotionConfig())
    dispatch  = EventDispatcher(FlorenceConfig())

    cap = cv2.VideoCapture(cam_cfg.get("source", 0))
    cap.set(cv2.CAP_PROP_FOURCC,    cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS,          cam_cfg["fps_target"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   cam_cfg["buffer_size"])

    if not cap.isOpened():
        logger.error(f"Cannot open video source: {video_source}")
        return

    logger.info(f"Pipeline started | Source: {video_source}")
    logger.info(f"Initial RAM: {mem_monitor.report()}")

    trigger_classes = set(config["detection"]["trigger_classes"])
    PROCESS_EVERY_N = 3
    BUFFER_LENGTH   = 10
    frame_count     = 0
    last_stats_log  = time.time()
    stats_interval  = config["database"]["log_fps_interval"]

    # account for yolo early trigger
    latest_frame = {"frame": None}

    # farneback/ v_motion
    frame_buffer = collections.deque(maxlen=BUFFER_LENGTH)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Frame read failed — end of stream or disconnected camera")
                break

            frame_count += 1
            fps_counter.tick()

            if frame_count % PROCESS_EVERY_N != 0:
                continue

            small_frame = cv2.resize(frame, (320, 240))
            detections  = detector.detect(small_frame)

            latest_frame["frame"] = frame.copy()
            frame_buffer.append(frame.copy())

            room_state, event = state_tracker.update(detections)

            # --- Entry / exit event -----------------------------------------
            if event:
                db.log_room_event(event)
                dispatch.mark_entry_exit_trigger()
                dispatch.dispatch(event, latest_frame)
                motion_stats = farneback_calculate(list(frame_buffer))
                if motion_stats:
                    db.log_motion_stats(event, motion_stats)
                motion.seed(small_frame)

            # --- Motion delta while occupied ---------------------------------
            elif room_state in (RoomState.OCCUPIED, RoomState.JUST_ENTERED):
                triggered, _ = motion.check(small_frame, dispatch.motion_cooldown_ok())
                if triggered:
                    activity_event = {"type": "activity_change", "timestamp": time.time()}
                    dispatch.mark_motion_trigger()
                    dispatch.dispatch(activity_event, latest_frame)
                    motion_stats = farneback_calculate(list(frame_buffer))
                    if motion_stats:
                        db.log_motion_stats(activity_event, motion_stats)


            # --- Room empty --------------------------------------------------
            else:
                motion.reset()

            # --- Per-detection logging ---------------------------------------
            for det in detections:
                if det["class_name"] in trigger_classes:
                    db.log_detection(detection=det)

            # --- Periodic stats ---------------------------------------------
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
    parser.add_argument("source", nargs="?", default=None)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    run(source=args.source, loop=args.loop)