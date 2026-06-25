import sqlite3
from pathlib import Path
from loguru import logger
import json
import datetime
import numpy as np
import time

class EventDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep one connection open for the lifetime of the object
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        # Use 'DEFAULT CURRENT_TIMESTAMP' to let SQLite handle timing
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    scene TEXT,
                    class_name TEXT,
                    confidence REAL,
                    bbox_x1 REAL, bbox_y1 REAL,
                    bbox_x2 REAL, bbox_y2 REAL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    fps REAL,
                    process_ram_mb REAL,
                    system_ram_pct REAL
                )
            """)
            # keep indexfor query ltr
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp)")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS room_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    duration_seconds REAL,
                    context_json TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS motion_stats (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     REAL NOT NULL,
                    event_type    TEXT NOT NULL,
                    mean_magnitude  REAL,
                    std_magnitude   REAL,
                    directionality  REAL,
                    coverage_ratio  REAL,
                    dominant_sin    REAL,
                    dominant_cos    REAL  
                )      
            """)
            
        logger.info(f"Database ready at {self.db_path}")

    # YOLO detections
    def log_detection(self, detection: dict):
        b = detection["bbox"]
        try:
            with self.conn: # This acts as a transaction context
                self.conn.execute(
                    """INSERT INTO detections 
                       (class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (detection["class_name"], detection["confidence"], *b)
                )
        except Exception as e:
            logger.error(f"Failed to log detection: {e}")
    
    # event logs 
    def log_room_event(self, event: dict):
        """Logs transitions: person_entered, person_left"""
        with self.conn:
            self.conn.execute(
                """INSERT INTO room_events 
                (timestamp, event_type, duration_seconds, context_json)
                VALUES (?, ?, ?, ?)""",
                (
                    datetime.datetime.now().isoformat(),
                    event["type"],
                    event.get("duration_away_seconds") or event.get("duration_in_room_seconds"),
                    json.dumps(event)
                )
            )

    # log system stats
    def log_stats(self, fps: float, process_ram_mb: float, system_ram_pct: float):
        with self.conn:
            self.conn.execute(
                "INSERT INTO system_stats (fps, process_ram_mb, system_ram_pct) VALUES (?, ?, ?)",
                (fps, process_ram_mb, system_ram_pct)
            )
    
    # motion stats
    def log_motion_stats(self, event: dict, stats: dict) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO motion_stats
                (timestamp, event_type, mean_magnitude, std_magnitude,
                    directionality, coverage_ratio, dominant_sin, dominant_cos)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    event["type"],
                    float(stats["mean_magnitude"]),
                    float(stats["std_magnitude"]),
                    float(stats["directionality"]),
                    float(stats["coverage_ratio"]),
                    float(stats["dominant_sin"]),
                    float(stats["dominant_cos"]),
                )
            )
    

    def __del__(self):
        self.conn.close()