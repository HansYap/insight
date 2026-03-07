import sqlite3
from pathlib import Path
from loguru import logger

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
            # Keep the index if you plan on querying data later
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp)")
        logger.info(f"Database ready at {self.db_path}")

    def log_detection(self, scene: str, detection: dict):
        b = detection["bbox"]
        try:
            with self.conn: # This acts as a transaction context
                self.conn.execute(
                    """INSERT INTO detections 
                       (scene, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (scene, detection["class_name"], detection["confidence"], *b)
                )
        except Exception as e:
            logger.error(f"Failed to log detection: {e}")

    def log_stats(self, fps: float, process_ram_mb: float, system_ram_pct: float):
        with self.conn:
            self.conn.execute(
                "INSERT INTO system_stats (fps, process_ram_mb, system_ram_pct) VALUES (?, ?, ?)",
                (fps, process_ram_mb, system_ram_pct)
            )

    def __del__(self):
        self.conn.close()