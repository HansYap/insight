import sqlite3
from pathlib import Path
from loguru import logger
import json
import datetime
import numpy as np
import time

class TrainingDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep one connection open for the lifetime of the object
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     REAL NOT NULL,
                    source        TEXT NOT NULL,
                    score         REAL NOT NULL,
                    embedding     TEXT NOT NULL,
                    label         TEXT NOT NULL
                )      
            """)
        logger.info(f"Database ready at {self.db_path}")

    
    def log_training_data(self, source: str, score: float, embedding: np.ndarray, label: str,) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO training_data
                (timestamp, source, score, embedding, label)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    source,
                    score,
                    embedding,
                    label,
                )
            )

    def __del__(self):
        self.conn.close()