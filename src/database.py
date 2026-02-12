import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "anpr_results.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    vehicle_id INTEGER,
                    image_path TEXT
                )
            ''')
            conn.commit()
            logger.info("Database initialized.")

    def log_detection(self, plate_number: str, confidence: float, vehicle_id: int = None, image_path: str = None):
        """
        Log detection to database with error handling.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO detections (plate_number, timestamp, confidence, vehicle_id, image_path)
                    VALUES (?, ?, ?, ?, ?)
                ''', (plate_number, timestamp, confidence, vehicle_id, image_path))
                conn.commit()
                logger.debug(f"Logged: {plate_number}")
        except Exception as e:
            logger.error(f"Database logging failed: {e}")

    def get_recent_log(self, limit: int = 10):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?", (limit,))
            return cursor.fetchall()
