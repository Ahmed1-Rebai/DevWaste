"""SQLite storage for container metrics."""

import sqlite3
from dataclasses import astuple, fields
from pathlib import Path

import pandas as pd

from devwaste.collectors.docker_collector import ContainerMetric
from devwaste.config import DEFAULT_DB_PATH

METRIC_COLUMNS = [f.name for f in fields(ContainerMetric)]


class SQLiteStore:
    """SQLite-based storage for container metrics."""

    def __init__(self, db_path: Path | None = None):
        """Initialize SQLite connection."""
        db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._create_tables()

    def _create_tables(self):
        """Create metrics table if not exists."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                container_id TEXT NOT NULL,
                container_name TEXT NOT NULL,
                state TEXT NOT NULL,
                cpu_percent REAL,
                memory_usage REAL,
                memory_limit REAL,
                memory_percent REAL,
                network_rx_bytes REAL,
                network_tx_bytes REAL,
                block_read_bytes REAL,
                block_write_bytes REAL,
                restart_count INTEGER
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_container_time "
            "ON metrics (container_id, timestamp)"
        )
        self.conn.commit()

    def save_metric(self, metric: ContainerMetric):
        """Save a single metric observation."""
        self.save_metrics([metric])

    def save_metrics(self, metrics: list[ContainerMetric]):
        """Save multiple metric observations in one transaction."""
        placeholders = ", ".join("?" for _ in METRIC_COLUMNS)
        self.conn.executemany(
            f"INSERT INTO metrics ({', '.join(METRIC_COLUMNS)}) VALUES ({placeholders})",
            [astuple(m) for m in metrics],
        )
        self.conn.commit()

    def load_metrics(
        self,
        container_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        """Load metrics as DataFrame."""
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []

        if container_id:
            query += " AND container_id = ?"
            params.append(container_id)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        return df

    def get_latest_metrics(self, container_id: str, n: int = 1) -> pd.DataFrame:
        """Get the latest N metrics for a container."""
        query = """
            SELECT * FROM metrics
            WHERE container_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, self.conn, params=[container_id, n])
        return df.sort_values("timestamp")

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
