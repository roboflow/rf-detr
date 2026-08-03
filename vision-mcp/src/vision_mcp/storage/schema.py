"""SQLite schema and forward-only migrations.

Aggregates only: per-bucket metrics, track lifecycles and discrete events. Per-frame
bounding boxes are never stored.
"""

from __future__ import annotations

#: Each entry is one migration; index + 1 is the resulting schema version.
MIGRATIONS: list[tuple[str, ...]] = [
    (
        """
        CREATE TABLE IF NOT EXISTS stream_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id TEXT NOT NULL,
            at REAL NOT NULL,
            state TEXT NOT NULL,
            health TEXT NOT NULL,
            processed_fps REAL NOT NULL,
            queue_depth INTEGER NOT NULL,
            dropped_frames INTEGER NOT NULL,
            last_error TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_stream_health_at ON stream_health (stream_id, at)",
        """
        CREATE TABLE IF NOT EXISTS processing_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id TEXT NOT NULL,
            bucket_start REAL NOT NULL,
            bucket_seconds REAL NOT NULL,
            captured_frames INTEGER NOT NULL,
            processed_frames INTEGER NOT NULL,
            dropped_frames INTEGER NOT NULL,
            frame_detections INTEGER NOT NULL,
            latency_samples INTEGER NOT NULL,
            latency_mean_ms REAL,
            latency_p50_ms REAL,
            latency_p95_ms REAL,
            latency_p99_ms REAL,
            latency_max_ms REAL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_processing_metrics_at ON processing_metrics (stream_id, bucket_start)",
        """
        CREATE TABLE IF NOT EXISTS detection_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id TEXT NOT NULL,
            bucket_start REAL NOT NULL,
            class_name TEXT NOT NULL,
            detections INTEGER NOT NULL,
            mean_confidence REAL NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_detection_summaries_at "
        "ON detection_summaries (stream_id, bucket_start)",
        """
        CREATE TABLE IF NOT EXISTS confidence_histogram (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id TEXT NOT NULL,
            bucket_start REAL NOT NULL,
            bin_index INTEGER NOT NULL,
            count INTEGER NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_confidence_histogram_at "
        "ON confidence_histogram (stream_id, bucket_start)",
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            frames INTEGER NOT NULL,
            mean_confidence REAL NOT NULL,
            UNIQUE (stream_id, track_id, first_seen)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_tracks_seen ON tracks (stream_id, last_seen)",
        """
        CREATE TABLE IF NOT EXISTS zone_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id TEXT NOT NULL,
            zone TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            entered_at REAL NOT NULL,
            exited_at REAL,
            dwell_seconds REAL,
            UNIQUE (stream_id, zone, track_id, entered_at)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_zone_transitions_at ON zone_transitions (stream_id, entered_at)",
        """
        CREATE TABLE IF NOT EXISTS line_crossings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id TEXT NOT NULL,
            line TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            at REAL NOT NULL,
            direction TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_line_crossings_at ON line_crossings (stream_id, at)",
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            at REAL NOT NULL,
            severity TEXT NOT NULL,
            details TEXT NOT NULL,
            artifact_id TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_events_at ON events (stream_id, at)",
        """
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            at REAL NOT NULL,
            source TEXT NOT NULL,
            code TEXT NOT NULL,
            message TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_errors_at ON errors (at)",
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            media_type TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at REAL NOT NULL,
            stream_id TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_artifacts_created ON artifacts (created_at)",
    ),
]

#: Tables pruned by retention, with the column holding their timestamp.
RETENTION_TABLES: tuple[tuple[str, str], ...] = (
    ("stream_health", "at"),
    ("processing_metrics", "bucket_start"),
    ("detection_summaries", "bucket_start"),
    ("confidence_histogram", "bucket_start"),
    ("tracks", "last_seen"),
    ("zone_transitions", "entered_at"),
    ("line_crossings", "at"),
    ("events", "at"),
    ("errors", "at"),
)
