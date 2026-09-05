# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SQLite schema and forward-only migrations.

Aggregates only: per-bucket metrics, track lifecycles and discrete events. Per-frame bounding boxes are never
stored.
"""

from __future__ import annotations

#: Each entry is one migration; index + 1 is the resulting schema version.
#: Statements are built from single-line string literals (not triple-quoted
#: blocks) because docformatter's tokenizer crashes on multi-line string
#: literals that sit inside a list/tuple literal rather than in docstring
#: position.
MIGRATIONS: list[tuple[str, ...]] = [
    (
        "CREATE TABLE IF NOT EXISTS stream_health ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, stream_id TEXT NOT NULL, at REAL NOT NULL, "
        "state TEXT NOT NULL, health TEXT NOT NULL, processed_fps REAL NOT NULL, "
        "queue_depth INTEGER NOT NULL, dropped_frames INTEGER NOT NULL, last_error TEXT)",
        "CREATE TABLE IF NOT EXISTS detection_summaries ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, stream_id TEXT NOT NULL, bucket_start REAL NOT NULL, "
        "class_name TEXT NOT NULL, detections INTEGER NOT NULL, mean_confidence REAL NOT NULL)",
        "CREATE TABLE IF NOT EXISTS confidence_histogram ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, stream_id TEXT NOT NULL, bucket_start REAL NOT NULL, "
        "bin_index INTEGER NOT NULL, count INTEGER NOT NULL)",
        "CREATE INDEX IF NOT EXISTS ix_confidence_histogram_at "
        "ON confidence_histogram (stream_id, bucket_start)",
        "CREATE TABLE IF NOT EXISTS tracks ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, stream_id TEXT NOT NULL, track_id INTEGER NOT NULL, "
        "class_name TEXT NOT NULL, first_seen REAL NOT NULL, last_seen REAL NOT NULL, "
        "frames INTEGER NOT NULL, mean_confidence REAL NOT NULL, "
        "UNIQUE (stream_id, track_id, first_seen))",
        "CREATE TABLE IF NOT EXISTS zone_transitions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, stream_id TEXT NOT NULL, zone TEXT NOT NULL, "
        "track_id INTEGER NOT NULL, class_name TEXT NOT NULL, entered_at REAL NOT NULL, "
        "exited_at REAL, dwell_seconds REAL, "
        "UNIQUE (stream_id, zone, track_id, entered_at))",
        "CREATE INDEX IF NOT EXISTS ix_zone_transitions_at ON zone_transitions (stream_id, entered_at)",
        "CREATE TABLE IF NOT EXISTS line_crossings ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, stream_id TEXT NOT NULL, line TEXT NOT NULL, "
        "track_id INTEGER NOT NULL, class_name TEXT NOT NULL, at REAL NOT NULL, "
        "direction TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS events ("
        "event_id TEXT PRIMARY KEY, stream_id TEXT NOT NULL, event_type TEXT NOT NULL, "
        "at REAL NOT NULL, severity TEXT NOT NULL, details TEXT NOT NULL, artifact_id TEXT)",
        "CREATE INDEX IF NOT EXISTS ix_events_at ON events (stream_id, at)",
        "CREATE TABLE IF NOT EXISTS errors ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, source TEXT NOT NULL, "
        "code TEXT NOT NULL, message TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS ix_errors_at ON errors (at)",
        "CREATE TABLE IF NOT EXISTS artifacts ("
        "artifact_id TEXT PRIMARY KEY, kind TEXT NOT NULL, media_type TEXT NOT NULL, "
        "relative_path TEXT NOT NULL, bytes INTEGER NOT NULL, created_at REAL NOT NULL, stream_id TEXT)",
        "CREATE INDEX IF NOT EXISTS ix_artifacts_created ON artifacts (created_at)",
    ),
    (
        "CREATE TABLE IF NOT EXISTS latency_histogram ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, stream_id TEXT NOT NULL, bucket_start REAL NOT NULL, "
        "latency_ms REAL NOT NULL, count INTEGER NOT NULL)",
        "CREATE INDEX IF NOT EXISTS ix_latency_histogram_at ON latency_histogram (stream_id, bucket_start)",
    ),
]

#: Tables pruned by retention, with the column holding their timestamp.
RETENTION_TABLES: tuple[tuple[str, str], ...] = (
    ("stream_health", "at"),
    ("processing_metrics", "bucket_start"),
    ("detection_summaries", "bucket_start"),
    ("confidence_histogram", "bucket_start"),
    ("latency_histogram", "bucket_start"),
    ("tracks", "last_seen"),
    ("zone_transitions", "entered_at"),
    ("line_crossings", "at"),
    ("events", "at"),
    ("errors", "at"),
)
