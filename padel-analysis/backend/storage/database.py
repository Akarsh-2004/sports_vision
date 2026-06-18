"""Sports analytics database — queryable match intelligence store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SportsAnalyticsDB:
    """
    Layer 7 storage — not just detections.json.

    Tables: matches, rallies, shots, players, movements, events, highlights, stats.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        source_video TEXT,
        duration_s REAL,
        data_json TEXT
    );
    CREATE TABLE IF NOT EXISTS rallies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT,
        start_frame INTEGER,
        end_frame INTEGER,
        shot_count INTEGER,
        interaction_chain TEXT,
        data_json TEXT
    );
    CREATE TABLE IF NOT EXISTS shots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT,
        frame_idx INTEGER,
        player_id INTEGER,
        stroke TEXT,
        intent TEXT,
        speed_kmh REAL,
        player_zone TEXT,
        position_x REAL,
        position_y REAL,
        outcome TEXT,
        data_json TEXT
    );
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT,
        frame_idx INTEGER,
        event_type TEXT,
        data_json TEXT
    );
    CREATE TABLE IF NOT EXISTS tactical (
        match_id TEXT PRIMARY KEY,
        data_json TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_shots_match ON shots(match_id);
    CREATE INDEX IF NOT EXISTS idx_shots_stroke ON shots(stroke);
    CREATE INDEX IF NOT EXISTS idx_rallies_match ON rallies(match_id);
    """

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(self.SCHEMA)

    def save_match(self, match_id: str, source_video: str, duration_s: float, meta: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO matches (match_id, source_video, duration_s, data_json) VALUES (?,?,?,?)",
            (match_id, source_video, duration_s, json.dumps(meta)),
        )
        self.conn.commit()

    def save_shots(self, match_id: str, shot_facts: list[dict]) -> None:
        for s in shot_facts:
            pos = s.get("player_position") or s.get("landing") or (0, 0)
            self.conn.execute(
                """INSERT INTO shots (match_id, frame_idx, player_id, stroke, intent,
                   speed_kmh, player_zone, position_x, position_y, outcome, data_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    match_id,
                    s.get("frame", 0),
                    0,
                    s.get("shot"),
                    s.get("intent"),
                    s.get("speed_kmh", 0),
                    s.get("player_zone"),
                    pos[0] if pos else 0,
                    pos[1] if pos else 1,
                    s.get("outcome"),
                    json.dumps(s),
                ),
            )
        self.conn.commit()

    def save_rallies(self, match_id: str, rallies: list[dict]) -> None:
        for r in rallies:
            self.conn.execute(
                """INSERT INTO rallies (match_id, start_frame, end_frame, shot_count,
                   interaction_chain, data_json) VALUES (?,?,?,?,?,?)""",
                (
                    match_id,
                    r.get("start_frame", 0),
                    r.get("end_frame", 0),
                    r.get("shot_count", 0),
                    r.get("interaction_chain", ""),
                    json.dumps(r),
                ),
            )
        self.conn.commit()

    def save_tactical(self, match_id: str, tactical: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO tactical (match_id, data_json) VALUES (?,?)",
            (match_id, json.dumps(tactical)),
        )
        self.conn.commit()

    def query_shots(self, match_id: str, stroke: str | None = None) -> list[dict]:
        if stroke:
            rows = self.conn.execute(
                "SELECT data_json FROM shots WHERE match_id=? AND stroke=?",
                (match_id, stroke),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT data_json FROM shots WHERE match_id=?", (match_id,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def close(self) -> None:
        self.conn.close()
