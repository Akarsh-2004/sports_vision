"""Persistent learning database — multi-match player history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.intelligence.shot.understanding import ShotUnderstanding
from backend.intelligence.world.world_model import WorldModel
from backend.storage.database import SportsAnalyticsDB


class LearningDatabase(SportsAnalyticsDB):
    """Extends analytics DB with player/match history for multi-match intelligence."""

    EXTRA_SCHEMA = """
    CREATE TABLE IF NOT EXISTS player_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER,
        match_id TEXT,
        shot_count INTEGER,
        tactical_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS shot_understanding (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT,
        player_id INTEGER,
        frame_idx INTEGER,
        data_json TEXT
    );
  CREATE INDEX IF NOT EXISTS idx_player_sessions ON player_sessions(player_id);
    """

    def __init__(self, db_path: str | Path):
        super().__init__(db_path)
        self.conn.executescript(self.EXTRA_SCHEMA)

    def save_match(
        self,
        world: WorldModel,
        match_id: str,
        source_video: str,
        duration_s: float,
        shots: list[ShotUnderstanding],
        rally_graphs: list[dict],
    ) -> None:
        super().save_match(match_id, source_video, duration_s, world.summary())
        super().save_rallies(match_id, rally_graphs)
        for s in shots:
            self.conn.execute(
                "INSERT INTO shot_understanding (match_id, player_id, frame_idx, data_json) VALUES (?,?,?,?)",
                (match_id, s.player_id, s.frame_idx, json.dumps(s.to_dict())),
            )
            super().save_shots(match_id, [s.to_dict()])
        self.conn.commit()

    def save_player_session(
        self, player_id: int, match_id: str, shots: list[ShotUnderstanding], tactical: dict
    ) -> None:
        self.conn.execute(
            "INSERT INTO player_sessions (player_id, match_id, shot_count, tactical_json) VALUES (?,?,?,?)",
            (player_id, match_id, len(shots), json.dumps(tactical)),
        )
        self.conn.commit()

    def player_history(self, player_id: int, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT match_id, shot_count, tactical_json, created_at FROM player_sessions "
            "WHERE player_id=? ORDER BY id DESC LIMIT ?",
            (player_id, limit),
        ).fetchall()
        return [
            {"match_id": r[0], "shots": r[1], "tactical": json.loads(r[2]), "date": r[3]}
            for r in rows
        ]

    def query_shots_by_stroke(self, player_id: int, stroke: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT data_json FROM shot_understanding WHERE player_id=? AND data_json LIKE ?",
            (player_id, f'%"stroke": "{stroke}"%'),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]
