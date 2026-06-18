"""Phase 13: padel event detection (wall winners, net errors, long rallies)."""

from __future__ import annotations

from backend.utils.types import EventType, MatchEvent, RallySegment, ShotQuality, StrokeType


class EventDetector:
    def __init__(self, config: dict):
        self.net_distance = config["analytics"]["net_approach_distance_m"]
        self.long_rally_shots = config.get("events", {}).get("long_rally_shots", 10)
        self.wall_exchange_min = config.get("events", {}).get("wall_exchange_min", 2)
        self.events: list[MatchEvent] = []

    def detect_from_rally(
        self,
        rally: RallySegment,
        last_shot: ShotQuality | None,
        target_track_id: int,
    ) -> list[MatchEvent]:
        found: list[MatchEvent] = []
        if rally.rally_length_shots >= self.long_rally_shots:
            found.append(
                MatchEvent(
                    frame_idx=rally.end_frame,
                    event_type=EventType.LONG_RALLY,
                    player_track_id=target_track_id,
                    metadata={"shots": rally.rally_length_shots},
                )
            )
        if rally.wall_hits >= self.wall_exchange_min:
            found.append(
                MatchEvent(
                    frame_idx=rally.end_frame,
                    event_type=EventType.WALL_EXCHANGE,
                    player_track_id=target_track_id,
                    metadata={"wall_hits": rally.wall_hits},
                )
            )
        if last_shot:
            if last_shot.off_wall and last_shot.in_court:
                found.append(
                    MatchEvent(
                        frame_idx=last_shot.frame_idx,
                        event_type=EventType.WALL_WINNER,
                        player_track_id=target_track_id,
                    )
                )
            elif last_shot.in_court:
                found.append(
                    MatchEvent(
                        frame_idx=last_shot.frame_idx,
                        event_type=EventType.WINNER,
                        player_track_id=target_track_id,
                    )
                )
            elif last_shot.zone == "net":
                found.append(
                    MatchEvent(
                        frame_idx=last_shot.frame_idx,
                        event_type=EventType.NET_ERROR,
                        player_track_id=target_track_id,
                    )
                )
            else:
                found.append(
                    MatchEvent(
                        frame_idx=last_shot.frame_idx,
                        event_type=EventType.UNFORCED_ERROR,
                        player_track_id=target_track_id,
                    )
                )
        self.events.extend(found)
        return found

    def detect_net_approach(self, frame_idx: int, court_y: float, track_id: int, court_length: float) -> MatchEvent | None:
        net_y = court_length / 2
        if abs(court_y - net_y) < self.net_distance:
            ev = MatchEvent(
                frame_idx=frame_idx,
                event_type=EventType.NET_APPROACH,
                player_track_id=track_id,
            )
            self.events.append(ev)
            return ev
        return None

    def detect_smash_winner(self, frame_idx: int, stroke_type: StrokeType, in_court: bool, track_id: int) -> None:
        if stroke_type == StrokeType.SMASH and in_court:
            self.events.append(
                MatchEvent(
                    frame_idx=frame_idx,
                    event_type=EventType.SMASH_WINNER,
                    player_track_id=track_id,
                )
            )
