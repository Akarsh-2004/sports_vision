from __future__ import annotations

from backend.utils.types import EventType, MatchEvent, PoseKeypoints, RallySegment, ShotQuality, StrokeType


class EventDetector:
    """Stage 13: tennis scoring-aware events (ace, double fault, winner, etc.)."""

    def __init__(self, config: dict):
        self.net_distance = config["analytics"]["net_approach_distance_m"]
        self.long_rally_shots = config.get("events", {}).get("long_rally_shots", 8)
        self.serve_toss_threshold = config.get("events", {}).get("serve_toss_height_px", 40)
        self.events: list[MatchEvent] = []
        self._opponent_moved_frames: set[int] = set()
        self._serve_frames: list[int] = []

    def detect_from_rally(
        self,
        rally: RallySegment,
        last_shot: ShotQuality | None,
        target_track_id: int,
        opponent_near_net: bool = False,
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
        if last_shot:
            if last_shot.in_court and rally.rally_length_shots <= 2:
                found.append(
                    MatchEvent(
                        frame_idx=last_shot.frame_idx,
                        event_type=EventType.WINNER,
                        player_track_id=target_track_id,
                        metadata={"candidate": "unreturned_serve_or_winner"},
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

    def detect_net_approach(self, frame_idx: int, court_y: float, track_id: int) -> MatchEvent | None:
        if court_y < self.net_distance:
            ev = MatchEvent(
                frame_idx=frame_idx,
                event_type=EventType.NET_APPROACH,
                player_track_id=track_id,
            )
            self.events.append(ev)
            return ev
        return None

    def note_opponent_movement(self, frame_idx: int) -> None:
        self._opponent_moved_frames.add(frame_idx)

    def detect_serve(
        self,
        frame_idx: int,
        pose: PoseKeypoints | None,
        stroke_type: StrokeType,
        track_id: int,
    ) -> bool:
        is_serve_stroke = stroke_type in (StrokeType.FIRST_SERVE, StrokeType.SECOND_SERVE)
        arm_raised = False
        if pose and "right_wrist" in pose.keypoints and "right_shoulder" in pose.keypoints:
            wx, wy, _ = pose.keypoints["right_wrist"]
            sx, sy, _ = pose.keypoints["right_shoulder"]
            arm_raised = sy - wy > self.serve_toss_threshold * 0.3
        is_serve = is_serve_stroke or arm_raised
        if is_serve:
            self._serve_frames.append(frame_idx)
        return is_serve

    def detect_ace(
        self,
        frame_idx: int,
        is_serve: bool,
        opponent_moved: bool,
        in_service_box: bool,
        track_id: int,
    ) -> MatchEvent | None:
        if is_serve and not opponent_moved and in_service_box:
            ev = MatchEvent(
                frame_idx=frame_idx,
                event_type=EventType.ACE,
                player_track_id=track_id,
                metadata={"serve_frame": frame_idx},
            )
            self.events.append(ev)
            return ev
        return None

    def detect_double_fault_candidate(
        self,
        frame_idx: int,
        is_serve: bool,
        ball_out: bool,
        track_id: int,
    ) -> MatchEvent | None:
        recent_serves = [f for f in self._serve_frames if frame_idx - f < 150]
        if is_serve and ball_out and len(recent_serves) >= 2:
            ev = MatchEvent(
                frame_idx=frame_idx,
                event_type=EventType.DOUBLE_FAULT,
                player_track_id=track_id,
            )
            self.events.append(ev)
            return ev
        return None

    def detect_winner_candidate(
        self,
        frame_idx: int,
        ball_tracked_after_contact: bool,
        near_baseline: bool,
        track_id: int,
    ) -> MatchEvent | None:
        if near_baseline and not ball_tracked_after_contact:
            ev = MatchEvent(
                frame_idx=frame_idx,
                event_type=EventType.WINNER,
                player_track_id=track_id,
                metadata={"candidate": "untracked_after_contact"},
            )
            self.events.append(ev)
            return ev
        return None
