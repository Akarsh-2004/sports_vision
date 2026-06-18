from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from backend.analytics.events.event_detector import EventDetector
from backend.analytics.events.point_segmentation import PointSegmentation
from backend.analytics.events.rally_gating import players_in_ready_position, walking_to_baseline
from backend.analytics.movement.movement_analytics import MovementAnalytics
from backend.analytics.pose.pose_estimator import PoseEstimator
from backend.analytics.quality.shot_quality import ShotQualityEstimator
from backend.analytics.strokes.stroke_classifier import StrokeClassifier
from backend.court.court_detector import CourtDetector
from backend.detection.ball_detector import BallDetector
from backend.detection.player_detector import PlayerDetector
from backend.highlights.highlight_generator import HighlightGenerator
from backend.ingestion.phone_preprocess import is_phone_footage
from backend.ingestion.video_ingestion import VideoIngestion
from backend.scoring.performance_scorer import PerformanceScorer
from backend.selection.target_selector import TargetSelector
from backend.summarization.report_generator import ReportGenerator
from backend.tracking.ball_tracker import BallTracker
from backend.tracking.player_tracker import PlayerTracker
from backend.utils.audio_rally import detect_hit_frames
from backend.utils.config import load_config
from backend.utils.logging import get_logger
from backend.utils.types import MatchStats, StrokeEvent, StrokeType
from backend.visualization.charts import VisualizationEngine

logger = get_logger(__name__)

STAGE_LABELS = [
    "Ingesting video",
    "Detecting court",
    "Detecting players",
    "Tracking players",
    "Selecting target",
    "Tracking ball",
    "Estimating pose",
    "Classifying strokes",
    "Movement analytics",
    "Shot quality",
    "Segmenting points",
    "Detecting events",
    "Generating highlights",
    "Scoring performance",
    "Writing report",
    "Rendering visualizations",
    "Finalizing output",
]


class PipelineOrchestrator:
    """Runs all 17 pipeline stages end-to-end."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.ingestion = VideoIngestion(self.config)
        self.court = CourtDetector(self.config)
        self.player_detector = PlayerDetector(self.config)
        self.player_tracker = PlayerTracker(self.config)
        self.ball_detector = BallDetector(self.config)
        self.selector = TargetSelector()
        self.pose = PoseEstimator(self.config)
        self.strokes = StrokeClassifier()
        self.movement = MovementAnalytics(self.config)
        self.quality = ShotQualityEstimator(self.config)
        self.points = PointSegmentation(self.config)
        self.events = EventDetector(self.config)
        self.highlights = HighlightGenerator(self.config)
        self.scorer = PerformanceScorer()
        self.reporter = ReportGenerator(self.config)
        self.viz = VisualizationEngine(self.config)
        self.fps = self.config["pipeline"]["target_fps"]
        self.court_length = self.config["court"]["court_length_m"]
        self.pose_on_contact_only = self.config["pipeline"].get("pose_on_ball_contact_only", True)

    def run(
        self,
        video_path: str,
        target_click: tuple[float, float] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        stage_callback: Callable[[int, str, float], None] | None = None,
        max_frames: int | None = None,
    ) -> MatchStats:
        def _stage(n: int, pct: float) -> None:
            if stage_callback:
                label = STAGE_LABELS[min(n - 1, len(STAGE_LABELS) - 1)]
                stage_callback(n, label, pct)

        match_id = Path(video_path).stem + "_" + uuid.uuid4().hex[:8]
        reports_dir = Path(self.config["paths"]["data_reports"]) / match_id
        reports_dir.mkdir(parents=True, exist_ok=True)

        self.config["pipeline"]["output_dir"] = str(Path(self.config["paths"]["data_processed"]) / match_id)
        _stage(1, 0.0)
        playable_path, frame_meta = self.ingestion.ingest(video_path)
        self.court.set_video_source(video_path)

        if self.config.get("phone", {}).get("auto_enable", True) and frame_meta:
            m0 = frame_meta[0]
            if is_phone_footage(m0.width, m0.height):
                self.config.setdefault("phone", {})["enabled"] = True
                self.player_detector.phone_preprocess = True
                logger.info("Phone footage detected; enabling preprocessing")

        estimated_frames = self.ingestion._count_output_frames(playable_path)
        total_frames = estimated_frames if max_frames is None else min(estimated_frames, max_frames)
        detection_stride = self.config["pipeline"].get("detection_stride", 4)

        audio_hits: set[int] = set()
        if self.config.get("audio", {}).get("rally_hints", True):
            audio_hits = set(detect_hit_frames(playable_path, self.fps))

        ball_tracker = BallTracker(fps=self.fps)
        stroke_events: list[StrokeEvent] = []
        shot_frames: list[int] = []
        max_ball_speed = 0.0
        target_id: int | None = None
        prev_court_pos = None
        prev_player_positions: dict[int, tuple[float, float]] = {}
        last_players: list = []
        frame_idx = -1
        last_ball_visible = False

        _stage(2, 5.0)
        for frame_idx, frame in self.ingestion.frame_generator(playable_path):
            if max_frames is not None and frame_idx >= max_frames:
                break

            if progress_callback and frame_idx % 25 == 0:
                pct = 5.0 + 75.0 * (frame_idx / max(total_frames, 1))
                progress_callback(frame_idx, total_frames, "processing")
                _stage(6, pct)

            run_detection = frame_idx % detection_stride == 0
            court_state = self.court.detect(frame, frame_idx)

            if run_detection:
                players = self.player_tracker.update(frame)
                if not players:
                    dets = self.player_detector.detect(frame, court_state)
                    players = self.player_tracker.update(frame, dets)
                last_players = players
            else:
                players = last_players

            for p in players:
                if court_state.homography:
                    p.court_xy = self.court.pixel_to_court(*p.bbox.centroid, court_state)

            if target_id is None and players:
                if target_click:
                    target_id = self.selector.select_by_click(target_click, players, frame)
                else:
                    target_id = self.selector.select_auto_far_baseline(players)
            elif players:
                target_id = self.selector.confirm_track(players, frame)

            ball_det = self.ball_detector.detect(frame)
            bx, by, bconf = ball_tracker.update(frame_idx, ball_det)
            ball_speed = ball_tracker.get_speed_kmh()
            max_ball_speed = max(max_ball_speed, ball_speed)
            ball_bounce = self.ball_detector.detect_bounce(ball_speed)

            if frame_idx in audio_hits:
                ball_bounce = True
                bconf = max(bconf, 0.45)

            ready = players_in_ready_position(players, court_state, self.court_length)
            walking = walking_to_baseline(players, prev_player_positions, self.fps)
            self.points.add_frame_state(
                frame_idx, bconf, ball_speed, ready, ball_bounce, walking
            )

            for p in players:
                if p.court_xy:
                    prev_player_positions[p.track_id] = p.court_xy

            player_speed = 0.0
            target_player = next((p for p in players if p.track_id == target_id), None)
            opponent = next((p for p in players if p.track_id != target_id), None)
            if opponent and opponent.court_xy and frame_idx > 0:
                opp_prev = prev_player_positions.get(opponent.track_id)
                if opp_prev:
                    dist = float(np.hypot(opponent.court_xy[0] - opp_prev[0], opponent.court_xy[1] - opp_prev[1]))
                    if dist > 0.15:
                        self.events.note_opponent_movement(frame_idx)

            if target_player and target_player.court_xy:
                cx, cy = target_player.court_xy
                self.movement.add_position(frame_idx, cx, cy)
                if prev_court_pos:
                    dt = 1.0 / self.fps
                    player_speed = float(np.hypot(cx - prev_court_pos[0], cy - prev_court_pos[1]) / dt)
                prev_court_pos = target_player.court_xy
                self.events.detect_net_approach(frame_idx, cy, target_id or -1)

            near_contact = bconf > 0.25 and target_player is not None
            if target_player and near_contact:
                if not self.pose_on_contact_only or self._near_ball_contact(target_player, bx, by):
                    pose = self.pose.estimate(frame, frame_idx, target_player.track_id, target_player.bbox)
                else:
                    pose = None
                at_baseline = target_player.court_xy and target_player.court_xy[1] > self.court_length * 0.7
                stroke = self.strokes.classify(
                    frame_idx, target_player.track_id, pose, ball_speed, at_baseline=bool(at_baseline)
                )
                if stroke.stroke_type.value != "unknown":
                    stroke_events.append(stroke)
                    shot_frames.append(frame_idx)
                    is_serve = self.events.detect_serve(
                        frame_idx, pose, stroke.stroke_type, target_id or -1
                    )
                    if is_serve and target_player.court_xy:
                        in_box = self.court.is_in_service_box(*target_player.court_xy[:2])
                        opp_moved = frame_idx in self.events._opponent_moved_frames
                        self.events.detect_ace(frame_idx, True, opp_moved, in_box, target_id or -1)
                sq = self.quality.estimate_shot(frame_idx, (bx, by), court_state, ball_speed)
                if stroke.stroke_type in (StrokeType.FIRST_SERVE, StrokeType.SECOND_SERVE) and not sq.in_court:
                    self.events.detect_double_fault_candidate(
                        frame_idx, True, True, target_id or -1
                    )
                if not last_ball_visible and not ball_det.visible and at_baseline:
                    self.events.detect_winner_candidate(
                        frame_idx, False, True, target_id or -1
                    )
            last_ball_visible = ball_det.visible

        _stage(11, 82.0)
        movement_stats = self.movement.compute()
        rallies = self.points.segment(shot_frames)
        all_rallies = list(rallies)
        for r in all_rallies:
            rally_events = [e for e in self.events.events if r.start_frame <= e.frame_idx <= r.end_frame]
            r.excitement_score = float(
                min(100.0, self.highlights._excitement_raw(r, rally_events, max_ball_speed))
            )
        rallies = self.highlights.score_rallies(
            all_rallies, self.events.events, max_ball_speed, movement_stats.total_distance_m
        )

        _stage(12, 85.0)
        for rally in rallies:
            last_shot = self.quality.shots[-1] if self.quality.shots else None
            self.events.detect_from_rally(rally, last_shot, target_id or -1)

        _stage(14, 90.0)
        scores = self.scorer.score(movement_stats, stroke_events, self.quality.shots, self.events.events)
        stats = MatchStats(
            match_id=match_id,
            target_track_id=target_id or -1,
            total_frames=frame_idx + 1,
            duration_s=(frame_idx + 1) / self.fps,
            fps=self.fps,
            movement=movement_stats,
            scores=scores,
            rallies=rallies,
            events=self.events.events,
            strokes=stroke_events,
            shot_qualities=self.quality.shots,
            stroke_distribution=self.strokes.distribution(stroke_events),
        )
        _stage(15, 93.0)
        stats.summary = self.reporter.generate(stats)
        _stage(16, 96.0)
        viz_paths = self.viz.render_all(stats, self.quality.shots)
        try:
            clip_paths = self.highlights.extract_clips(playable_path, rallies)
        except FileNotFoundError as exc:
            logger.warning("Highlight extraction skipped: %s", exc)
            clip_paths = []

        output = {
            "stats": stats.to_dict(),
            "rallies_all": [
                {
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "rally_length_shots": r.rally_length_shots,
                    "excitement_score": r.excitement_score,
                }
                for r in all_rallies
            ],
            "summary_md": stats.summary,
            "visualizations": viz_paths,
            "highlights": clip_paths,
        }
        (reports_dir / "match_stats.json").write_text(json.dumps(output["stats"], indent=2), encoding="utf-8")
        (reports_dir / "report.md").write_text(stats.summary, encoding="utf-8")
        (reports_dir / "full_output.json").write_text(
            json.dumps({**output, "source_video": video_path, "playable_video": playable_path}, indent=2),
            encoding="utf-8",
        )

        self.pose.close()
        _stage(17, 100.0)
        logger.info(
            "Pipeline complete: %s (%d/%d rallies above excitement threshold)",
            match_id, len(rallies), len(all_rallies),
        )
        return stats

    def _near_ball_contact(self, player, bx: float, by: float, max_dist_px: float = 120.0) -> bool:
        cx, cy = player.bbox.centroid
        return float(np.hypot(cx - bx, cy - by)) < max_dist_px
