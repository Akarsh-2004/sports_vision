from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from backend.analytics.contact.ball_contact import BallContactDetector
from backend.analytics.contact.shot_gating import ShotDebouncer
from backend.analytics.events.active_play import ActivePlayGate
from backend.analytics.events.event_detector import EventDetector
from backend.analytics.events.point_segmentation import PointSegmentation
from backend.analytics.events.point_end_detector import PointEndDetector
from backend.analytics.events.serve_detector import ServeDetector
from backend.analytics.events.score_tracker import ScoreTracker, PointResult
from backend.analytics.events.rally_gating import players_in_ready_position, walking_to_baseline
from backend.analytics.events.wall_detector import WallDetector
from backend.analytics.movement.movement_analytics import MovementAnalytics
from backend.analytics.pose.pose_estimator import PoseEstimator
from backend.analytics.quality.shot_quality import ShotQualityEstimator
from backend.analytics.strokes.stroke_classifier import StrokeClassifier
from backend.analytics.tactical.tactical_engine import TacticalEngine
from backend.court.court_detector import CourtDetector
from backend.detection.ball_detector import BallDetector
from backend.detection.player_detector import PlayerDetector
from backend.highlights.highlight_generator import HighlightGenerator
from backend.ingestion.phone_preprocess import is_phone_footage
from backend.ingestion.video_ingestion import VideoIngestion
from backend.intelligence.confidence.propagation import ModuleConfidence
from backend.intelligence.epv.model import estimate_epv
from backend.intelligence.geometry.entities import StrokeObservation
from backend.intelligence.geometry.projector import GeometryProjector
from backend.intelligence.interaction.builder import InteractionBuilder
from backend.intelligence.interaction.graph import InteractionGraph
from backend.intelligence.pipeline import IntelligencePipeline
from backend.intelligence.shot.understanding import (
    ShotUnderstanding,
    infer_expected_outcome,
    infer_pressure,
    infer_risk,
)
from backend.intelligence.tactical.rules import evaluate_decision, infer_shot_intent
from backend.intelligence.court.semantic_regions import classify_region
from backend.intelligence.world.world_model import WorldModel
from backend.scoring.performance_scorer import PerformanceScorer
from backend.selection.team_selector import TeamSelector
from backend.summarization.report_generator import ReportGenerator
from backend.tracking.ball_tracker import BallTracker
from backend.tracking.ball_shot_detector import BallShotDetector
from backend.tracking.player_tracker import PlayerTracker
from backend.utils.audio_rally import detect_hit_frames
from backend.utils.config import load_config
from backend.utils.logging import get_logger
from backend.utils.speed import clamp_ball_speed_kmh
from backend.utils.types import MatchStats, RallySegment, StrokeEvent, StrokeType
from backend.visualization.charts import VisualizationEngine

logger = get_logger(__name__)

STAGE_LABELS = [
    "Ingesting video",
    "Court calibration",
    "Detecting players",
    "Tracking players",
    "Selecting target",
    "Tracking ball",
    "Wall events",
    "Estimating pose",
    "Classifying strokes",
    "Ball contact",
    "Movement analytics",
    "Shot quality",
    "Tactical engine",
    "Segmenting rallies",
    "Detecting events",
    "Generating highlights",
    "Scoring performance",
    "Writing report",
    "Rendering visualizations",
    "Finalizing output",
]


class PipelineOrchestrator:
    """Padel match analysis — OpenCV homography backbone + perception stack."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        sel_mode = self.config.get("selection", {}).get("mode", "single")
        self.ingestion = VideoIngestion(self.config)
        self.court = CourtDetector(self.config)
        self.player_detector = PlayerDetector(self.config)
        self.player_tracker = PlayerTracker(self.config)
        self.ball_detector = BallDetector(self.config)
        self.selector = TeamSelector(mode=sel_mode)
        self.pose = PoseEstimator(self.config)
        self.strokes = StrokeClassifier()
        self.contacts = BallContactDetector()
        self.movement = MovementAnalytics(self.config)
        self.quality = ShotQualityEstimator(self.config)
        self.walls = WallDetector(self.config)
        self.tactical = TacticalEngine(self.config)
        self.points = PointSegmentation(self.config)
        self.events = EventDetector(self.config)
        self.highlights = HighlightGenerator(self.config)
        self.scorer = PerformanceScorer()
        self.reporter = ReportGenerator(self.config)
        self.viz = VisualizationEngine(self.config)
        self.fps = self.config["pipeline"]["target_fps"]
        scfg = self.config.get("strokes", {})
        self.shot_debouncer = ShotDebouncer(
            self.fps,
            min_gap_s=scfg.get("min_shot_gap_s", 0.55),
            global_gap_s=scfg.get("global_shot_gap_s", 0.22),
        )
        self.contact_max_dist_px = scfg.get("contact_max_dist_px", 120.0)
        self.court_length = self.config["court"]["court_length_m"]
        self.court_width = self.config["court"]["court_width_m"]
        self.pose_on_contact_only = self.config["pipeline"].get("pose_on_ball_contact_only", True)
        self.use_intelligence = self.config.get("intelligence", {}).get("enabled", True)
        self.geometry = GeometryProjector(self.config)
        self.world = WorldModel(self.config)
        self.interactions = InteractionBuilder()
        self.shot_understandings: list[ShotUnderstanding] = []
        self.all_player_ids: set[int] = set()

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
        self.highlights.set_output_dir(reports_dir / "highlights")
        self._speed_cap = self.config.get("ball", {}).get("max_speed_kmh", 120.0)

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

        active_gate = ActivePlayGate(self.config, self.fps, audio_hits)

        ball_tracker = BallTracker(fps=self.fps)
        ball_shot_detector = BallShotDetector(fps=self.fps)
        player_court_positions: dict[int, dict[int, tuple[float, float]]] = {}
        ball_court_positions: dict[int, tuple[float, float]] = {}
        stroke_events: list[StrokeEvent] = []
        shot_frames: list[int] = []
        ball_trajectory_court: list[tuple[int, float, float]] = []
        ball_trajectory_px: list[tuple[int, float, float, float]] = []  # (frame, x_px, y_px, conf)
        ground_bounces: set[int] = set()
        max_ball_speed = 0.0
        ball_speeds_by_frame: dict[int, float] = {}
        target_id: int | None = None
        prev_court_pos = None
        prev_player_positions: dict[int, tuple[float, float]] = {}
        partner_history: dict[int, list[tuple[float, float]]] = {}
        last_players: list = []
        frame_idx = -1
        last_ball_visible = False
        last_wall_hit_frame = False
        # Per-frame player pixel boxes for annotated export: {frame: {track_id: (x1,y1,x2,y2,team)}}
        player_pixel_boxes: dict[int, dict[int, tuple]] = {}

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

            self.selector.assign_teams(players, self.court_width)

            if target_id is None and players:
                if target_click:
                    target_id = self.selector.select_by_click(target_click, players, frame)
                else:
                    target_id = self.selector.select_auto(players, self.court_length)
            elif players:
                target_id = self.selector.confirm_track(players, frame)

            ball_det = self.ball_detector.detect(frame)
            bx, by, bconf = ball_tracker.update(frame_idx, ball_det)
            ball_speed = clamp_ball_speed_kmh(ball_tracker.get_speed_kmh(), self._speed_cap)
            ball_speeds_by_frame[frame_idx] = ball_speed
            max_ball_speed = max(max_ball_speed, ball_speed)
            ball_bounce = self.ball_detector.detect_bounce(ball_speed)

            if frame_idx in audio_hits:
                ball_bounce = True
                bconf = max(bconf, 0.45)

            if ball_bounce:
                ground_bounces.add(frame_idx)

            if court_state.homography and bconf > 0.15:
                court_ball = self.court.pixel_to_court(bx, by, court_state)
                if court_ball:
                    ball_trajectory_court.append((frame_idx, court_ball[0], court_ball[1]))
                    ball_court_positions[frame_idx] = court_ball

            # Store pixel-space ball positions for annotated export
            if bconf > 0.10:
                ball_trajectory_px.append((frame_idx, bx, by, bconf))

            # Store player pixel boxes for annotated export (stride-aligned)
            if run_detection and players:
                player_pixel_boxes[frame_idx] = {
                    p.track_id: (
                        p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2,
                        getattr(p, 'team', p.track_id % 2),
                    )
                    for p in players
                }

            ready = players_in_ready_position(players, court_state, self.court_length)
            walking = walking_to_baseline(players, prev_player_positions, self.fps)

            wall_hit_this_frame = False
            wall_hit_type: str | None = None
            if len(ball_trajectory_court) >= 3:
                recent = ball_trajectory_court[-3:]
                wall_ev = self.walls.analyze_trajectory(recent, ground_bounces)
                wall_hit_this_frame = len(wall_ev) > 0 and wall_ev[-1].frame_idx == frame_idx
                if wall_hit_this_frame:
                    wall_hit_type = wall_ev[-1].hit_type.value

            active_play = active_gate.evaluate(
                frame_idx,
                players,
                bconf,
                ball_speed,
                court_state,
                frame.shape,
                prev_player_positions,
                self.court_length,
            )

            if active_play:
                frame_players: dict[int, tuple[float, float]] = {}
                for p in players:
                    if p.court_xy:
                        frame_players[p.track_id] = p.court_xy
                if frame_players:
                    player_court_positions[frame_idx] = frame_players

            self.points.add_frame_state(
                frame_idx, bconf, ball_speed, ready, ball_bounce, walking, wall_hit_this_frame, active_play
            )

            for tid in self.selector.target_ids():
                partner_history.setdefault(tid, [])
                p = next((x for x in players if x.track_id == tid), None)
                if active_play and p and p.court_xy:
                    partner_history[tid].append(p.court_xy)

            for p in players:
                if p.court_xy:
                    prev_player_positions[p.track_id] = p.court_xy

            target_player = next((p for p in players if p.track_id == target_id), None)

            hitter_player = None
            stroke_obs: StrokeObservation | None = None
            hitter_id: int | None = None

            if active_play and bconf > 0.2 and players:
                dists = []
                for p in players:
                    cx, cy = p.bbox.centroid
                    dists.append((float(np.hypot(cx - bx, cy - by)), p))
                dists.sort(key=lambda x: x[0])
                if dists and dists[0][0] < self.contact_max_dist_px:
                    hitter_player = dists[0][1]
                    pose = None
                    if not self.pose_on_contact_only or self._near_ball_contact(
                        hitter_player, bx, by, self.contact_max_dist_px
                    ):
                        pose = self.pose.estimate(
                            frame, frame_idx, hitter_player.track_id, hitter_player.bbox
                        )

                    contact = self.contacts.detect(
                        frame_idx,
                        players,
                        (bx, by),
                        ball_speed,
                        pose,
                        hitter_player.court_xy,
                    )

                    if contact and self.shot_debouncer.allow(contact.track_id, frame_idx):
                        hitter_id = contact.track_id
                        hp = next((x for x in players if x.track_id == hitter_id), hitter_player)
                        at_net = bool(hp.court_xy and self.court.is_at_net(*hp.court_xy))
                        at_back = bool(
                            hp.court_xy
                            and (
                                hp.court_xy[1] < 2.5
                                or hp.court_xy[1] > self.court_length - 2.5
                            )
                        )
                        stroke = self.strokes.classify(
                            frame_idx,
                            hitter_id,
                            pose,
                            ball_speed,
                            at_net=at_net,
                            at_back_wall=at_back,
                        )
                        if stroke.stroke_type.value != "unknown":
                            stroke_events.append(stroke)
                            shot_frames.append(frame_idx)
                            stroke_obs = StrokeObservation(
                                stroke.stroke_type, stroke.confidence, hitter_id
                            )
                            if hitter_id == target_id and stroke.stroke_type == StrokeType.SMASH:
                                sq = self.quality.estimate_shot(
                                    frame_idx,
                                    (bx, by),
                                    court_state,
                                    ball_speed,
                                    is_bounce=ball_bounce,
                                    off_wall=wall_hit_this_frame,
                                )
                                self.events.detect_smash_winner(
                                    frame_idx, stroke.stroke_type, sq.in_court, target_id or -1
                                )

            if target_player and active_play and target_player.court_xy:
                cx, cy = target_player.court_xy
                self.movement.add_position(frame_idx, cx, cy)
                prev_court_pos = target_player.court_xy
                self.events.detect_net_approach(frame_idx, cy, target_id or -1, self.court_length)

            if self.use_intelligence:
                geo = self.geometry.project_frame(
                    frame_idx,
                    players,
                    (bx, by),
                    bconf,
                    ball_speed,
                    ball_det.visible,
                    court_state,
                    active_play,
                )
                ball_entity = geo.ball
                physics_state = self.world.physics_engine.update(frame_idx, ball_entity, self.fps)

                for p in players:
                    self.all_player_ids.add(p.track_id)

                player_conf = min(0.99, 0.5 + len(players) * 0.12) if players else 0.3
                conf = ModuleConfidence(
                    ball=min(0.95, bconf + (physics_state.confidence if physics_state else 0) * 0.3),
                    players=player_conf,
                    court=court_state.confidence if court_state.valid_for_analytics else 0.3,
                    pose=0.75 if stroke_obs else 0.0,
                    stroke=stroke_obs.confidence if stroke_obs else 0.0,
                    wall_hit=0.65 if wall_hit_this_frame else 0.0,
                    physics=physics_state.confidence if physics_state else 0.0,
                )

                shot_understanding = None
                frame_interactions: list = []
                if stroke_obs and hitter_id is not None:
                    hp = next((p for p in geo.players if p.track_id == hitter_id), None)
                    if hp:
                        region = classify_region(hp.position[0], hp.position[1])
                        intent = infer_shot_intent(stroke_obs.stroke_type, geo, hitter_id)
                        at_net = hp.zone.value == "net"
                        epv_before, epv_after = estimate_epv(
                            hp.position[1],
                            stroke_obs.stroke_type,
                            intent,
                            ball_speed,
                            at_net,
                            opponents_deep=True,
                        )
                        dq, note = evaluate_decision(
                            stroke_obs.stroke_type, geo, hitter_id, ball_speed
                        )
                        shot_understanding = ShotUnderstanding(
                            frame_idx=frame_idx,
                            player_id=hitter_id,
                            stroke=stroke_obs.stroke_type,
                            intent=intent,
                            pressure=infer_pressure(hp.position[1], ball_speed, at_net),
                            risk=infer_risk(stroke_obs.stroke_type, region),
                            expected_outcome=infer_expected_outcome(stroke_obs.stroke_type, intent),
                            region=region,
                            position=hp.position,
                            speed_kmh=ball_speed,
                            confidence=conf.stroke,
                            decision_quality=dq,
                            decision_note=note,
                            epv_before=epv_before,
                            epv_after=epv_after,
                        )
                        self.shot_understandings.append(shot_understanding)

                if stroke_obs and hitter_id is not None:
                    wf_stub = type("W", (), {"geometry": geo, "active_play": active_play, "match_state": type("M", (), {"value": "rally"})()})()
                    frame_interactions = self.interactions.process_frame(
                        wf_stub, stroke_obs, wall_hit_type, ball_bounce, hitter_id
                    )

                self.world.update(
                    geo,
                    physics_state,
                    conf,
                    shot_understanding,
                    wall_hit_type,
                    ball_bounce,
                    frame_interactions,
                )

            last_ball_visible = ball_det.visible
            last_wall_hit_frame = wall_hit_this_frame

        _stage(14, 82.0)
        if len(ball_trajectory_court) >= 3:
            self.walls.analyze_trajectory(ball_trajectory_court, ground_bounces)

        active_segments = active_gate.build_segments()
        ball_shot_frames = ball_shot_detector.detect_from_trajectory(ball_tracker.trajectory)
        merged_shots = sorted(set(shot_frames) | set(ball_shot_frames))
        shot_frames = self.shot_debouncer.dedupe_frames(active_gate.filter_frames(merged_shots))
        active_frames = active_gate.active_frame_count()
        total_analyzed = frame_idx + 1

        movement_stats = self.movement.compute()
        ball_rallies = self.points.segment(shot_frames)
        raw_rallies = self.points.best_segments(
            shot_frames, ball_rallies, active_segments=active_segments
        )

        point_end_detector = PointEndDetector(self.config, self.court_length, self.court_width)
        serve_detector = ServeDetector(self.config, self.court_length)
        score_tracker = ScoreTracker()
        frame_index = point_end_detector.build_frame_index(
            self.points._frames,
            ball_court_positions,
            player_court_positions,
        )

        completed_rallies: list[RallySegment] = []
        for i, rally in enumerate(raw_rallies):
            dead_end = serve_detector.dead_gap_before(rally, active_segments)
            serve_frame = serve_detector.detect(rally, frame_index, dead_end)
            end_reason = point_end_detector.classify_end(rally, frame_index)
            winner = (
                PointEndDetector.infer_winner(rally, frame_index, self.court_length)
                if end_reason == "point_complete"
                else None
            )
            duration_s = (rally.end_frame - rally.start_frame) / self.fps
            score_tracker.add_point(
                PointResult(
                    rally_id=i,
                    start_frame=serve_frame or rally.start_frame,
                    end_frame=rally.end_frame,
                    serve_frame=serve_frame,
                    end_reason=end_reason,
                    winner_side=winner,
                    shot_count=rally.rally_length_shots,
                    duration_s=round(duration_s, 2),
                )
            )
            if end_reason == "point_complete":
                completed_rallies.append(
                    RallySegment(
                        start_frame=serve_frame or rally.start_frame,
                        end_frame=rally.end_frame,
                        rally_length_shots=rally.rally_length_shots,
                        wall_hits=rally.wall_hits,
                        excitement_score=getattr(rally, "excitement_score", 0.0),
                    )
                )

        all_rallies = completed_rallies if completed_rallies else raw_rallies
        score_summary = score_tracker.summary()

        logger.info(
            "Point segmentation: %d raw, %d complete (score %s)",
            len(raw_rallies),
            score_summary["points_complete"],
            score_summary["score"],
        )
        for r in all_rallies:
            rally_events = [e for e in self.events.events if r.start_frame <= e.frame_idx <= r.end_frame]
            r.excitement_score = float(
                min(
                    100.0,
                    self.highlights._excitement_raw(r, rally_events, max_ball_speed)
                    + r.wall_hits * 5,
                )
            )
        rallies = self.highlights.score_rallies(
            all_rallies, self.events.events, max_ball_speed, movement_stats.total_distance_m
        )

        _stage(15, 85.0)
        for rally in rallies:
            last_shot = self.quality.shots[-1] if self.quality.shots else None
            self.events.detect_from_rally(rally, last_shot, target_id or -1)

        team_stats = self.tactical.compute_team_stats(partner_history, self.selector.target_ids())
        tactical = self.tactical.compute_tactical(
            movement_stats, stroke_events, self.quality.shots, all_rallies, self.events.events
        )

        intelligence_output: dict = {}
        from collections import Counter

        shot_counts = Counter(s.player_id for s in self.shot_understandings)
        if shot_counts:
            best_id, best_n = shot_counts.most_common(1)[0]
            if best_n > 0 and (target_id is None or shot_counts.get(target_id, 0) == 0):
                logger.info(
                    "Retargeting analysis from player %s to player %s (%d shots)",
                    target_id,
                    best_id,
                    best_n,
                )
                target_id = best_id

        if self.use_intelligence and target_id is not None:
            intel = IntelligencePipeline(self.config, target_id)
            intelligence_output = intel.finalize(
                self.world,
                match_id,
                video_path,
                (frame_idx + 1) / self.fps,
                self.interactions.graph,
                self.shot_understandings,
                self.all_player_ids,
                analytics_rallies=all_rallies,
            )
            intel.close()

        _stage(17, 90.0)
        scores = self.scorer.score(movement_stats, stroke_events, self.quality.shots, self.events.events)
        stats = MatchStats(
            match_id=match_id,
            target_track_id=target_id or -1,
            selection_mode=self.config.get("selection", {}).get("mode", "single"),
            total_frames=frame_idx + 1,
            duration_s=(frame_idx + 1) / self.fps,
            fps=self.fps,
            movement=movement_stats,
            team=team_stats,
            tactical=tactical,
            scores=scores,
            rallies=rallies,
            events=self.events.events,
            strokes=stroke_events,
            wall_events=self.walls.events,
            shot_qualities=self.quality.shots,
            stroke_distribution=self.strokes.distribution(stroke_events),
        )
        _stage(18, 93.0)
        stats.summary = intelligence_output.get("coach_report") or self.reporter.generate(stats)
        _stage(19, 96.0)
        self.viz.output_dir = reports_dir / "viz"
        viz_paths = self.viz.render_all(
            stats,
            self.quality.shots,
            shot_understanding=intelligence_output.get("shot_understanding", []),
            all_rallies=all_rallies,
        )
        clip_paths: list[str] = []
        coach_highlights: dict = {}
        try:
            from backend.highlights.coach_highlights import CoachHighlightEngine

            coach_engine = CoachHighlightEngine(self.config, reports_dir)

            # Build broadcast-quality frame annotations for annotated clip export
            try:
                from backend.visualization.annotated_exporter import (
                    AnnotatedExporter, FrameAnnotation, PlayerAnnotation, build_frame_annotations
                )
                frame_anns = build_frame_annotations(
                    player_court_positions,
                    ball_court_positions,
                    ball_trajectory_px,
                    all_rallies,
                    self.fps,
                )
                # Enrich with pixel-space player boxes
                for fidx, boxes in player_pixel_boxes.items():
                    if fidx in frame_anns:
                        fa = frame_anns[fidx]
                        fa.players = [
                            PlayerAnnotation(
                                track_id=tid,
                                team=int(v[4]),
                                x1=v[0], y1=v[1], x2=v[2], y2=v[3],
                                court_x=(
                                    player_court_positions.get(fidx, {}).get(tid, (None, None))[0]
                                ),
                                court_y=(
                                    player_court_positions.get(fidx, {}).get(tid, (None, None))[1]
                                ),
                                label=f"P{tid}",
                            )
                            for tid, v in boxes.items()
                        ]
                coach_engine.frame_annotations = frame_anns
            except Exception as fa_exc:
                logger.warning("Frame annotation build failed (non-fatal): %s", fa_exc)

            coach_highlights = coach_engine.generate(
                playable_path,
                all_rallies,
                rallies,
                self.events.events,
                intelligence_output,
                max_ball_speed,
                target_id,
                ball_speeds_by_frame=ball_speeds_by_frame,
            )
            clip_paths = coach_highlights.get("paths", [])
            if coach_highlights.get("error"):
                logger.warning("Coach highlights error: %s", coach_highlights["error"])
            if not clip_paths:
                logger.warning(
                    "No highlight clips extracted (%d rallies, coach enabled=%s)",
                    len(all_rallies),
                    self.config.get("coach_highlights", {}).get("enabled", True),
                )
                clip_paths = self.highlights.extract_clips(playable_path, rallies)
        except FileNotFoundError as exc:
            logger.warning("Highlight extraction skipped: %s", exc)
        except Exception as exc:
            logger.warning("Coach highlights failed, falling back to legacy: %s", exc, exc_info=True)
            try:
                clip_paths = self.highlights.extract_clips(playable_path, rallies)
            except FileNotFoundError:
                clip_paths = []

        output = {
            "stats": stats.to_dict(),
            "active_play": {
                "enabled": self.config.get("active_play", {}).get("enabled", True),
                "active_frames": active_frames,
                "total_frames": total_analyzed,
                "active_ratio": round(active_frames / max(total_analyzed, 1), 3),
                "segments": active_segments,
            },
            "intelligence": intelligence_output,
            "rallies_all": [
                {
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "rally_length_shots": r.rally_length_shots,
                    "wall_hits": r.wall_hits,
                    "excitement_score": r.excitement_score,
                }
                for r in all_rallies
            ],
            "rallies_raw": [
                {
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "rally_length_shots": r.rally_length_shots,
                }
                for r in raw_rallies
            ],
            "score": score_summary,
            "ball_shot_frames": ball_shot_frames,
            "shot_frames": shot_frames,
            "summary_md": stats.summary,
            "visualizations": viz_paths,
            "highlights": clip_paths,
            "coach_highlights": coach_highlights,
        }
        (reports_dir / "match_stats.json").write_text(json.dumps(output["stats"], indent=2), encoding="utf-8")
        (reports_dir / "report.md").write_text(stats.summary, encoding="utf-8")
        if intelligence_output.get("reports"):
            (reports_dir / "player_report.md").write_text(
                intelligence_output["reports"].get("player", ""), encoding="utf-8"
            )
            (reports_dir / "training_report.md").write_text(
                intelligence_output["reports"].get("training", ""), encoding="utf-8"
            )
            per_player = intelligence_output["reports"].get("per_player", {})
            if per_player:
                players_dir = reports_dir / "players"
                players_dir.mkdir(parents=True, exist_ok=True)
                index_lines = ["# Player Reports", ""]
                for pid, text in sorted(per_player.items(), key=lambda x: int(x[0])):
                    path = players_dir / f"player_{pid}.md"
                    path.write_text(text, encoding="utf-8")
                    index_lines.append(f"- [Player {pid}](player_{pid}.md)")
                (players_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        (reports_dir / "full_output.json").write_text(
            json.dumps({**output, "source_video": video_path, "playable_video": playable_path, "analysis_fps": self.fps}, indent=2),
            encoding="utf-8",
        )

        if self.use_intelligence:
            try:
                import subprocess

                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve().parents[2] / "scripts" / "generate_dashboard.py"),
                        "--match-dir",
                        str(reports_dir),
                        "--video",
                        video_path,
                    ],
                    check=False,
                    cwd=str(Path(__file__).resolve().parents[2]),
                )
            except Exception as exc:
                logger.warning("Dashboard generation skipped: %s", exc)

        self.pose.close()
        _stage(20, 100.0)
        logger.info(
            "Padel pipeline complete: %s (%d/%d rallies above excitement threshold)",
            match_id,
            len(rallies),
            len(all_rallies),
        )
        return stats

    def _near_ball_contact(self, player, bx: float, by: float, max_dist_px: float = 120.0) -> bool:
        cx, cy = player.bbox.centroid
        return float(np.hypot(cx - bx, cy - by)) < max_dist_px
