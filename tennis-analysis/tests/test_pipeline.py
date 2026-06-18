import numpy as np
import pytest

from backend.analytics.events.point_segmentation import PointSegmentation
from backend.analytics.events.rally_gating import players_in_ready_position, walking_to_baseline
from backend.court.court_detector import CourtDetector
from backend.highlights.highlight_generator import HighlightGenerator
from backend.highlights.interval_merge import HighlightInterval, merge_overlapping_intervals
from backend.tracking.ball_tracker import BallTracker
from backend.utils.config import load_config
from backend.utils.types import BallDetection, BBox, PlayerDetection, RallySegment
from backend.scoring.performance_scorer import PerformanceScorer
from backend.analytics.movement.movement_analytics import MovementAnalytics


@pytest.fixture
def config():
    return load_config()


def test_court_is_in_bounds(config):
    cd = CourtDetector(config)
    assert cd.is_in_court(4.0, 12.0)
    assert not cd.is_in_court(-1, 12.0)
    assert not cd.is_in_court(4.0, 30.0)


def test_court_service_box(config):
    cd = CourtDetector(config)
    assert cd.is_in_service_box(4.0, 3.0)
    assert not cd.is_in_service_box(4.0, 12.0)


def test_ball_tracker_kalman():
    bt = BallTracker(fps=25.0)
    det = BallDetection(x=100, y=200, confidence=0.9, visible=True)
    x, y, c = bt.update(0, det)
    assert c > 0
    assert abs(x - 100) < 5
    miss = BallDetection(x=0, y=0, confidence=0, visible=False)
    for i in range(1, 4):
        x2, y2, c2 = bt.update(i, miss)
    assert c2 > 0


def test_movement_analytics(config):
    ma = MovementAnalytics(config)
    for i in range(50):
        ma.add_position(i, float(i) * 0.1, float(i) * 0.2)
    stats = ma.compute()
    assert stats.total_distance_m > 0
    assert len(stats.heatmap) == 20


def test_performance_scorer():
    from backend.utils.types import MovementStats

    scorer = PerformanceScorer()
    scores = scorer.score(MovementStats(total_distance_m=1500), [], [], [])
    assert 0 <= scores.overall <= 100


def test_bbox_centroid():
    b = BBox(10, 20, 30, 60)
    assert b.centroid == (20.0, 40.0)
    assert b.area == 800.0


def test_interval_merge():
    fps = 25.0
    intervals = [
        HighlightInterval(0, 250, 60.0),
        HighlightInterval(100, 350, 80.0),
        HighlightInterval(500, 700, 50.0),
    ]
    merged = merge_overlapping_intervals(intervals, fps, min_overlap_s=2.0)
    assert len(merged) == 2
    assert merged[0].excitement == 80.0
    assert merged[0].end_frame == 350


def test_point_segmentation_gating(config):
    ps = PointSegmentation(config)
    fps = config["pipeline"]["target_fps"]
    # 5 seconds of rally signal with bounces
    for i in range(int(5 * fps)):
        ps.add_frame_state(i, ball_conf=0.8, ball_speed=10.0, players_ready=True, ball_bounce=(i % 20 == 0))
    rallies = ps.segment(shot_frames=[50, 75, 100])
    assert len(rallies) >= 1
    assert rallies[0].rally_length_shots >= 3


def test_point_segmentation_rejects_warmup(config):
    ps = PointSegmentation(config)
    fps = config["pipeline"]["target_fps"]
    for i in range(int(2 * fps)):
        ps.add_frame_state(i, ball_conf=0.0, ball_speed=0.0, players_ready=False, walking_to_baseline=True)
    rallies = ps.segment()
    assert len(rallies) == 0


def test_players_ready():
    p1 = PlayerDetection(1, BBox(10, 400, 50, 600), court_xy=(2.0, 2.0))
    p2 = PlayerDetection(2, BBox(200, 100, 250, 300), court_xy=(6.0, 20.0))
    assert players_in_ready_position([p1, p2])


def test_highlight_excitement_threshold(config):
    hg = HighlightGenerator(config)
    rallies = [
        RallySegment(0, 100, rally_length_shots=2),
        RallySegment(200, 400, rally_length_shots=12),
    ]
    for r in rallies:
        r.excitement_score = hg._excitement_raw(r, [], 80.0)
    scored = hg.score_rallies(rallies, [], max_ball_speed=80, max_distance=200)
    assert len(scored) >= 1
    assert scored[0].rally_length_shots == 12
