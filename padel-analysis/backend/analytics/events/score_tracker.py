"""Point-level score state from completed rallies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from backend.utils.types import RallySegment


@dataclass
class PointResult:
    rally_id: int
    start_frame: int
    end_frame: int
    serve_frame: int | None
    end_reason: str  # point_complete | interrupted | ambiguous
    winner_side: str | None  # A | B | None
    shot_count: int
    duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreTracker:
    points: list[PointResult] = field(default_factory=list)
    score: dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0})

    def add_point(self, result: PointResult) -> None:
        if result.end_reason == "point_complete" and result.winner_side in ("A", "B"):
            self.score[result.winner_side] += 1
        self.points.append(result)

    def completed_points(self) -> list[PointResult]:
        return [p for p in self.points if p.end_reason == "point_complete"]

    def completed_rallies(self, raw_rallies: list[RallySegment]) -> list[RallySegment]:
        """Map completed PointResults back to RallySegment objects for highlights."""
        out: list[RallySegment] = []
        for i, pr in enumerate(self.completed_points()):
            if i < len(raw_rallies):
                r = raw_rallies[i]
                out.append(
                    RallySegment(
                        start_frame=pr.start_frame,
                        end_frame=pr.end_frame,
                        rally_length_shots=pr.shot_count,
                        wall_hits=getattr(r, "wall_hits", 0),
                        excitement_score=getattr(r, "excitement_score", 0.0),
                    )
                )
            else:
                out.append(
                    RallySegment(
                        start_frame=pr.start_frame,
                        end_frame=pr.end_frame,
                        rally_length_shots=pr.shot_count,
                    )
                )
        return out

    def get_scoreline(self, frame_idx: int) -> str:
        a = b = 0
        for p in self.points:
            if p.end_frame > frame_idx:
                break
            if p.end_reason == "point_complete" and p.winner_side == "A":
                a += 1
            elif p.end_reason == "point_complete" and p.winner_side == "B":
                b += 1
        return f"{a}-{b}"

    def summary(self) -> dict:
        completed = self.completed_points()
        total_shots = sum(p.shot_count for p in completed)
        avg_dur = (
            sum(p.duration_s for p in completed) / len(completed) if completed else 0.0
        )
        return {
            "score": dict(self.score),
            "points_total": len(self.points),
            "points_complete": len(completed),
            "points_interrupted": sum(1 for p in self.points if p.end_reason == "interrupted"),
            "points_ambiguous": sum(1 for p in self.points if p.end_reason == "ambiguous"),
            "total_shots": total_shots,
            "avg_point_duration_s": round(avg_dur, 2),
            "points": [p.to_dict() for p in self.points],
        }
