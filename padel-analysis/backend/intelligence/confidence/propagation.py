"""Confidence propagation across all perception modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModuleConfidence:
    ball: float = 0.0
    players: float = 0.0
    court: float = 0.0
    pose: float = 0.0
    stroke: float = 0.0
    wall_hit: float = 0.0
    physics: float = 0.0

    @property
    def overall(self) -> float:
        weights = [0.25, 0.2, 0.15, 0.1, 0.15, 0.1, 0.05]
        vals = [self.ball, self.players, self.court, self.pose, self.stroke, self.wall_hit, self.physics]
        return sum(w * v for w, v in zip(weights, vals))

    def to_dict(self) -> dict:
        return {
            "ball": round(self.ball, 3),
            "players": round(self.players, 3),
            "court": round(self.court, 3),
            "pose": round(self.pose, 3),
            "stroke": round(self.stroke, 3),
            "wall_hit": round(self.wall_hit, 3),
            "physics": round(self.physics, 3),
            "overall": round(self.overall, 3),
        }


@dataclass
class ConfidenceTracker:
    """Aggregate confidence across match for self-evaluation."""

    samples: list[ModuleConfidence] = field(default_factory=list)

    def record(self, conf: ModuleConfidence) -> None:
        self.samples.append(conf)

    def match_average(self) -> ModuleConfidence:
        if not self.samples:
            return ModuleConfidence()
        n = len(self.samples)
        return ModuleConfidence(
            ball=sum(s.ball for s in self.samples) / n,
            players=sum(s.players for s in self.samples) / n,
            court=sum(s.court for s in self.samples) / n,
            pose=sum(s.pose for s in self.samples) / n,
            stroke=sum(s.stroke for s in self.samples) / n,
            wall_hit=sum(s.wall_hit for s in self.samples) / n,
            physics=sum(s.physics for s in self.samples) / n,
        )

    def reliability_note(self) -> str:
        avg = self.match_average()
        low = []
        if avg.stroke < 0.7:
            low.append("stroke recognition")
        if avg.ball < 0.6:
            low.append("ball tracking")
        if avg.court < 0.5:
            low.append("court calibration")
        if not low:
            return "All modules above reliability threshold — insights are high-confidence."
        return f"Lower confidence in: {', '.join(low)}. Related insights should be interpreted cautiously."
