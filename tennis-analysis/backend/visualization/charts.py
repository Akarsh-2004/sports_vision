from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backend.utils.logging import get_logger
from backend.utils.types import MatchStats, ShotQuality

logger = get_logger(__name__)


class VisualizationEngine:
    """Stage 17: heatmaps, shot charts, radar, timeline exports."""

    def __init__(self, config: dict):
        self.output_dir = Path(config["paths"]["data_reports"]) / "viz"
        self.court_length = config["court"]["court_length_m"]
        self.singles_width = config["court"]["singles_width_m"]

    def render_all(self, stats: MatchStats, shots: list[ShotQuality]) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "heatmap": self._render_heatmap(stats),
            "radar": self._render_radar(stats),
            "shot_chart": self._render_shot_chart(shots),
            "timeline": self._render_timeline(stats),
        }
        manifest = self.output_dir / "manifest.json"
        manifest.write_text(json.dumps(paths, indent=2), encoding="utf-8")
        return paths

    def _render_heatmap(self, stats: MatchStats) -> str:
        path = self.output_dir / "movement_heatmap.png"
        grid = np.array(stats.movement.heatmap or [[0]])
        fig, ax = plt.subplots(figsize=(4, 8))
        ax.imshow(grid, origin="lower", cmap="YlOrRd", aspect="auto")
        ax.set_title("Movement Heatmap")
        ax.set_xlabel("Court width")
        ax.set_ylabel("Court length")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def _render_radar(self, stats: MatchStats) -> str:
        path = self.output_dir / "performance_radar.png"
        labels = ["Serve", "Return", "Movement", "Consistency", "Aggression", "Stamina", "Coverage"]
        s = stats.scores
        values = [s.serve, s.return_score, s.movement, s.consistency, s.aggression, s.stamina, s.court_coverage]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_cycle = values + values[:1]
        angles_cycle = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles_cycle, values_cycle, "o-", linewidth=2)
        ax.fill(angles_cycle, values_cycle, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 100)
        ax.set_title("Performance Radar")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def _render_shot_chart(self, shots: list[ShotQuality]) -> str:
        path = self.output_dir / "shot_chart.png"
        fig, ax = plt.subplots(figsize=(4, 8))
        ax.set_xlim(0, self.singles_width)
        ax.set_ylim(0, self.court_length)
        ax.plot([0, self.singles_width, self.singles_width, 0, 0], [0, 0, self.court_length, self.court_length, 0], "w-", linewidth=2)
        ax.set_facecolor("#2d5a27")
        for shot in shots:
            if shot.landing_xy:
                cx, cy = shot.landing_xy
                color = "lime" if shot.in_court else "red"
                ax.scatter(cx, cy, c=color, s=20, alpha=0.7)
        ax.set_title("Shot Landing Chart")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def _render_timeline(self, stats: MatchStats) -> str:
        path = self.output_dir / "timeline.png"
        fig, ax = plt.subplots(figsize=(10, 3))
        for rally in stats.rallies:
            t0 = rally.start_frame / stats.fps
            t1 = rally.end_frame / stats.fps
            ax.barh(0, t1 - t0, left=t0, height=0.4, color="steelblue", alpha=0.7)
        for ev in stats.events:
            t = ev.frame_idx / stats.fps
            ax.axvline(t, color="orange", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_title("Match Timeline (rallies & events)")
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)
