from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backend.utils.logging import get_logger
from backend.utils.types import MatchStats, ShotQuality

logger = get_logger(__name__)


class VisualizationEngine:
    """Phase 17: padel court heatmaps, shot maps, radar, timeline."""

    def __init__(self, config: dict):
        self.output_dir = Path(config["paths"]["data_reports"]) / "viz"
        self.court_length = config["court"]["court_length_m"]
        self.court_width = config["court"]["court_width_m"]

    def render_all(
        self,
        stats: MatchStats,
        shots: list[ShotQuality],
        shot_understanding: list[dict] | None = None,
        all_rallies: list | None = None,
    ) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rally_pool = all_rallies if all_rallies is not None else stats.rallies
        paths = {
            "heatmap": self._render_heatmap(stats),
            "radar": self._render_radar(stats),
            "shot_chart": self._render_shot_chart(shots, shot_understanding or []),
            "timeline": self._render_timeline(stats, rally_pool),
        }
        manifest = self.output_dir / "manifest.json"
        manifest.write_text(json.dumps(paths, indent=2), encoding="utf-8")
        return paths

    def _render_heatmap(self, stats: MatchStats) -> str:
        path = self.output_dir / "movement_heatmap.png"
        grid = np.array(stats.movement.heatmap or [[0]])
        fig, ax = plt.subplots(figsize=(5, 10))
        ax.imshow(grid, origin="lower", cmap="YlOrRd", aspect="auto")
        ax.set_title("Player Movement Heatmap (10×20 m)")
        ax.set_xlabel("Court width (m)")
        ax.set_ylabel("Court length (m)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def _render_radar(self, stats: MatchStats) -> str:
        path = self.output_dir / "performance_radar.png"
        labels = ["Movement", "Net", "Wall Def", "Position", "Consistency", "Aggression", "Shot Q"]
        s = stats.scores
        values = [s.movement, s.net_play, s.wall_defense, s.positioning, s.consistency, s.aggression, s.shot_quality]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_cycle = values + values[:1]
        angles_cycle = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles_cycle, values_cycle, "o-", linewidth=2, color="#1a73e8")
        ax.fill(angles_cycle, values_cycle, alpha=0.25, color="#1a73e8")
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 100)
        ax.set_title("Padel Performance Radar")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def _render_shot_chart(
        self,
        shots: list[ShotQuality],
        shot_understanding: list[dict],
    ) -> str:
        path = self.output_dir / "shot_chart.png"
        fig, ax = plt.subplots(figsize=(5, 10))
        ax.set_xlim(0, self.court_width)
        ax.set_ylim(0, self.court_length)
        ax.plot(
            [0, self.court_width, self.court_width, 0, 0],
            [0, 0, self.court_length, self.court_length, 0],
            "w-",
            linewidth=2,
        )
        ax.axhline(self.court_length / 2, color="white", linestyle="--", alpha=0.5, label="Net")
        ax.set_facecolor("#1e4d6b")
        plotted = 0

        for shot in shots:
            if shot.landing_xy:
                cx, cy = shot.landing_xy
                if shot.off_wall:
                    color = "cyan"
                elif shot.in_court:
                    color = "lime"
                else:
                    color = "red"
                ax.scatter(cx, cy, c=color, s=24, alpha=0.75)
                plotted += 1

        # Fallback: intelligence layer tracks player contact positions on court
        stroke_colors = {
            "forehand": "#4ade80",
            "backhand": "#60a5fa",
            "volley": "#fbbf24",
            "smash": "#f87171",
            "lob": "#c084fc",
            "salida": "#22d3ee",
            "drop_shot": "#fb923c",
        }
        for s in shot_understanding:
            pos = s.get("position")
            if not pos or len(pos) < 2:
                continue
            cx, cy = float(pos[0]), float(pos[1])
            if not (0 <= cx <= self.court_width and 0 <= cy <= self.court_length):
                continue
            stroke = (s.get("stroke") or "unknown").lower()
            if hasattr(stroke, "value"):
                stroke = stroke.value
            color = stroke_colors.get(str(stroke).lower(), "#ffffff")
            ax.scatter(cx, cy, c=color, s=28, alpha=0.85, marker="o", edgecolors="white", linewidths=0.3)
            plotted += 1

        ax.set_title(f"Shot Map ({plotted} contacts)")
        ax.set_xlabel("Court width (m)")
        ax.set_ylabel("Court length (m)")
        ax.set_aspect("equal")
        if plotted == 0:
            ax.text(
                self.court_width / 2,
                self.court_length / 2,
                "No court positions tracked",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def _render_timeline(self, stats: MatchStats, rallies: list | None = None) -> str:
        path = self.output_dir / "timeline.png"
        rally_list = rallies if rallies is not None else stats.rallies
        fig, ax = plt.subplots(figsize=(12, 3.5))
        fps = stats.fps or 25

        for i, rally in enumerate(sorted(rally_list, key=lambda r: r.start_frame)):
            t0 = rally.start_frame / fps
            t1 = rally.end_frame / fps
            ax.barh(
                0,
                t1 - t0,
                left=t0,
                height=0.45,
                color="#1a73e8",
                alpha=0.75,
                label="Point" if i == 0 else None,
            )
            ax.text(
                (t0 + t1) / 2,
                0,
                f"P{i + 1}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

        event_labels = set()
        for ev in stats.events:
            t = ev.frame_idx / fps
            et = ev.event_type.value
            if "wall" in et:
                color, label = "orange", "Wall event"
            elif "net_approach" in et:
                color, label = "coral", "Net approach"
            elif "error" in et or "winner" in et:
                color, label = "gold", "Point event"
            else:
                color, label = "gray", "Other"
            ax.axvline(t, color=color, alpha=0.35, linewidth=0.8, label=label if label not in event_labels else None)
            event_labels.add(label)

        ax.set_xlim(0, max(stats.duration_s, 1))
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Point Timeline ({len(rally_list)} points detected)")
        ax.set_yticks([])
        if event_labels:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)
