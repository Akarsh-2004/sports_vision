"""AI Coach highlight engine — indexed, categorized, explainable coaching artifacts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from backend.utils.ffmpeg import get_ffmpeg
from backend.utils.logging import get_logger
from backend.utils.types import EventType, MatchEvent, RallySegment

logger = get_logger(__name__)


class HighlightLevel(str, Enum):
    SHOT = "shot"
    RALLY = "rally"
    TACTICAL = "tactical"
    STORY = "story"


# folder slug → display label
CATEGORY_LABELS: dict[str, str] = {
    "best_rallies": "Best Rallies",
    "best_smashes": "Best Smashes",
    "best_defense": "Best Defense",
    "best_viboras": "Best Viboras",
    "best_volleys": "Best Volleys",
    "longest_rally": "Longest Rally",
    "fastest_point": "Fastest Point",
    "smartest_point": "Smartest Point",
    "biggest_mistake": "Biggest Mistake",
    "wall_play": "Wall Play",
    "net_battle": "Net Battle",
    "coaching_moments": "Coaching Moments",
    "top_moments": "Top 10 Moments",
}


@dataclass
class CoachingHighlight:
    event_id: int
    level: str
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    time_s: float
    rally_length: int = 0
    difficulty: float = 0.0
    excitement: float = 0.0
    categories: list[str] = field(default_factory=list)
    primary_category: str = "top_moments"
    stroke: str = ""
    winner: str = ""
    commentary: str = ""
    chain: str = ""
    clip_path: str = ""
    overlay: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.75
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def _stroke_category(stroke: str) -> str | None:
    s = (stroke or "").lower()
    if s in ("smash",):
        return "best_smashes"
    if s in ("vibora", "bandeja"):
        return "best_viboras"
    if "volley" in s:
        return "best_volleys"
    if s in ("lob", "salida"):
        return "best_defense"
    if s == "drop_shot":
        return "best_defense"
    return None


class CoachHighlightEngine:
    """Build ranked, categorized highlight objects and extract per-category clips."""

    def __init__(self, config: dict, match_dir: Path):
        self.config = config
        hcfg = config.get("coach_highlights", {})
        legacy = config.get("highlights", {})
        self.fps = config["pipeline"]["target_fps"]
        self.shot_preroll = int(hcfg.get("shot_preroll_s", 3) * self.fps)
        self.shot_postroll = int(hcfg.get("shot_postroll_s", 3) * self.fps)
        self.rally_preroll = int(hcfg.get("rally_preroll_s", 2) * self.fps)
        self.rally_postroll = int(hcfg.get("rally_postroll_s", 3) * self.fps)
        self.max_per_category = hcfg.get("max_per_category", 8)
        self.top_moments = hcfg.get("top_moments", 10)
        self.enabled = hcfg.get("enabled", True)
        self.min_excitement = legacy.get("min_excitement", 40.0)
        self.output_dir = match_dir / "highlights"
        self._next_id = 1

    def generate(
        self,
        video_path: str,
        all_rallies: list[RallySegment],
        scored_rallies: list[RallySegment],
        events: list[MatchEvent],
        intelligence: dict[str, Any],
        max_ball_speed: float,
        target_id: int | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"manifest": [], "by_category": {}, "paths": []}

        shots = intelligence.get("shot_understanding", [])
        rally_graphs = intelligence.get("rally_graphs", [])
        recommendations = intelligence.get("recommendations", [])
        patterns = intelligence.get("pattern_mining", {})
        timeline = intelligence.get("timeline_events", [])

        highlights: list[CoachingHighlight] = []
        highlights.extend(self._from_rallies(all_rallies, scored_rallies, events, rally_graphs, max_ball_speed))
        highlights.extend(self._from_shots(shots, rally_graphs, target_id))
        highlights.extend(self._from_events(events, all_rallies))
        highlights.extend(self._from_recommendations(recommendations))
        highlights.extend(self._from_patterns(patterns, shots, rally_graphs))
        highlights.extend(self._story_highlights(highlights, scored_rallies))

        # Deduplicate overlapping same-level clips in same category (keep higher excitement)
        highlights = self._dedupe(highlights)
        highlights.sort(key=lambda h: h.excitement, reverse=True)

        # Assign top_moments
        for i, h in enumerate(highlights[: self.top_moments]):
            if "top_moments" not in h.categories:
                h.categories.append("top_moments")
            if i == 0 and h.level == HighlightLevel.RALLY.value and "longest_rally" not in h.categories:
                h.categories.append("longest_rally")

        by_category: dict[str, list[CoachingHighlight]] = {k: [] for k in CATEGORY_LABELS}
        for h in highlights:
            for cat in h.categories:
                if cat in by_category and len(by_category[cat]) < self.max_per_category:
                    by_category[cat].append(h)

        paths: list[str] = []
        if video_path and Path(video_path).exists():
            paths = self._extract_clips(video_path, by_category)

        manifest = [h.to_dict() for h in highlights]
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "categories": CATEGORY_LABELS,
                    "events": manifest,
                    "by_category": {k: [h.event_id for h in v] for k, v in by_category.items() if v},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "manifest": manifest,
            "by_category": {k: [h.to_dict() for h in v] for k, v in by_category.items() if v},
            "paths": paths,
            "manifest_path": str(manifest_path),
        }

    def _new_id(self) -> int:
        eid = self._next_id
        self._next_id += 1
        return eid

    def _from_rallies(
        self,
        all_rallies: list[RallySegment],
        scored: list[RallySegment],
        events: list[MatchEvent],
        rally_graphs: list[dict],
        max_ball_speed: float,
    ) -> list[CoachingHighlight]:
        out: list[CoachingHighlight] = []
        graph_by_range = {(g["start_frame"], g["end_frame"]): g for g in rally_graphs}

        pool = scored if scored else sorted(all_rallies, key=lambda r: r.excitement_score, reverse=True)[:5]
        if not pool and rally_graphs:
            for g in rally_graphs:
                pool.append(
                    RallySegment(
                        start_frame=g["start_frame"],
                        end_frame=g["end_frame"],
                        rally_length_shots=len([n for n in g.get("nodes", []) if n.get("type") == "player_hit"]),
                        excitement_score=60.0,
                    )
                )

        for r in pool:
            rally_events = [e for e in events if r.start_frame <= e.frame_idx <= r.end_frame]
            graph = graph_by_range.get((r.start_frame, r.end_frame))
            if not graph:
                for g in rally_graphs:
                    if g["start_frame"] <= r.end_frame and g["end_frame"] >= r.start_frame:
                        graph = g
                        break

            chain = graph.get("chain", "") if graph else ""
            nodes = graph.get("nodes", []) if graph else []
            wall_count = sum(1 for n in nodes if n.get("type") in ("wall_hit", "ball_ground"))
            net_count = sum(1 for e in rally_events if e.event_type == EventType.NET_APPROACH)

            categories = ["best_rallies"]
            tags: list[str] = []
            if r.rally_length_shots >= 10:
                tags.append("Long Rally")
                categories.append("longest_rally")
            if wall_count >= 2 or any(e.event_type == EventType.WALL_EXCHANGE for e in rally_events):
                categories.append("wall_play")
                tags.append("Wall Play")
            if net_count >= 2:
                categories.append("net_battle")
                tags.append("Net Battle")
            if any(e.event_type in (EventType.SMASH_WINNER, EventType.WINNER) for e in rally_events):
                tags.append("Winner")

            difficulty = min(100.0, r.rally_length_shots * 4 + wall_count * 8 + net_count * 5)
            excitement = float(r.excitement_score or min(100.0, difficulty + max_ball_speed / 5))

            start_f = max(0, r.start_frame - self.rally_preroll)
            end_f = r.end_frame + self.rally_postroll
            start_s = start_f / self.fps

            out.append(
                CoachingHighlight(
                    event_id=self._new_id(),
                    level=HighlightLevel.RALLY.value,
                    start_frame=start_f,
                    end_frame=end_f,
                    start_time=_fmt_time(start_s),
                    end_time=_fmt_time(end_f / self.fps),
                    time_s=round(start_s, 2),
                    rally_length=r.rally_length_shots,
                    difficulty=round(difficulty, 1),
                    excitement=round(excitement, 1),
                    categories=categories,
                    primary_category="best_rallies",
                    chain=chain,
                    commentary=self._commentary_rally(r, chain, tags, wall_count, net_count),
                    overlay={
                        "pressure": "high" if net_count >= 2 else "medium",
                        "win_probability": min(0.95, 0.4 + excitement / 200),
                        "ball_speed_kmh": round(max_ball_speed, 1),
                    },
                    confidence=0.82,
                    tags=tags,
                )
            )
        return out

    def _from_shots(
        self,
        shots: list[dict],
        rally_graphs: list[dict],
        target_id: int | None,
    ) -> list[CoachingHighlight]:
        out: list[CoachingHighlight] = []
        for s in shots:
            stroke = s.get("stroke", "")
            cat = _stroke_category(stroke)
            if not cat and s.get("intent") not in ("finishing", "attack"):
                if s.get("intent") != "defensive" or stroke not in ("lob", "salida", "drop_shot"):
                    continue

            frame = int(s.get("frame", 0))
            speed = float(s.get("speed_kmh", 0))
            epv = float(s.get("epv_delta", 0))
            decision = s.get("decision_quality", s.get("decision", "neutral"))
            pressure = s.get("pressure", "medium")

            categories: list[str] = []
            if cat:
                categories.append(cat)
            if decision in ("poor", "suboptimal"):
                categories.append("biggest_mistake")
            if epv >= 0.15 and decision in ("good", "excellent", "neutral"):
                categories.append("smartest_point")
            if speed >= 100:
                categories.append("fastest_point")
            if s.get("intent") == "defensive":
                categories.append("best_defense")

            if not categories:
                continue

            start_f = max(0, frame - self.shot_preroll)
            end_f = frame + self.shot_postroll
            excitement = min(100.0, 40 + speed / 3 + epv * 100 + (20 if decision == "poor" else 0))

            out.append(
                CoachingHighlight(
                    event_id=self._new_id(),
                    level=HighlightLevel.SHOT.value,
                    start_frame=start_f,
                    end_frame=end_f,
                    start_time=_fmt_time(start_f / self.fps),
                    end_time=_fmt_time(end_f / self.fps),
                    time_s=round(frame / self.fps, 2),
                    difficulty=round(min(100, speed / 1.5), 1),
                    excitement=round(excitement, 1),
                    categories=categories,
                    primary_category=categories[0],
                    stroke=stroke,
                    commentary=self._commentary_shot(s, rally_graphs),
                    overlay={
                        "ball_speed_kmh": round(speed, 1),
                        "pressure": pressure,
                        "win_probability": round(0.5 + epv, 2),
                        "decision": decision,
                        "region": s.get("region", ""),
                    },
                    confidence=float(s.get("confidence", 0.7)),
                    tags=[stroke.replace("_", " ").title(), s.get("intent", "")],
                )
            )
        return out

    def _from_events(self, events: list[MatchEvent], rallies: list[RallySegment]) -> list[CoachingHighlight]:
        out: list[CoachingHighlight] = []
        for e in events:
            cat = None
            if e.event_type == EventType.SMASH_WINNER:
                cat = "best_smashes"
            elif e.event_type == EventType.WALL_WINNER:
                cat = "wall_play"
            elif e.event_type == EventType.NET_APPROACH:
                cat = "net_battle"
            elif e.event_type in (EventType.UNFORCED_ERROR, EventType.FORCED_ERROR):
                cat = "biggest_mistake"
            if not cat:
                continue

            frame = e.frame_idx
            start_f = max(0, frame - self.shot_preroll)
            end_f = frame + self.shot_postroll
            rally_len = 0
            for r in rallies:
                if r.start_frame <= frame <= r.end_frame:
                    rally_len = r.rally_length_shots
                    break

            out.append(
                CoachingHighlight(
                    event_id=self._new_id(),
                    level=HighlightLevel.SHOT.value,
                    start_frame=start_f,
                    end_frame=end_f,
                    start_time=_fmt_time(start_f / self.fps),
                    end_time=_fmt_time(end_f / self.fps),
                    time_s=round(frame / self.fps, 2),
                    rally_length=rally_len,
                    excitement=75.0,
                    categories=[cat],
                    primary_category=cat,
                    commentary=f"Match event: {e.event_type.value.replace('_', ' ')} at {_fmt_time(frame / self.fps)}.",
                    confidence=0.7,
                    tags=[e.event_type.value],
                )
            )
        return out

    def _from_recommendations(self, recommendations: list[dict]) -> list[CoachingHighlight]:
        out: list[CoachingHighlight] = []
        for rec in recommendations:
            for clip in rec.get("clips", [])[:3]:
                frame = int(clip.get("frame", 0))
                if frame <= 0:
                    continue
                start_f = max(0, frame - self.shot_preroll)
                end_f = frame + self.shot_postroll
                out.append(
                    CoachingHighlight(
                        event_id=self._new_id(),
                        level=HighlightLevel.TACTICAL.value,
                        start_frame=start_f,
                        end_frame=end_f,
                        start_time=_fmt_time(start_f / self.fps),
                        end_time=_fmt_time(end_f / self.fps),
                        time_s=round(frame / self.fps, 2),
                        excitement=65.0,
                        categories=["coaching_moments"],
                        primary_category="coaching_moments",
                        commentary=f"Coaching focus: {rec.get('advice', '')} Evidence: {rec.get('evidence', '')}",
                        confidence=float(rec.get("confidence", 0.7)),
                        tags=["Coaching"],
                    )
                )
        return out

    def _from_patterns(
        self,
        patterns: dict,
        shots: list[dict],
        rally_graphs: list[dict],
    ) -> list[CoachingHighlight]:
        out: list[CoachingHighlight] = []
        sequences = patterns.get("sequences", patterns.get("top_sequences", []))
        if isinstance(sequences, dict):
            sequences = list(sequences.values())

        for seq in sequences[:5]:
            if isinstance(seq, str):
                label = seq
                frame = shots[0]["frame"] if shots else 0
            else:
                label = seq.get("sequence", seq.get("pattern", ""))
                frame = int(seq.get("frame", shots[0]["frame"] if shots else 0))

            if not label or frame <= 0:
                continue

            start_f = max(0, frame - self.rally_preroll)
            end_f = frame + self.rally_postroll + int(4 * self.fps)
            out.append(
                CoachingHighlight(
                    event_id=self._new_id(),
                    level=HighlightLevel.TACTICAL.value,
                    start_frame=start_f,
                    end_frame=end_f,
                    start_time=_fmt_time(start_f / self.fps),
                    end_time=_fmt_time(end_f / self.fps),
                    time_s=round(frame / self.fps, 2),
                    excitement=70.0,
                    categories=["smartest_point", "coaching_moments"],
                    primary_category="smartest_point",
                    chain=label,
                    commentary=f"Recurring tactical pattern: {label}. This sequence appeared multiple times in the match.",
                    confidence=0.68,
                    tags=["Tactical Sequence"],
                )
            )
        return out

    def _story_highlights(
        self,
        existing: list[CoachingHighlight],
        scored_rallies: list[RallySegment],
    ) -> list[CoachingHighlight]:
        if not existing:
            return []
        best = max(existing, key=lambda h: h.excitement)
        out: list[CoachingHighlight] = []
        if scored_rallies:
            r = scored_rallies[0]
            start_f = max(0, r.start_frame - self.rally_preroll)
            end_f = r.end_frame + self.rally_postroll
            out.append(
                CoachingHighlight(
                    event_id=self._new_id(),
                    level=HighlightLevel.STORY.value,
                    start_frame=start_f,
                    end_frame=end_f,
                    start_time=_fmt_time(start_f / self.fps),
                    end_time=_fmt_time(end_f / self.fps),
                    time_s=round(start_f / self.fps, 2),
                    rally_length=r.rally_length_shots,
                    excitement=best.excitement,
                    categories=["top_moments"],
                    primary_category="top_moments",
                    commentary=(
                        "Match story highlight: the highest-excitement passage combines rally length, "
                        f"shot variety, and tactical pressure (peak excitement {best.excitement:.0f}/100)."
                    ),
                    confidence=0.75,
                    tags=["Story", "Top Moment"],
                )
            )
        return out

    def _commentary_rally(
        self,
        rally: RallySegment,
        chain: str,
        tags: list[str],
        wall_count: int,
        net_count: int,
    ) -> str:
        parts: list[str] = []
        if rally.rally_length_shots:
            parts.append(f"This rally lasted {rally.rally_length_shots} shots.")
        if wall_count >= 2:
            parts.append(
                f"The turning point involved {wall_count} wall interactions — classic padel tempo before the finish."
            )
        if net_count >= 2:
            parts.append("Both teams traded at the net, increasing pressure through volleys and quick reactions.")
        if chain and chain != "empty":
            parts.append(f"Shot sequence: {chain}.")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}.")
        return " ".join(parts) if parts else "Full rally replay for tactical review."

    def _commentary_shot(self, shot: dict, rally_graphs: list[dict]) -> str:
        stroke = shot.get("stroke", "shot").replace("_", " ")
        intent = shot.get("intent", "neutral")
        region = shot.get("region", "court").replace("_", " ")
        speed = shot.get("speed_kmh", 0)
        decision = shot.get("decision_quality", shot.get("decision", "neutral"))
        parts = [
            f"{stroke.title()} from the {region} with {intent} intent at {speed:.0f} km/h.",
        ]
        if decision in ("good", "excellent"):
            parts.append("The decision quality was strong given player positions and pressure.")
        elif decision in ("poor", "suboptimal"):
            parts.append("This was a suboptimal choice — review positioning before this shot.")
        epv = shot.get("epv_delta", 0)
        if epv > 0.1:
            parts.append(f"Expected point value increased by {epv*100:.0f}% after this shot.")
        return " ".join(parts)

    def _dedupe(self, highlights: list[CoachingHighlight]) -> list[CoachingHighlight]:
        seen: list[CoachingHighlight] = []
        for h in sorted(highlights, key=lambda x: x.excitement, reverse=True):
            overlap = False
            for prev in seen:
                if prev.level != h.level:
                    continue
                if prev.start_frame <= h.end_frame and h.start_frame <= prev.end_frame:
                    if set(prev.categories) & set(h.categories):
                        overlap = True
                        break
            if not overlap:
                seen.append(h)
        return seen

    def _extract_clips(self, video_path: str, by_category: dict[str, list[CoachingHighlight]]) -> list[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        extracted: set[tuple[int, int]] = set()

        for cat, items in by_category.items():
            cat_dir = self.output_dir / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            for i, h in enumerate(items):
                key = (h.start_frame, h.end_frame)
                rel_name = f"{cat}/{cat}_{i:02d}.mp4"
                out = self.output_dir / rel_name
                if key not in extracted:
                    if self._ffmpeg_cut(video_path, out, h.start_frame, h.end_frame):
                        extracted.add(key)
                        paths.append(str(out))
                h.clip_path = rel_name.replace("\\", "/")

        return paths

    def _ffmpeg_cut(self, video_path: str, out: Path, start_f: int, end_f: int) -> bool:
        start_t = start_f / self.fps
        duration = max(0.1, (end_f - start_f) / self.fps)
        cmd = [
            get_ffmpeg(),
            "-y",
            "-i",
            video_path,
            "-ss",
            str(start_t),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Clip extraction failed %s: %s", out.name, exc)
            return False
