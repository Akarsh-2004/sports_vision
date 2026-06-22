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
from backend.utils.speed import peak_speed_in_range
from backend.utils.types import EventType, MatchEvent, RallySegment

logger = get_logger(__name__)

# Lazy import to avoid circular deps
def _get_annotated_exporter():
    try:
        from backend.visualization.annotated_exporter import AnnotatedExporter, FrameAnnotation
        return AnnotatedExporter, FrameAnnotation
    except Exception:
        return None, None


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
        return None
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
        self.max_rally_highlight_s = hcfg.get("max_rally_highlight_s", 35.0)
        self.max_rally_highlight_frames = int(self.max_rally_highlight_s * self.fps)
        self.max_per_category = hcfg.get("max_per_category", 8)
        self.category_caps: dict[str, int] = hcfg.get("category_caps", {})
        self.top_moments = hcfg.get("top_moments", 10)
        self.enabled = hcfg.get("enabled", True)
        self.min_excitement = legacy.get("min_excitement", 40.0)
        self.output_dir = match_dir / "highlights"
        self._next_id = 1
        # Annotated export
        self.annotate = hcfg.get("annotate_clips", True)
        self._ann_suffix = hcfg.get("annotation_output_suffix", "_annotated")
        # Frame data set by orchestrator before calling generate()
        self.frame_annotations: dict = {}
        self.ball_speeds_by_frame: dict[int, float] = {}

    def generate(
        self,
        video_path: str,
        all_rallies: list[RallySegment],
        scored_rallies: list[RallySegment],
        events: list[MatchEvent],
        intelligence: dict[str, Any],
        max_ball_speed: float,
        target_id: int | None,
        ball_speeds_by_frame: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"manifest": [], "by_category": {}, "paths": [], "events": []}
        self.ball_speeds_by_frame = ball_speeds_by_frame or {}
        try:
            return self._generate_internal(
                video_path,
                all_rallies,
                scored_rallies,
                events,
                intelligence,
                max_ball_speed,
                target_id,
            )
        except Exception as exc:
            logger.error("CoachHighlightEngine failed: %s", exc, exc_info=True)
            return {
                "manifest": [],
                "by_category": {},
                "paths": [],
                "events": [],
                "error": str(exc),
            }

    def _generate_internal(
        self,
        video_path: str,
        all_rallies: list[RallySegment],
        scored_rallies: list[RallySegment],
        events: list[MatchEvent],
        intelligence: dict[str, Any],
        max_ball_speed: float,
        target_id: int | None,
    ) -> dict[str, Any]:

        shots = intelligence.get("shot_understanding", [])
        rally_graphs = intelligence.get("rally_graphs", [])
        recommendations = intelligence.get("recommendations", [])
        patterns = intelligence.get("pattern_mining", {})
        timeline = intelligence.get("timeline_events", [])

        highlights: list[CoachingHighlight] = []
        highlights.extend(self._from_rallies(all_rallies, scored_rallies, events, rally_graphs, max_ball_speed))
        highlights.extend(self._from_shots(shots, rally_graphs, target_id, all_rallies))
        highlights.extend(self._from_events(events, all_rallies))
        highlights.extend(self._from_recommendations(recommendations))
        highlights.extend(self._from_patterns(patterns, shots, rally_graphs))
        highlights.extend(self._story_highlights(highlights, scored_rallies))

        highlights = self._filter_quality(highlights, all_rallies, scored_rallies)
        highlights = self._dedupe(highlights)
        highlights = self._cap_defense_per_point(highlights, all_rallies)
        highlights = self._assign_longest_rally(highlights)
        highlights.sort(key=lambda h: h.excitement, reverse=True)

        # Assign top_moments
        for i, h in enumerate(highlights[: self.top_moments]):
            if "top_moments" not in h.categories:
                h.categories.append("top_moments")

        by_category: dict[str, list[CoachingHighlight]] = {k: [] for k in CATEGORY_LABELS}
        for h in highlights:
            for cat in h.categories:
                if cat not in by_category:
                    continue
                cap = self.category_caps.get(cat, self.max_per_category)
                if len(by_category[cat]) < cap:
                    by_category[cat].append(h)

        paths: list[str] = []
        if video_path and Path(video_path).exists():
            paths = self._extract_clips(video_path, by_category)
            # Generate annotated versions of all top clips
            if self.annotate and self.frame_annotations:
                annotated_paths = self._annotate_clips(video_path, highlights[:self.top_moments])
                paths.extend(annotated_paths)

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

        pool = scored if scored else sorted(all_rallies, key=lambda r: r.excitement_score, reverse=True)
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
            core_duration_s = (r.end_frame - r.start_frame) / self.fps
            if core_duration_s > self.max_rally_highlight_s:
                logger.warning(
                    "Skipping mega-rally highlight (%.1fs > %.1fs cap)",
                    core_duration_s,
                    self.max_rally_highlight_s,
                )
                continue

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
            if wall_count >= 2 or any(e.event_type == EventType.WALL_EXCHANGE for e in rally_events):
                categories.append("wall_play")
                tags.append("Wall Play")
            if net_count >= 2:
                categories.append("net_battle")
                tags.append("Net Battle")
            if any(e.event_type in (EventType.SMASH_WINNER, EventType.WINNER) for e in rally_events):
                tags.append("Winner")

            difficulty = min(100.0, r.rally_length_shots * 4 + wall_count * 8 + net_count * 5)

            # Improved excitement scoring:
            # Base: existing excitement score (already factors in rally length + speed)
            # Bonus: wall/glass play is KEY in padel (unique to the sport)
            # Bonus: net dominance battles are crowd pleasers
            # Bonus: peak ball speed in the rally (not just max over whole match)
            has_winner = any(e.event_type in (EventType.SMASH_WINNER, EventType.WINNER) for e in rally_events)
            glass_hit_count = sum(
                1 for n in nodes
                if n.get("type") == "wall_hit" and "glass" in str(n.get("hit_type", "")).lower()
            )
            rally_peak_speed = peak_speed_in_range(
                self.ball_speeds_by_frame, r.start_frame, r.end_frame
            )
            if rally_peak_speed <= 0:
                rally_peak_speed = max_ball_speed * 0.5
            base_exc = float(r.excitement_score or 40.0)
            wall_bonus = min(30.0, wall_count * 7.0)
            glass_bonus = min(15.0, glass_hit_count * 8.0)
            net_bonus = min(15.0, net_count * 5.0)
            winner_bonus = 20.0 if has_winner else 0.0
            speed_bonus = min(15.0, rally_peak_speed / 12.0)
            length_bonus = min(10.0, max(0.0, (r.rally_length_shots - 5) * 1.5))
            excitement = min(100.0, base_exc + wall_bonus + glass_bonus + net_bonus + winner_bonus + speed_bonus + length_bonus)

            start_f = max(0, r.start_frame - self.rally_preroll)
            end_f = r.end_frame + self.rally_postroll
            start_s = start_f / self.fps
            duration_s = (r.end_frame - r.start_frame) / self.fps
            point_idx = self._point_index(all_rallies, r)

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
                    commentary=self._commentary_rally(
                        r, chain, tags, wall_count, net_count, point_idx, duration_s
                    ),
                    overlay={
                        "pressure": "high" if net_count >= 2 else "medium",
                        "win_probability": min(0.95, 0.4 + excitement / 200),
                        "ball_speed_kmh": round(rally_peak_speed, 1),
                        "point_number": point_idx,
                    },
                    confidence=0.82,
                    tags=tags + [f"Point {point_idx}"],
                )
            )
        return out

    @staticmethod
    def _point_index(rallies: list[RallySegment], rally: RallySegment) -> int:
        ordered = sorted(rallies, key=lambda r: r.start_frame)
        for i, r in enumerate(ordered, start=1):
            if r.start_frame == rally.start_frame and r.end_frame == rally.end_frame:
                return i
        return 1

    def _from_shots(
        self,
        shots: list[dict],
        rally_graphs: list[dict],
        target_id: int | None,
        all_rallies: list[RallySegment],
    ) -> list[CoachingHighlight]:
        out: list[CoachingHighlight] = []
        for s in shots:
            stroke = s.get("stroke", "")
            intent = s.get("intent", "neutral")
            cat = _stroke_category(stroke)
            if stroke == "drop_shot" and intent in ("recovery", "defensive"):
                cat = "best_defense"
            if not cat and intent not in ("finishing", "attack"):
                if intent != "defensive" or stroke not in ("lob", "salida"):
                    continue

            frame = int(s.get("frame", 0))
            if not any(r.start_frame <= frame <= r.end_frame for r in all_rallies):
                continue
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
        """Only emit highlights for decisive match events — not raw net approach spam."""
        out: list[CoachingHighlight] = []
        for e in events:
            cat = None
            if e.event_type == EventType.SMASH_WINNER:
                cat = "best_smashes"
            elif e.event_type == EventType.WALL_WINNER:
                cat = "wall_play"
            elif e.event_type in (EventType.UNFORCED_ERROR, EventType.FORCED_ERROR):
                cat = "biggest_mistake"
            elif e.event_type == EventType.WINNER:
                cat = "top_moments"
            elif e.event_type == EventType.WALL_EXCHANGE:
                cat = "wall_play"
            # LONG_RALLY / NET_APPROACH — stats only, no standalone clip (use full point replay)
            if not cat:
                continue

            frame = e.frame_idx
            start_f = max(0, frame - self.shot_preroll)
            end_f = frame + self.shot_postroll
            rally_len = 0
            matched_rally = None
            for r in rallies:
                if r.start_frame <= frame <= r.end_frame:
                    rally_len = r.rally_length_shots
                    matched_rally = r
                    break
            if not matched_rally:
                continue

            label = e.event_type.value.replace("_", " ")
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
                    excitement=78.0,
                    categories=[cat if cat in CATEGORY_LABELS else "top_moments"],
                    primary_category=cat if cat in CATEGORY_LABELS else "top_moments",
                    commentary=(
                        f"Point {self._point_index(rallies, matched_rally)} — "
                        f"{label} at {_fmt_time(frame / self.fps)} "
                        f"({rally_len} shots in rally)."
                    ),
                    confidence=0.75,
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
        point_idx: int = 1,
        duration_s: float = 0.0,
    ) -> str:
        parts: list[str] = [f"Point {point_idx}"]
        if rally.rally_length_shots:
            dur = duration_s or (rally.end_frame - rally.start_frame) / self.fps
            parts.append(f"— {rally.rally_length_shots}-shot rally lasting {dur:.1f}s.")
        if wall_count >= 2:
            parts.append(
                f"Wall play featured {wall_count} interactions before the finish."
            )
        if net_count >= 2:
            parts.append("Both pairs traded at the net under pressure.")
        if chain and chain != "empty":
            parts.append(f"Sequence: {chain}.")
        elif tags:
            parts.append(f"Highlights: {', '.join(tags)}.")
        return " ".join(parts) if len(parts) > 1 else f"Point {point_idx} — full rally replay for tactical review."

    def _filter_quality(
        self,
        highlights: list[CoachingHighlight],
        all_rallies: list[RallySegment],
        scored_rallies: list[RallySegment],
    ) -> list[CoachingHighlight]:
        """Drop vague, unanchored highlights that aren't tied to detected points."""
        if not all_rallies:
            return highlights

        rally_pool = scored_rallies or all_rallies
        kept: list[CoachingHighlight] = []
        for h in highlights:
            if h.level == HighlightLevel.RALLY.value:
                kept.append(h)
                continue
            if h.level == HighlightLevel.STORY.value:
                kept.append(h)
                continue
            if h.level == HighlightLevel.TACTICAL.value and h.categories == ["coaching_moments"]:
                kept.append(h)
                continue

            in_point = any(r.start_frame <= h.start_frame <= r.end_frame + self.rally_postroll for r in rally_pool)
            if not in_point:
                continue
            if h.excitement < self.min_excitement and h.level == HighlightLevel.SHOT.value:
                continue
            if "net_battle" in h.categories and h.level == HighlightLevel.SHOT.value:
                continue
            if "longest_rally" in h.categories and h.level != HighlightLevel.RALLY.value:
                continue
            kept.append(h)
        return kept

    def _assign_longest_rally(self, highlights: list[CoachingHighlight]) -> list[CoachingHighlight]:
        """Tag exactly one full point replay as longest_rally — never a 6s shot snippet."""
        for h in highlights:
            if "longest_rally" in h.categories:
                h.categories = [c for c in h.categories if c != "longest_rally"]

        rally_hls = [h for h in highlights if h.level == HighlightLevel.RALLY.value]
        if not rally_hls:
            return highlights

        longest = max(
            rally_hls,
            key=lambda h: (h.rally_length, h.end_frame - h.start_frame),
        )
        longest.categories.append("longest_rally")
        return highlights

    def _cap_defense_per_point(
        self,
        highlights: list[CoachingHighlight],
        all_rallies: list[RallySegment],
    ) -> list[CoachingHighlight]:
        """Keep salida/lob always; max one drop_shot defense clip per point."""
        defense = [
            h for h in highlights
            if "best_defense" in h.categories and h.level == HighlightLevel.SHOT.value
        ]
        if not defense:
            return highlights

        others = [h for h in highlights if h not in defense]
        by_point: dict[int, list[CoachingHighlight]] = {}
        for h in defense:
            pt = 0
            for r in all_rallies:
                if r.start_frame <= h.start_frame <= r.end_frame:
                    pt = self._point_index(all_rallies, r)
                    break
            by_point.setdefault(pt, []).append(h)

        kept: list[CoachingHighlight] = []
        for items in by_point.values():
            wall = [h for h in items if h.stroke in ("salida", "lob")]
            drops = [h for h in items if h.stroke == "drop_shot"]
            kept.extend(wall)
            if drops:
                kept.append(max(drops, key=lambda h: (h.excitement, h.overlay.get("ball_speed_kmh", 0))))

        return others + kept

    def _commentary_shot(self, shot: dict, rally_graphs: list[dict]) -> str:
        stroke = shot.get("stroke", "shot").replace("_", " ")
        intent = shot.get("intent", "neutral")
        region = shot.get("region", "court").replace("_", " ")
        speed = shot.get("speed_kmh", 0)
        decision = shot.get("decision_quality", shot.get("decision", "neutral"))
        parts = [
            f"At {_fmt_time(shot.get('frame', 0) / self.fps)} — "
            f"{stroke.title()} from the {region} ({intent} intent, {speed:.0f} km/h).",
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
                # Keep one highlight per point even when preroll/postroll overlaps adjacent points
                if h.level == HighlightLevel.RALLY.value:
                    p_prev = prev.overlay.get("point_number")
                    p_h = h.overlay.get("point_number")
                    if p_prev and p_h and p_prev != p_h:
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

    def _annotate_clips(
        self,
        video_path: str,
        highlights: list[CoachingHighlight],
    ) -> list[str]:
        """
        Generate broadcast-style annotated versions of top highlight clips.
        Each annotated clip has:
          - ball trajectory trail
          - player bounding boxes + team labels
          - mini court diagram
          - speed meter + excitement bar
        """
        AnnotatedExporter, _ = _get_annotated_exporter()
        if AnnotatedExporter is None:
            logger.warning("AnnotatedExporter unavailable; skipping annotated clips")
            return []

        ann_dir = self.output_dir / "annotated"
        ann_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        try:
            exporter = AnnotatedExporter(self.config, fps=self.fps)
        except Exception as exc:
            logger.warning("Failed to init AnnotatedExporter: %s", exc)
            return []

        seen: set[tuple[int, int]] = set()
        for i, h in enumerate(highlights):
            key = (h.start_frame, h.end_frame)
            if key in seen:
                continue
            seen.add(key)

            out_name = f"top_{i:02d}_{h.primary_category}{self._ann_suffix}.mp4"
            out_path = ann_dir / out_name

            # Build per-frame annotation slice for this clip
            clip_frames = {
                f: ann
                for f, ann in self.frame_annotations.items()
                if h.start_frame <= f <= h.end_frame
            }

            # Update excitement level in annotations for this specific highlight
            for ann in clip_frames.values():
                ann.excitement = h.excitement
                ann.rally_length = h.rally_length
                ann.state = h.primary_category.replace("_", " ").upper()

            try:
                result = exporter.export(
                    video_path=video_path,
                    out_path=out_path,
                    frame_data=clip_frames,
                    highlight=h,
                    start_frame=h.start_frame,
                    end_frame=h.end_frame,
                )
                if result:
                    h.clip_path = str(Path(result).relative_to(self.output_dir.parent))
                    paths.append(result)
                    logger.info("Annotated clip → %s", out_name)
            except Exception as exc:
                logger.warning("Annotated clip failed for %s: %s", out_name, exc)

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
