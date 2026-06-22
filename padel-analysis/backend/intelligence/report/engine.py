"""Layer 7 — multi-audience report generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


class ReportEngine:
    """Coach, player, training, and match reports from knowledge package."""

    def generate_all(
        self,
        knowledge: dict,
        intelligence_report: str,
        all_player_ids: set[int] | None = None,
        fps: float = 25.0,
    ) -> dict[str, str]:
        player_ids = sorted(all_player_ids or {knowledge.get("match", {}).get("target_player", 0)})
        shots = knowledge.get("shot_understanding") or knowledge.get("shot_facts", [])
        if shots:
            from collections import Counter

            counts = Counter(s.get("player_id") for s in shots if s.get("player_id", 0) > 0)
            player_ids = sorted(pid for pid, _ in counts.most_common(4))
        per_player = self.generate_per_player(knowledge, player_ids, fps)
        target_id = knowledge.get("match", {}).get("target_player")
        target_report = per_player.get(target_id, self._player_report(knowledge, player_id=target_id))

        return {
            "coach": intelligence_report,
            "player": target_report,
            "training": self._training_report(knowledge, player_id=target_id),
            "match": self._match_summary(knowledge),
            "per_player": {str(pid): text for pid, text in per_player.items()},
        }

    def generate_per_player(
        self,
        knowledge: dict,
        player_ids: list[int],
        fps: float = 25.0,
    ) -> dict[int, str]:
        return {pid: self._player_report(knowledge, player_id=pid, fps=fps) for pid in player_ids}

    def _shots_for_player(self, knowledge: dict, player_id: int | None) -> list[dict]:
        if player_id is None:
            return []
        shots = knowledge.get("shot_understanding") or knowledge.get("shot_facts", [])
        return [s for s in shots if s.get("player_id") == player_id]

    def _interactions_for_player(self, knowledge: dict, player_id: int) -> list[dict]:
        graph = knowledge.get("interaction_graph") or []
        return [
            n
            for n in graph
            if n.get("type") == "player_hit" and n.get("actor") == player_id
        ]

    def _player_report(
        self,
        k: dict,
        player_id: int | None = None,
        fps: float = 25.0,
    ) -> str:
        pid = player_id if player_id is not None else k.get("match", {}).get("target_player")
        shots = self._shots_for_player(k, pid)
        interactions = self._interactions_for_player(k, pid) if pid is not None else []
        tactical = k.get("tactical", {})
        pos = tactical.get("positioning", {}) if pid == k.get("match", {}).get("target_player") else {}

        stroke_counts: Counter[str] = Counter()
        region_counts: Counter[str] = Counter()
        intent_counts: Counter[str] = Counter()
        for s in shots:
            stroke_counts[s.get("stroke", "unknown")] += 1
            region_counts[s.get("region", "unknown")] += 1
            intent_counts[s.get("intent", "neutral")] += 1

        if not shots and interactions:
            for n in interactions:
                stroke_counts[n.get("stroke") or "unknown"] += 1
                intent_counts[n.get("intent") or "neutral"] += 1

        total_shots = sum(stroke_counts.values()) or len(interactions)
        top_strokes = stroke_counts.most_common(3)
        points = k.get("match", {}).get("points_detected", len(k.get("rally_narratives", [])))

        errors = sum(
            1 for s in shots
            if s.get("decision_quality", s.get("decision")) in ("poor", "suboptimal")
        )
        winners = sum(
            1 for s in shots
            if s.get("decision_quality", s.get("decision")) in ("good", "excellent")
            and s.get("intent") in ("attack", "finishing")
        )

        rally_lengths: list[int] = []
        for rg in k.get("rally_narratives", []):
            nodes = rg.get("nodes", [])
            player_hits = [
                n for n in nodes
                if n.get("type") == "player_hit" and n.get("actor") == pid
            ]
            if player_hits:
                rally_lengths.append(len(player_hits))
        avg_rally_hits = (
            round(sum(rally_lengths) / len(rally_lengths), 1) if rally_lengths else 0.0
        )

        lines = [
            f"# Player Report — Player {pid}",
            "",
            "## Match Snapshot",
            f"- Points detected: **{points}**",
            f"- Shots tracked: **{total_shots}**",
            f"- Match duration: **{k.get('match', {}).get('duration_s', 0):.0f}s**",
            "",
        ]

        if pid == k.get("match", {}).get("target_player") and pos:
            lines.extend(
                [
                    "## Positioning (target player)",
                    f"- Net positioning optimal: **{pos.get('optimal_pct', 0) * 100:.0f}%** of active frames",
                    f"- Too deep: **{pos.get('too_deep_pct', 0) * 100:.0f}%**",
                    f"- Wrong lane: **{pos.get('wrong_side_pct', 0) * 100:.0f}%**",
                    "",
                ]
            )

        if top_strokes:
            lines.append("## Shot Profile")
            for stroke, count in top_strokes:
                lines.append(f"- **{stroke.replace('_', ' ').title()}**: {count}×")
            lines.append("")

        if intent_counts:
            attack = intent_counts.get("attack", 0) + intent_counts.get("finishing", 0)
            defense = intent_counts.get("defensive", 0)
            lines.extend(
                [
                    "## Tendencies",
                    f"- Attack / finish intent: **{attack}** shots",
                    f"- Defensive intent: **{defense}** shots",
                    f"- Strong decisions: **{winners}** · Suboptimal: **{errors}**",
                    f"- Avg hits per rally (this player): **{avg_rally_hits}**",
                    "",
                ]
            )

        lines.append("## Key Moments")
        if shots:
            notable = sorted(shots, key=lambda s: s.get("speed_kmh", 0), reverse=True)[:3]
            for s in notable:
                t = _fmt_time(s.get("frame", 0) / fps)
                stroke = (s.get("stroke") or "shot").replace("_", " ")
                speed = s.get("speed_kmh", 0)
                decision = s.get("decision_quality", s.get("decision", "neutral"))
                lines.append(
                    f"- **{t}** — {stroke} ({speed:.0f} km/h, {decision} decision)"
                )
        elif interactions:
            for n in interactions[:3]:
                t = _fmt_time(n.get("frame", 0) / fps)
                stroke = (n.get("stroke") or "hit").replace("_", " ")
                lines.append(f"- **{t}** — {stroke}")
        else:
            lines.append("- No individual shots tracked for this player in this clip.")

        lines.extend(["", "## Focus for Next Session"])
        defense = intent_counts.get("defensive", 0) if intent_counts else 0
        attack = intent_counts.get("attack", 0) + intent_counts.get("finishing", 0) if intent_counts else 0
        if defense > attack and total_shots >= 2:
            lines.append("1. After defensive wall resets, push to the transition zone instead of staying at the glass.")
        else:
            lines.append("1. Build net presence after short rallies — look for volley or bandeja opportunities.")
        if top_strokes and top_strokes[0][0] in ("forehand", "backhand"):
            lines.append(f"2. Your most-used groundstroke is **{top_strokes[0][0]}** — drill depth and placement under pressure.")
        else:
            lines.append("2. Work on volley consistency when approaching the net.")
        lines.append("3. Partner spacing: aim for ~3 m lateral distance during defense.")

        return "\n".join(lines) + "\n"

    def _training_report(self, k: dict, player_id: int | None = None) -> str:
        pid = player_id if player_id is not None else k.get("match", {}).get("target_player")
        shots = self._shots_for_player(k, pid)
        stroke_counts: dict[str, int] = defaultdict(int)
        poor_decisions = 0
        for s in shots:
            stroke = s.get("stroke", "unknown")
            stroke_counts[stroke] += 1
            if s.get("decision_quality", s.get("decision")) in ("poor", "suboptimal"):
                poor_decisions += 1

        top = max(stroke_counts.items(), key=lambda x: x[1], default=("none", 0))
        points = k.get("match", {}).get("points_detected", 0)

        drills = [
            "Wall reset → transition → net approach (10 min)",
            "Bandeja consistency from net (15 min)",
            "Defensive lob depth under pressure (10 min)",
        ]
        if poor_decisions >= 2:
            drills.insert(0, "Decision review: pause before low-percentage shots (10 min)")
        if top[0] in ("volley", "forehand_volley", "backhand_volley"):
            drills[1] = "Reaction volleys at net — partner feed drill (15 min)"

        return f"""# Training Plan — Player {pid}

## Pattern Identified
- Points in clip: **{points}**
- Most frequent shot: **{top[0]}** ({top[1]}×)
- Suboptimal decisions flagged: **{poor_decisions}**

## Drills
{chr(10).join(f"{i + 1}. {d}" for i, d in enumerate(drills[:4]))}
"""

    def _match_summary(self, k: dict) -> str:
        ws = k.get("world_summary", {})
        points = k.get("match", {}).get("points_detected", 0)
        return f"""# Match Summary
- Points detected: {points}
- Active frames: {ws.get('active_frames', 0)} / {ws.get('total_frames', 0)}
- State distribution: {ws.get('state_distribution', {})}
- Interactions logged: {ws.get('interaction_count', 0)}
- Rally segments (FSM): {ws.get('rallies_detected', 0)}
"""
