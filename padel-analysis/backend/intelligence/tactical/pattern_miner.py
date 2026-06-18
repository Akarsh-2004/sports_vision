"""Tactical pattern mining across the match."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.intelligence.interaction.graph import RallyGraph
from backend.intelligence.shot.understanding import ShotUnderstanding


class PatternMiner:
    """Discover recurring shot sequences and winning patterns."""

    def mine(
        self,
        shots: list[ShotUnderstanding],
        rally_graphs: list[RallyGraph],
        target_id: int,
    ) -> dict[str, Any]:
        patterns: dict[str, Any] = {"sequences": [], "insights": []}

        # 2-shot sequences
        seq_counts: dict[str, int] = defaultdict(int)
        seq_wins: dict[str, int] = defaultdict(int)
        target_shots = [s for s in shots if s.player_id == target_id]
        for i in range(len(target_shots) - 1):
            key = f"{target_shots[i].stroke.value} → {target_shots[i+1].stroke.value}"
            seq_counts[key] += 1

        top_seqs = sorted(seq_counts.items(), key=lambda x: -x[1])[:8]
        patterns["sequences"] = [{"pattern": k, "count": v} for k, v in top_seqs]

        # Net approach correlation
        net_shots = [s for s in target_shots if s.region.value in ("net_front", "smash_zone", "attack_zone")]
        if target_shots:
            net_pct = len(net_shots) / len(target_shots)
            if net_pct > 0.35:
                patterns["insights"].append(
                    f"Aggressive net-oriented play: {net_pct*100:.0f}% of shots from attack zones."
                )

        # Lob → smash patterns from rally chains
        for rg in rally_graphs:
            chain = rg.chain_summary().lower()
            if "lob" in chain and "smash" in chain:
                patterns["insights"].append(
                    f"Rally {rg.rally_id}: lob → smash pattern detected ({rg.chain_summary()})."
                )

        # Opponent lob when target at net
        lobs_after_net = sum(
            1
            for s in target_shots
            if s.intent.value == "finishing" and s.expected_outcome == "force_lob_recovery"
        )
        if lobs_after_net >= 2:
            patterns["insights"].append(
                "Opponent frequently forced into lob recovery after your net pressure."
            )

        return patterns
