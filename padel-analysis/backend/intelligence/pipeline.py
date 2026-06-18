"""Post-vision intelligence pipeline — World Model → reasoning → reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.intelligence.explain.evidence import ExplainabilityEngine
from backend.intelligence.interaction.graph import InteractionGraph
from backend.intelligence.interaction.rally_graph_builder import RallyGraphBuilder
from backend.intelligence.knowledge.reasoner import KnowledgeReasoner
from backend.intelligence.knowledge.structured_events import KnowledgeEngine
from backend.intelligence.opponent.profiler import OpponentProfiler
from backend.intelligence.report.engine import ReportEngine
from backend.intelligence.shot.understanding import ShotUnderstanding
from backend.intelligence.tactical.coach_engine import CoachTacticalEngine
from backend.intelligence.tactical.pattern_miner import PatternMiner
from backend.intelligence.world.world_model import WorldModel
from backend.utils.types import RallySegment
from backend.storage.learning_db import LearningDatabase
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class IntelligencePipeline:
    """Layers 2–7: everything reads from WorldModel."""

    def __init__(self, config: dict, target_id: int):
        self.config = config
        self.target_id = target_id
        self.fps = config["pipeline"]["target_fps"]
        self.tactical = CoachTacticalEngine(config, target_id)
        self.knowledge = KnowledgeEngine()
        self.reasoner = KnowledgeReasoner(config)
        self.reporter = ReportEngine()
        self.pattern_miner = PatternMiner()
        self.opponent_profiler = OpponentProfiler()
        self.explainer = ExplainabilityEngine()
        self.rally_builder = RallyGraphBuilder(self.fps)
        db_path = Path(config["paths"]["data_reports"]) / "padel_learning.db"
        self.learning_db = LearningDatabase(db_path)

    def finalize(
        self,
        world: WorldModel,
        match_id: str,
        source_video: str,
        duration_s: float,
        interaction_graph: InteractionGraph,
        shots: list[ShotUnderstanding],
        all_player_ids: set[int],
        analytics_rallies: list[RallySegment] | None = None,
    ) -> dict[str, Any]:
        rally_graphs = self.rally_builder.build(world, analytics_rallies)
        rally_exports = [rg.to_dict() for rg in rally_graphs]

        tactical = self.tactical.analyze_match_from_shots(shots, world)
        patterns = self.pattern_miner.mine(shots, rally_graphs, self.target_id)
        opponents = self.opponent_profiler.profile(shots, self.target_id, all_player_ids)
        recommendations = self.explainer.build_recommendations(tactical, shots, patterns, self.fps)
        self_eval = world.self_evaluation()

        knowledge_pkg = self.knowledge.package_from_shots(
            world,
            shots,
            tactical,
            self.target_id,
            {
                "match_id": match_id,
                "duration_s": duration_s,
                "target_player": self.target_id,
                "patterns": patterns,
                "opponents": opponents,
                "self_evaluation": self_eval,
            },
        )
        knowledge_pkg["recommendations"] = recommendations
        knowledge_pkg["rally_graphs"] = rally_exports

        intelligence_report = self.reasoner.reason(knowledge_pkg)
        reports = self.reporter.generate_all(knowledge_pkg, intelligence_report)

        self.learning_db.save_match(world, match_id, source_video, duration_s, shots, rally_exports)
        self.learning_db.save_player_session(self.target_id, match_id, shots, tactical.to_dict())

        logger.info(
            "Intelligence: %d interactions, %d shots, %d rally graphs, %d patterns",
            len(world.all_interactions),
            len(shots),
            len(rally_graphs),
            len(patterns.get("sequences", [])),
        )

        return {
            "world_model_summary": world.summary(),
            "self_evaluation": self_eval,
            "timeline_events": world.timeline_events(),
            "interaction_graph": [n.to_dict() for n in world.all_interactions],
            "rally_graphs": rally_exports,
            "shot_understanding": [s.to_dict() for s in shots],
            "tactical_intelligence": tactical.to_dict(),
            "pattern_mining": patterns,
            "opponent_profiles": opponents,
            "recommendations": recommendations,
            "knowledge_package": knowledge_pkg,
            "reports": reports,
            "coach_report": reports["coach"],
        }

    def close(self) -> None:
        self.learning_db.close()
