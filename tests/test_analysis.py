import unittest

from catan_analyzer.analysis.draft import rank_vertices
from catan_analyzer.analysis.scoring import score_vertex
from catan_analyzer.analysis.simulation import (
    HeuristicAnalyzer,
    HybridOpeningAnalyzer,
    MonteCarloAnalyzer,
    PhaseRolloutAnalyzer,
    create_analyzer,
)
from catan_analyzer.analysis.topology import (
    TOPOLOGY_MAX_ROAD_DISTANCE,
    build_settlement_blocked_vertices,
    candidate_expansion_vertices,
    road_distance_map,
)
from catan_analyzer.analysis.types import AnalysisConfig, AnalysisMode, RobberPolicy
from catan_analyzer.domain.randomizer import generate_randomized_board


class AnalysisTests(unittest.TestCase):
    def test_heuristic_mode_is_deterministic_for_same_board(self) -> None:
        board = generate_randomized_board(seed=123)
        config = AnalysisConfig(player_count=4, mode=AnalysisMode.HEURISTIC, include_ports=True)
        analyzer = HeuristicAnalyzer()

        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)
        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:10]],
            [score.vertex_id for score in result_two.global_ranking[:10]],
        )

    def test_draft_sequence_obeys_settlement_legality(self) -> None:
        board = generate_randomized_board(seed=45)
        config = AnalysisConfig(player_count=4, mode=AnalysisMode.HEURISTIC)
        result = HeuristicAnalyzer().analyze(board, config)

        occupied: set[int] = set()
        self.assertEqual(len(result.predicted_sequence), 8)
        for pick in result.predicted_sequence:
            self.assertTrue(board.is_legal_settlement(pick.vertex_id, occupied))
            occupied.add(pick.vertex_id)

    def test_output_size_changes_with_player_count(self) -> None:
        board = generate_randomized_board(seed=98)
        analyzer = HeuristicAnalyzer()
        for player_count in (2, 3, 4):
            config = AnalysisConfig(player_count=player_count, mode=AnalysisMode.HEURISTIC)
            result = analyzer.analyze(board, config)
            self.assertEqual(len(result.top_recommendations), player_count * 2)

    def test_port_scoring_changes_when_ports_are_enabled(self) -> None:
        board = generate_randomized_board(seed=88)
        port_vertex_id = next(vertex.id for vertex in board.vertices.values() if vertex.port_type is not None)

        with_ports = score_vertex(board, port_vertex_id, include_ports=True)
        without_ports = score_vertex(board, port_vertex_id, include_ports=False)
        self.assertGreater(with_ports.port_score, without_ports.port_score)

    def test_monte_carlo_mode_returns_stable_seeded_results(self) -> None:
        board = generate_randomized_board(seed=77)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.MONTE_CARLO,
            mc_iterations=200,
            mc_rolls_per_game=20,
            mc_seed=11,
        )
        analyzer = MonteCarloAnalyzer()
        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)

        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:10]],
            [score.vertex_id for score in result_two.global_ranking[:10]],
        )

    def test_monte_carlo_mode_is_reproducible_without_explicit_seed(self) -> None:
        board = generate_randomized_board(seed=170)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.MONTE_CARLO,
            mc_iterations=150,
            mc_rolls_per_game=20,
            mc_seed=None,
        )
        analyzer = MonteCarloAnalyzer()
        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)
        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:10]],
            [score.vertex_id for score in result_two.global_ranking[:10]],
        )

    def test_topology_scores_are_present_in_rankings(self) -> None:
        board = generate_randomized_board(seed=31)
        config = AnalysisConfig(player_count=4, mode=AnalysisMode.HEURISTIC)
        result = HeuristicAnalyzer().analyze(board, config)

        self.assertTrue(any(score.frontier_score > 0 for score in result.global_ranking))
        self.assertTrue(any(score.best_path_score > 0 for score in result.global_ranking))

    def test_topology_candidates_respect_distance_and_legality(self) -> None:
        board = generate_randomized_board(seed=5)
        source_vertex_id = min(board.vertices.keys())
        occupied = {max(board.vertices.keys())}

        settlement_blocked_vertices = build_settlement_blocked_vertices(
            board,
            occupied_vertices=occupied,
            source_vertex_id=source_vertex_id,
        )
        distance_map = road_distance_map(
            board,
            source_vertex_id=source_vertex_id,
            blocked_vertices=occupied,
        )
        candidates = candidate_expansion_vertices(
            board,
            source_vertex_id=source_vertex_id,
            settlement_blocked_vertices=settlement_blocked_vertices,
            distance_map=distance_map,
            max_distance=TOPOLOGY_MAX_ROAD_DISTANCE,
        )

        self.assertGreater(len(candidates), 0)
        for candidate_vertex_id in candidates:
            self.assertNotIn(candidate_vertex_id, settlement_blocked_vertices)
            self.assertGreaterEqual(distance_map[candidate_vertex_id], 2)
            self.assertLessEqual(distance_map[candidate_vertex_id], TOPOLOGY_MAX_ROAD_DISTANCE)

    def test_ranked_rows_include_topology_fields(self) -> None:
        board = generate_randomized_board(seed=112)

        def score_fn(vertex_id: int, occupied: set[int], player_vertices: list[int]):
            return score_vertex(
                board,
                vertex_id,
                include_ports=True,
                occupied_vertices=occupied,
                player_existing_vertices=player_vertices,
            )

        ranked = rank_vertices(board, score_fn)
        self.assertGreater(len(ranked), 0)
        top = ranked[0]
        self.assertIsInstance(top.frontier_score, float)
        self.assertIsInstance(top.best_path_score, float)

    def test_phase_rollout_mode_returns_stable_seeded_results(self) -> None:
        board = generate_randomized_board(seed=51)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.PHASE_ROLLOUT_MC,
            mc_seed=9,
            phase_rollout_count=36,
            phase_turn_horizon=12,
        )
        analyzer = PhaseRolloutAnalyzer()
        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)
        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:10]],
            [score.vertex_id for score in result_two.global_ranking[:10]],
        )

    def test_phase_rollout_mode_is_reproducible_without_explicit_seed(self) -> None:
        board = generate_randomized_board(seed=171)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.PHASE_ROLLOUT_MC,
            mc_seed=None,
            phase_rollout_count=20,
            phase_turn_horizon=9,
        )
        analyzer = PhaseRolloutAnalyzer()
        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)
        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:10]],
            [score.vertex_id for score in result_two.global_ranking[:10]],
        )

    def test_phase_rollout_populates_milestone_components(self) -> None:
        board = generate_randomized_board(seed=52)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.PHASE_ROLLOUT_MC,
            mc_seed=13,
            phase_rollout_count=24,
            phase_turn_horizon=10,
        )
        result = PhaseRolloutAnalyzer().analyze(board, config)
        top = result.global_ranking[0]
        self.assertNotEqual(top.tempo_score, 0.0)
        self.assertGreater(top.recipe_coverage_score, 0.0)
        self.assertGreaterEqual(top.fragility_penalty, 0.0)
        self.assertGreaterEqual(top.port_conversion_score, 0.0)
        self.assertGreaterEqual(top.robber_penalty, 0.0)

    def test_phase_rollout_robber_policy_changes_scores(self) -> None:
        board = generate_randomized_board(seed=53)
        base_kwargs = dict(
            player_count=4,
            mode=AnalysisMode.PHASE_ROLLOUT_MC,
            mc_seed=21,
            phase_rollout_count=24,
            phase_turn_horizon=10,
        )
        strongest_config = AnalysisConfig(
            **base_kwargs,
            robber_policy=RobberPolicy.TARGET_STRONGEST_OPPONENT,
        )
        worst_case_config = AnalysisConfig(
            **base_kwargs,
            robber_policy=RobberPolicy.WORST_CASE_US,
        )

        analyzer = PhaseRolloutAnalyzer()
        strongest = analyzer.analyze(board, strongest_config)
        worst_case = analyzer.analyze(board, worst_case_config)
        self.assertNotEqual(
            [round(score.total_score, 3) for score in strongest.global_ranking[:10]],
            [round(score.total_score, 3) for score in worst_case.global_ranking[:10]],
        )

    def test_mcts_lite_mode_returns_stable_seeded_results(self) -> None:
        board = generate_randomized_board(seed=91)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.MCTS_LITE_OPENING,
            mc_seed=7,
            phase_rollout_count=12,
            phase_turn_horizon=8,
            mcts_iterations=30,
            mcts_candidate_settlements=7,
            mcts_candidate_road_directions=2,
        )
        analyzer = create_analyzer(config.mode)
        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)
        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:10]],
            [score.vertex_id for score in result_two.global_ranking[:10]],
        )

    def test_mcts_lite_mode_is_reproducible_without_explicit_seed(self) -> None:
        board = generate_randomized_board(seed=172)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.MCTS_LITE_OPENING,
            mc_seed=None,
            phase_rollout_count=10,
            phase_turn_horizon=8,
            mcts_iterations=26,
            mcts_candidate_settlements=7,
            mcts_candidate_road_directions=2,
        )
        analyzer = create_analyzer(config.mode)
        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)
        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:10]],
            [score.vertex_id for score in result_two.global_ranking[:10]],
        )

    def test_mcts_lite_populates_explainer_lines(self) -> None:
        board = generate_randomized_board(seed=92)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.MCTS_LITE_OPENING,
            mc_seed=17,
            phase_rollout_count=12,
            phase_turn_horizon=8,
            mcts_iterations=40,
        )
        result = create_analyzer(config.mode).analyze(board, config)
        self.assertGreater(len(result.explain_lines), 0)
        self.assertIsNotNone(result.mcts_summary)
        self.assertGreater(result.mcts_summary.root_visits, 0)

    def test_mcts_lite_block_weight_changes_top_candidates(self) -> None:
        board = generate_randomized_board(seed=93)
        base_kwargs = dict(
            player_count=4,
            mode=AnalysisMode.MCTS_LITE_OPENING,
            mc_seed=19,
            phase_rollout_count=12,
            phase_turn_horizon=8,
            mcts_iterations=40,
            mcts_candidate_settlements=8,
        )
        low_block = AnalysisConfig(**base_kwargs, opponent_block_weight=0.1)
        high_block = AnalysisConfig(**base_kwargs, opponent_block_weight=0.6)

        analyzer = create_analyzer(AnalysisMode.MCTS_LITE_OPENING)
        low_result = analyzer.analyze(board, low_block)
        high_result = analyzer.analyze(board, high_block)
        self.assertIsNotNone(low_result.mcts_summary)
        self.assertIsNotNone(high_result.mcts_summary)
        self.assertNotEqual(
            round(low_result.mcts_summary.best_line_score, 3),
            round(high_result.mcts_summary.best_line_score, 3),
        )

    def test_hybrid_opening_mode_returns_stable_seeded_results(self) -> None:
        board = generate_randomized_board(seed=94)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.HYBRID_OPENING,
            mc_seed=23,
            phase_rollout_count=12,
            phase_turn_horizon=8,
            mcts_iterations=35,
            mcts_candidate_settlements=8,
            mcts_candidate_road_directions=2,
        )
        analyzer = create_analyzer(config.mode)
        self.assertIsInstance(analyzer, HybridOpeningAnalyzer)
        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)
        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:10]],
            [score.vertex_id for score in result_two.global_ranking[:10]],
        )
        self.assertEqual(
            [line.action for line in result_one.explain_lines[:4]],
            [line.action for line in result_two.explain_lines[:4]],
        )

    def test_hybrid_opening_produces_recommendations_and_explanations(self) -> None:
        board = generate_randomized_board(seed=95)
        config = AnalysisConfig(
            player_count=4,
            mode=AnalysisMode.HYBRID_OPENING,
            mc_seed=31,
            mc_iterations=120,
            mc_rolls_per_game=16,
            phase_rollout_count=10,
            phase_turn_horizon=8,
            mcts_iterations=30,
            hybrid_include_monte_carlo=True,
        )

        result = create_analyzer(config.mode).analyze(board, config)
        self.assertGreater(len(result.top_recommendations), 0)
        self.assertGreater(len(result.explain_lines), 0)
        self.assertTrue(any(line.action.startswith("HYB#") for line in result.explain_lines))
        self.assertIsNotNone(result.mcts_summary)
        self.assertGreater(result.mcts_summary.root_visits, 0)


if __name__ == "__main__":
    unittest.main()
