import unittest

from catan_analyzer.analysis.simulation import create_analyzer
from catan_analyzer.analysis.types import AnalysisConfig, AnalysisMode
from catan_analyzer.domain.randomizer import generate_randomized_board


class FullGameAnalysisTests(unittest.TestCase):
    def test_full_game_mode_returns_summary_and_ranking(self) -> None:
        board = generate_randomized_board(seed=610)
        config = AnalysisConfig(
            player_count=3,
            mode=AnalysisMode.FULL_GAME,
            mc_seed=21,
            full_game_rollouts=3,
            full_game_candidate_vertices=4,
            full_game_max_turns=70,
            full_game_trade_offer_limit=1,
        )
        analyzer = create_analyzer(config.mode)
        result = analyzer.analyze(board, config)

        self.assertGreater(len(result.global_ranking), 0)
        self.assertGreater(len(result.top_recommendations), 0)
        self.assertIsNotNone(result.full_game_summary)
        assert result.full_game_summary is not None
        self.assertEqual(result.full_game_summary.rollout_count, config.full_game_rollouts * config.full_game_candidate_vertices)
        self.assertEqual(set(result.full_game_summary.player_win_rates.keys()), {1, 2, 3})
        self.assertGreater(len(result.full_game_summary.top_candidates), 0)
        self.assertLessEqual(len(result.full_game_summary.top_candidates), 8)
        self.assertTrue(any(line.action.startswith("FG#") for line in result.explain_lines))

    def test_full_game_mode_is_reproducible_with_seed(self) -> None:
        board = generate_randomized_board(seed=611)
        config = AnalysisConfig(
            player_count=3,
            mode=AnalysisMode.FULL_GAME,
            mc_seed=77,
            full_game_rollouts=2,
            full_game_candidate_vertices=3,
            full_game_max_turns=60,
            full_game_trade_offer_limit=1,
        )
        analyzer = create_analyzer(config.mode)
        result_one = analyzer.analyze(board, config)
        result_two = analyzer.analyze(board, config)

        self.assertEqual(
            [score.vertex_id for score in result_one.global_ranking[:8]],
            [score.vertex_id for score in result_two.global_ranking[:8]],
        )
        self.assertEqual(
            [round(score.total_score, 3) for score in result_one.global_ranking[:8]],
            [round(score.total_score, 3) for score in result_two.global_ranking[:8]],
        )
        self.assertIsNotNone(result_one.full_game_summary)
        self.assertIsNotNone(result_two.full_game_summary)
        assert result_one.full_game_summary is not None
        assert result_two.full_game_summary is not None
        self.assertEqual(
            result_one.full_game_summary.player_win_rates,
            result_two.full_game_summary.player_win_rates,
        )


if __name__ == "__main__":
    unittest.main()
