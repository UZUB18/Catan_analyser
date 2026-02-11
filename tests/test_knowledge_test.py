import unittest

from catan_analyzer.analysis.knowledge_test import evaluate_user_settlement_picks
from catan_analyzer.analysis.types import VertexScore


def _score(vertex_id: int, total: float) -> VertexScore:
    return VertexScore(
        vertex_id=vertex_id,
        total_score=total,
        expected_yield=total,
        diversity_score=0.0,
        port_score=0.0,
        risk_penalty=0.0,
    )


class KnowledgeTestEvaluationTests(unittest.TestCase):
    def test_evaluation_tracks_hits_misses_and_average_rank(self) -> None:
        ranking = [_score(10, 9.0), _score(20, 8.0), _score(30, 7.0), _score(40, 6.0), _score(50, 5.0)]
        evaluation = evaluate_user_settlement_picks([30, 99, 10, 50], ranking, top_n=4)

        self.assertEqual(evaluation.top_vertices, (10, 20, 30, 40))
        self.assertEqual(evaluation.hits, (30, 10))
        self.assertEqual(evaluation.missed_top_vertices, (20, 40))
        self.assertEqual(evaluation.extra_vertices, (99, 50))
        self.assertEqual(evaluation.average_rank, 3.0)
        self.assertEqual(evaluation.numeric_score, 56.0)
        self.assertEqual(evaluation.letter_grade, "F")

    def test_duplicate_user_picks_are_deduplicated_preserving_order(self) -> None:
        ranking = [_score(10, 9.0), _score(20, 8.0), _score(30, 7.0), _score(40, 6.0)]
        evaluation = evaluate_user_settlement_picks([10, 10, 20, 20], ranking, top_n=4)

        self.assertEqual(evaluation.user_picks, (10, 20))
        self.assertEqual(evaluation.hits, (10, 20))
        self.assertEqual(evaluation.hit_count, 2)
        self.assertEqual(evaluation.numeric_score, 50.0)
        self.assertEqual(evaluation.letter_grade, "F")

    def test_perfect_top_four_gets_full_score_and_a_grade(self) -> None:
        ranking = [_score(10, 9.0), _score(20, 8.0), _score(30, 7.0), _score(40, 6.0), _score(50, 5.0)]
        evaluation = evaluate_user_settlement_picks([10, 20, 30, 40], ranking, top_n=4)

        self.assertEqual(evaluation.hit_count, 4)
        self.assertEqual(evaluation.numeric_score, 100.0)
        self.assertEqual(evaluation.letter_grade, "A")

    def test_top_n_must_be_positive(self) -> None:
        ranking = [_score(10, 9.0)]
        with self.assertRaises(ValueError):
            evaluate_user_settlement_picks([10], ranking, top_n=0)


if __name__ == "__main__":
    unittest.main()
