import unittest

from catan_analyzer.analysis.types import VertexScore
from catan_analyzer.ui.explainability import compute_sensitivity_badges, explain_vertex_score


def _score(vertex_id: int, total: float) -> VertexScore:
    return VertexScore(
        vertex_id=vertex_id,
        total_score=total,
        expected_yield=total,
        diversity_score=0.2,
        port_score=0.1,
        risk_penalty=0.05,
        synergy_score=0.15,
        frontier_score=0.12,
        best_path_score=0.11,
        tempo_score=0.18,
        recipe_coverage_score=0.20,
        fragility_penalty=0.03,
        port_conversion_score=0.04,
        robber_penalty=0.02,
    )


class ExplainabilityUiTests(unittest.TestCase):
    def test_sensitivity_badges_cover_every_vertex(self) -> None:
        ranking = [_score(1, 10.0), _score(2, 9.7), _score(3, 9.65), _score(4, 8.9)]
        badges = compute_sensitivity_badges(ranking)

        self.assertEqual(set(badges.keys()), {1, 2, 3, 4})
        self.assertEqual(badges[1].label, "Stable")
        self.assertIn(badges[2].label, {"Watch", "Volatile"})

    def test_explain_vertex_score_contains_contributions_penalties_and_sensitivity(self) -> None:
        ranking = [_score(9, 7.2)]
        badge = compute_sensitivity_badges(ranking)[9]
        explanation = explain_vertex_score(ranking[0], badge)

        self.assertIn("Vertex 9", explanation)
        self.assertIn("Top contributions", explanation)
        self.assertIn("Top penalties", explanation)
        self.assertIn("Sensitivity:", explanation)


if __name__ == "__main__":
    unittest.main()
