import unittest

from catan_analyzer.domain.randomizer import generate_randomized_board


class BoardGraphTests(unittest.TestCase):
    def test_vertex_adjacency_is_symmetric(self) -> None:
        board = generate_randomized_board(seed=12)
        for vertex_id, vertex in board.vertices.items():
            self.assertNotIn(vertex_id, vertex.adjacent_vertex_ids)
            for neighbor_id in vertex.adjacent_vertex_ids:
                self.assertIn(vertex_id, board.vertices[neighbor_id].adjacent_vertex_ids)

    def test_distance_rule_blocks_neighbor_vertices(self) -> None:
        board = generate_randomized_board(seed=13)
        first_vertex_id = min(board.vertices.keys())
        neighbor_id = board.vertices[first_vertex_id].adjacent_vertex_ids[0]

        occupied = {first_vertex_id}
        self.assertFalse(board.is_legal_settlement(neighbor_id, occupied))
        self.assertNotIn(neighbor_id, board.legal_settlement_vertices(occupied))
        self.assertGreater(len(board.legal_settlement_vertices(occupied)), 0)

    def test_standard_board_vertex_count(self) -> None:
        board = generate_randomized_board(seed=14)
        self.assertEqual(len(board.vertices), 54)

    def test_normalize_edge_key_is_order_independent(self) -> None:
        board = generate_randomized_board(seed=15)
        edge = next(iter(board.edges.keys()))
        first, second = edge
        self.assertEqual(
            board.normalize_edge_key(first, second),
            board.normalize_edge_key(second, first),
        )

    def test_edge_exists_matches_adjacency_graph(self) -> None:
        board = generate_randomized_board(seed=16)
        existing_edge = next(iter(board.edges.keys()))
        self.assertTrue(board.edge_exists(existing_edge))

        all_vertices = sorted(board.vertices.keys())
        non_edge = None
        for first in all_vertices:
            neighbors = set(board.vertices[first].adjacent_vertex_ids)
            for second in all_vertices:
                if first == second or second in neighbors:
                    continue
                non_edge = (first, second)
                break
            if non_edge is not None:
                break
        self.assertIsNotNone(non_edge)
        assert non_edge is not None
        self.assertFalse(board.edge_exists(non_edge))

    def test_first_road_is_legal_when_touching_settlement(self) -> None:
        board = generate_randomized_board(seed=17)
        edge = next(iter(board.edges.keys()))
        settlement_vertex = edge[0]
        self.assertTrue(
            board.is_legal_road(
                edge,
                settlements={settlement_vertex},
                roads=set(),
            )
        )

    def test_disconnected_road_is_illegal(self) -> None:
        board = generate_randomized_board(seed=18)
        first_edge = next(iter(board.edges.keys()))
        settlements = {first_edge[0]}
        roads = {board.normalize_edge_key(*first_edge)}

        disconnected_edge = None
        blocked_vertices = set(first_edge)
        for candidate in board.edges.keys():
            if candidate == first_edge:
                continue
            if blocked_vertices.intersection(candidate):
                continue
            disconnected_edge = candidate
            break

        self.assertIsNotNone(disconnected_edge)
        assert disconnected_edge is not None
        self.assertFalse(board.is_legal_road(disconnected_edge, settlements=settlements, roads=roads))

    def test_road_chain_extension_is_legal(self) -> None:
        board = generate_randomized_board(seed=19)
        first_edge = next(iter(board.edges.keys()))
        settlement_vertex = first_edge[0]
        roads = {board.normalize_edge_key(*first_edge)}

        extension_edge = None
        extension_anchor = first_edge[1]
        for candidate in board.edges.keys():
            if candidate == first_edge:
                continue
            if extension_anchor in candidate and settlement_vertex not in candidate:
                extension_edge = candidate
                break

        self.assertIsNotNone(extension_edge)
        assert extension_edge is not None
        self.assertTrue(
            board.is_legal_road(
                extension_edge,
                settlements={settlement_vertex},
                roads=roads,
            )
        )

    def test_duplicate_road_is_illegal(self) -> None:
        board = generate_randomized_board(seed=20)
        edge = next(iter(board.edges.keys()))
        normalized = board.normalize_edge_key(*edge)
        self.assertFalse(
            board.is_legal_road(
                edge,
                settlements={edge[0]},
                roads={normalized},
            )
        )


if __name__ == "__main__":
    unittest.main()
