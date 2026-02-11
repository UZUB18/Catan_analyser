import unittest

from catan_analyzer.domain.board import Resource
from catan_analyzer.domain.randomizer import (
    NUMBER_TOKENS,
    RESOURCE_COUNTS,
    generate_randomized_board,
    validate_red_token_spacing,
    validate_standard_counts,
)


class RandomizerTests(unittest.TestCase):
    def test_randomized_board_has_standard_counts(self) -> None:
        board = generate_randomized_board(seed=42)
        self.assertTrue(validate_standard_counts(board))

    def test_desert_has_no_number_token(self) -> None:
        board = generate_randomized_board(seed=7)
        deserts = [tile for tile in board.tiles if tile.resource is Resource.DESERT]
        self.assertEqual(len(deserts), 1)
        self.assertIsNone(deserts[0].token_number)

    def test_number_tokens_cover_official_set(self) -> None:
        board = generate_randomized_board(seed=99)
        numbers = [tile.token_number for tile in board.tiles if tile.token_number is not None]
        self.assertEqual(sorted(numbers), sorted(NUMBER_TOKENS))

    def test_resource_count_distribution_is_exact(self) -> None:
        board = generate_randomized_board(seed=101)
        actual_counts = {resource: 0 for resource in RESOURCE_COUNTS}
        for tile in board.tiles:
            actual_counts[tile.resource] += 1
        self.assertEqual(actual_counts, RESOURCE_COUNTS)

    def test_no_adjacent_red_tokens(self) -> None:
        for seed in range(30):
            board = generate_randomized_board(seed=seed)
            self.assertTrue(
                validate_red_token_spacing(board),
                msg=f"Red token spacing invalid for seed {seed}",
            )


if __name__ == "__main__":
    unittest.main()
