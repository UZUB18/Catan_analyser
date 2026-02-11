from __future__ import annotations

import random
from typing import Dict, Optional

from .board import BoardState, Resource, build_standard_board

RESOURCE_COUNTS: Dict[Resource, int] = {
    Resource.WOOD: 4,
    Resource.BRICK: 3,
    Resource.SHEEP: 4,
    Resource.WHEAT: 4,
    Resource.ORE: 3,
    Resource.DESERT: 1,
}

NUMBER_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
RED_TOKEN_NUMBERS = {6, 8}
MAX_RANDOMIZATION_ATTEMPTS = 10_000


def generate_randomized_board(seed: Optional[int] = None) -> BoardState:
    rng = random.Random(seed)

    resource_pool = []
    for resource, count in RESOURCE_COUNTS.items():
        resource_pool.extend([resource] * count)

    for _ in range(MAX_RANDOMIZATION_ATTEMPTS):
        resources = resource_pool[:]
        rng.shuffle(resources)

        numbers = NUMBER_TOKENS[:]
        rng.shuffle(numbers)

        board = build_standard_board(resource_order=resources, token_order=numbers)
        if validate_red_token_spacing(board):
            return board

    raise RuntimeError(
        "Unable to generate a board that satisfies red-token spacing constraints "
        f"after {MAX_RANDOMIZATION_ATTEMPTS} attempts."
    )


def validate_standard_counts(board: BoardState) -> bool:
    resource_counts: Dict[Resource, int] = {resource: 0 for resource in RESOURCE_COUNTS}
    numbers = []
    for tile in board.tiles:
        resource_counts[tile.resource] += 1
        if tile.resource is Resource.DESERT:
            if tile.token_number is not None:
                return False
        elif tile.token_number is None:
            return False
        else:
            numbers.append(tile.token_number)

    if resource_counts != RESOURCE_COUNTS:
        return False

    return sorted(numbers) == sorted(NUMBER_TOKENS)


def validate_red_token_spacing(board: BoardState) -> bool:
    red_tile_ids = {tile.id for tile in board.tiles if tile.token_number in RED_TOKEN_NUMBERS}
    if len(red_tile_ids) <= 1:
        return True

    for adjacent_tile_ids in board.edges.values():
        if len(adjacent_tile_ids) != 2:
            continue
        first_tile_id, second_tile_id = adjacent_tile_ids
        if first_tile_id in red_tile_ids and second_tile_id in red_tile_ids:
            return False
    return True
