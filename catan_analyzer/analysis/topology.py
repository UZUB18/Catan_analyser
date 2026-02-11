from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from catan_analyzer.domain.board import BoardState

TOPOLOGY_MAX_ROAD_DISTANCE = 5
TOPOLOGY_FRONTIER_TOP_K = 5
TOPOLOGY_DISTANCE_LAMBDA = 0.55
TOPOLOGY_FRONTIER_WEIGHT = 0.45
TOPOLOGY_BEST_PATH_WEIGHT = 0.30


def build_settlement_blocked_vertices(
    board: BoardState,
    occupied_vertices: Iterable[int],
    *,
    source_vertex_id: int | None = None,
) -> set[int]:
    occupied = set(occupied_vertices)
    if source_vertex_id is not None:
        occupied.add(source_vertex_id)
    return board.blocked_vertices(occupied)


def road_distance_map(
    board: BoardState,
    source_vertex_id: int,
    blocked_vertices: Iterable[int] | None = None,
) -> dict[int, int]:
    blocked = set(blocked_vertices or ())
    blocked.discard(source_vertex_id)

    distances = {source_vertex_id: 0}
    queue: deque[int] = deque([source_vertex_id])

    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        for neighbor in board.vertices[current].adjacent_vertex_ids:
            if neighbor in blocked or neighbor in distances:
                continue
            distances[neighbor] = current_distance + 1
            queue.append(neighbor)
    return distances


def candidate_expansion_vertices(
    board: BoardState,
    source_vertex_id: int,
    settlement_blocked_vertices: set[int],
    distance_map: dict[int, int],
    *,
    max_distance: int = TOPOLOGY_MAX_ROAD_DISTANCE,
) -> list[int]:
    candidates = []
    for vertex_id, distance in distance_map.items():
        if vertex_id == source_vertex_id:
            continue
        if vertex_id in settlement_blocked_vertices:
            continue
        if distance < 2 or distance > max_distance:
            continue
        candidates.append(vertex_id)
    return sorted(candidates, key=lambda vertex_id: (distance_map[vertex_id], vertex_id))


def frontier_metrics(
    board: BoardState,
    source_vertex_id: int,
    settlement_blocked_vertices: set[int],
    distance_map: dict[int, int],
    *,
    value_fn: Callable[[int], float],
    max_distance: int = TOPOLOGY_MAX_ROAD_DISTANCE,
    top_k: int = TOPOLOGY_FRONTIER_TOP_K,
    distance_lambda: float = TOPOLOGY_DISTANCE_LAMBDA,
) -> tuple[float, float]:
    candidates = candidate_expansion_vertices(
        board=board,
        source_vertex_id=source_vertex_id,
        settlement_blocked_vertices=settlement_blocked_vertices,
        distance_map=distance_map,
        max_distance=max_distance,
    )
    if not candidates:
        return (0.0, 0.0)

    path_adjusted_values = [
        value_fn(vertex_id) - (distance_lambda * distance_map[vertex_id]) for vertex_id in candidates
    ]

    best_path_score = max(path_adjusted_values, default=0.0)
    positive_values = sorted((value for value in path_adjusted_values if value > 0.0), reverse=True)
    if not positive_values:
        frontier_score = 0.0
    else:
        top_values = positive_values[: max(1, top_k)]
        frontier_score = sum(top_values) / len(top_values)

    return (round(max(0.0, frontier_score), 4), round(max(0.0, best_path_score), 4))
