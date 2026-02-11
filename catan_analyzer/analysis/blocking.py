from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from catan_analyzer.domain.board import BoardState, PortType, Resource

from .scoring import pip_value
from .topology import (
    TOPOLOGY_DISTANCE_LAMBDA,
    build_settlement_blocked_vertices,
    candidate_expansion_vertices,
    road_distance_map,
)


@dataclass(frozen=True)
class ExpansionSnapshot:
    best_vertex_id: int | None
    best_distance: int
    best_path_value: float


@dataclass(frozen=True)
class BlockingImpact:
    delta: float
    distance_increase: int
    denied_best_vertex: bool


def blocking_externality_delta(
    board: BoardState,
    *,
    source_vertices: Iterable[int],
    occupied_before: Iterable[int],
    occupied_after: Iterable[int],
    include_ports: bool,
    expected_yield_overrides: Mapping[int, float] | None = None,
) -> BlockingImpact:
    source = tuple(dict.fromkeys(source_vertices))
    if not source:
        return BlockingImpact(delta=0.0, distance_increase=0, denied_best_vertex=False)

    before = best_expansion_snapshot(
        board,
        source_vertices=source,
        occupied_vertices=occupied_before,
        include_ports=include_ports,
        expected_yield_overrides=expected_yield_overrides,
    )
    after = best_expansion_snapshot(
        board,
        source_vertices=source,
        occupied_vertices=occupied_after,
        include_ports=include_ports,
        expected_yield_overrides=expected_yield_overrides,
    )

    value_drop = max(0.0, before.best_path_value - after.best_path_value)
    before_distance = before.best_distance if before.best_distance > 0 else 0
    after_distance = after.best_distance if after.best_distance > 0 else before_distance + 2
    distance_increase = max(0, after_distance - before_distance)
    denied_best = bool(
        before.best_vertex_id is not None
        and before.best_vertex_id in set(occupied_after)
        and before.best_vertex_id not in set(occupied_before)
    )

    delta = value_drop + (0.35 * distance_increase) + (0.8 if denied_best else 0.0)
    return BlockingImpact(
        delta=round(max(0.0, delta), 4),
        distance_increase=distance_increase,
        denied_best_vertex=denied_best,
    )


def best_expansion_snapshot(
    board: BoardState,
    *,
    source_vertices: Iterable[int],
    occupied_vertices: Iterable[int],
    include_ports: bool,
    expected_yield_overrides: Mapping[int, float] | None = None,
) -> ExpansionSnapshot:
    occupied = set(occupied_vertices)
    sources = list(dict.fromkeys(source_vertices))
    if not sources:
        return ExpansionSnapshot(best_vertex_id=None, best_distance=0, best_path_value=0.0)

    settlement_blocked = build_settlement_blocked_vertices(board, occupied)
    best_vertex_id: int | None = None
    best_distance = 0
    best_value = float("-inf")

    for source_vertex_id in sources:
        distance_map = road_distance_map(
            board,
            source_vertex_id=source_vertex_id,
            blocked_vertices=occupied - {source_vertex_id},
        )
        candidates = candidate_expansion_vertices(
            board,
            source_vertex_id=source_vertex_id,
            settlement_blocked_vertices=settlement_blocked,
            distance_map=distance_map,
        )

        for candidate_vertex_id in candidates:
            distance = distance_map[candidate_vertex_id]
            candidate_value = static_vertex_value(
                board,
                candidate_vertex_id,
                include_ports=include_ports,
                expected_yield_overrides=expected_yield_overrides,
            )
            path_value = candidate_value - (TOPOLOGY_DISTANCE_LAMBDA * distance)
            if path_value > best_value:
                best_value = path_value
                best_vertex_id = candidate_vertex_id
                best_distance = distance

    if best_vertex_id is None:
        return ExpansionSnapshot(best_vertex_id=None, best_distance=0, best_path_value=0.0)
    return ExpansionSnapshot(
        best_vertex_id=best_vertex_id,
        best_distance=best_distance,
        best_path_value=round(best_value, 4),
    )


def static_vertex_value(
    board: BoardState,
    vertex_id: int,
    *,
    include_ports: bool,
    expected_yield_overrides: Mapping[int, float] | None = None,
) -> float:
    expected_override = expected_yield_overrides.get(vertex_id) if expected_yield_overrides else None

    adjacent_tiles = board.vertex_adjacent_tiles(vertex_id)
    expected_yield = float(
        expected_override
        if expected_override is not None
        else sum(pip_value(tile.token_number) for tile in adjacent_tiles)
    )

    resources = [tile.resource for tile in adjacent_tiles if tile.resource is not Resource.DESERT]
    unique_count = len(set(resources))
    diversity_score = {0: 0.0, 1: 0.2, 2: 1.0, 3: 2.0}.get(unique_count, 0.0)

    port_score = 0.0
    if include_ports:
        port_type = board.vertices[vertex_id].port_type
        if port_type is PortType.ANY_3TO1:
            port_score = 0.3
        elif port_type is not None:
            port_resource = {
                PortType.WOOD_2TO1: Resource.WOOD,
                PortType.BRICK_2TO1: Resource.BRICK,
                PortType.SHEEP_2TO1: Resource.SHEEP,
                PortType.WHEAT_2TO1: Resource.WHEAT,
                PortType.ORE_2TO1: Resource.ORE,
            }[port_type]
            port_score = 0.55 if port_resource in resources else 0.1

    return expected_yield + diversity_score + port_score
