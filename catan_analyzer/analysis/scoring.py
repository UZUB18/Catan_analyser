from __future__ import annotations

from typing import Iterable, Mapping, Optional

from catan_analyzer.domain.board import BoardState, PortType, Resource

from .topology import (
    TOPOLOGY_BEST_PATH_WEIGHT,
    TOPOLOGY_FRONTIER_WEIGHT,
    build_settlement_blocked_vertices,
    frontier_metrics,
    road_distance_map,
)
from .types import VertexScore

PIP_VALUES = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    8: 5,
    9: 4,
    10: 3,
    11: 2,
    12: 1,
}


def pip_value(token_number: Optional[int]) -> int:
    if token_number is None:
        return 0
    return PIP_VALUES.get(token_number, 0)


def score_vertex(
    board: BoardState,
    vertex_id: int,
    *,
    include_ports: bool = True,
    occupied_vertices: Optional[Iterable[int]] = None,
    player_existing_vertices: Optional[Iterable[int]] = None,
    expected_yield_override: Optional[float] = None,
    expected_yield_overrides: Optional[Mapping[int, float]] = None,
    include_topology: bool = True,
) -> VertexScore:
    adjacent_tiles = board.vertex_adjacent_tiles(vertex_id)
    resources = [tile.resource for tile in adjacent_tiles if tile.resource is not Resource.DESERT]

    expected_yield = _expected_yield(
        board,
        vertex_id,
        direct_override=expected_yield_override,
        expected_yield_overrides=expected_yield_overrides,
    )
    diversity_score = _diversity_bonus(resources)
    port_score = _port_bonus(board.vertices[vertex_id].port_type, resources, include_ports)
    risk_penalty = _risk_penalty(board, vertex_id)
    synergy_score = _synergy_bonus(board, resources, player_existing_vertices or ())
    frontier_score = 0.0
    best_path_score = 0.0

    if include_topology:
        frontier_score, best_path_score = _topology_scores(
            board=board,
            vertex_id=vertex_id,
            include_ports=include_ports,
            occupied_vertices=occupied_vertices or (),
            player_existing_vertices=player_existing_vertices or (),
            expected_yield_overrides=expected_yield_overrides,
        )

    topology_bonus = (TOPOLOGY_FRONTIER_WEIGHT * frontier_score) + (
        TOPOLOGY_BEST_PATH_WEIGHT * best_path_score
    )

    total_score = (
        expected_yield
        + diversity_score
        + port_score
        + synergy_score
        + topology_bonus
        - risk_penalty
    )
    return VertexScore(
        vertex_id=vertex_id,
        total_score=round(total_score, 4),
        expected_yield=round(expected_yield, 4),
        diversity_score=round(diversity_score, 4),
        port_score=round(port_score, 4),
        risk_penalty=round(risk_penalty, 4),
        synergy_score=round(synergy_score, 4),
        frontier_score=round(frontier_score, 4),
        best_path_score=round(best_path_score, 4),
    )


def _expected_yield(
    board: BoardState,
    vertex_id: int,
    *,
    direct_override: Optional[float] = None,
    expected_yield_overrides: Optional[Mapping[int, float]] = None,
) -> float:
    if direct_override is not None:
        return float(direct_override)
    if expected_yield_overrides and vertex_id in expected_yield_overrides:
        return float(expected_yield_overrides[vertex_id])
    adjacent_tiles = board.vertex_adjacent_tiles(vertex_id)
    return float(sum(pip_value(tile.token_number) for tile in adjacent_tiles))


def _topology_scores(
    *,
    board: BoardState,
    vertex_id: int,
    include_ports: bool,
    occupied_vertices: Iterable[int],
    player_existing_vertices: Iterable[int],
    expected_yield_overrides: Optional[Mapping[int, float]],
) -> tuple[float, float]:
    occupied = set(occupied_vertices)
    player_owned = set(player_existing_vertices)

    settlement_blocked_vertices = build_settlement_blocked_vertices(
        board,
        occupied,
        source_vertex_id=vertex_id,
    )
    travel_blocked_vertices = occupied - player_owned
    travel_blocked_vertices.discard(vertex_id)
    distance_map = road_distance_map(
        board,
        source_vertex_id=vertex_id,
        blocked_vertices=travel_blocked_vertices,
    )

    def value_fn(target_vertex_id: int) -> float:
        return _static_vertex_value(
            board,
            target_vertex_id,
            include_ports=include_ports,
            expected_yield_overrides=expected_yield_overrides,
        )

    return frontier_metrics(
        board=board,
        source_vertex_id=vertex_id,
        settlement_blocked_vertices=settlement_blocked_vertices,
        distance_map=distance_map,
        value_fn=value_fn,
    )


def _static_vertex_value(
    board: BoardState,
    vertex_id: int,
    *,
    include_ports: bool,
    expected_yield_overrides: Optional[Mapping[int, float]],
) -> float:
    adjacent_tiles = board.vertex_adjacent_tiles(vertex_id)
    resources = [tile.resource for tile in adjacent_tiles if tile.resource is not Resource.DESERT]
    expected_yield = _expected_yield(
        board,
        vertex_id,
        expected_yield_overrides=expected_yield_overrides,
    )
    diversity_score = _diversity_bonus(resources)
    port_score = _port_bonus(board.vertices[vertex_id].port_type, resources, include_ports)
    return expected_yield + diversity_score + port_score


def _diversity_bonus(resources: list[Resource]) -> float:
    unique_count = len(set(resources))
    base_bonus = {0: 0.0, 1: 0.2, 2: 1.0, 3: 2.0}.get(unique_count, 0.0)

    combo_bonus = 0.0
    resource_set = set(resources)
    if {Resource.WOOD, Resource.BRICK}.issubset(resource_set):
        combo_bonus += 0.25
    if {Resource.WHEAT, Resource.ORE}.issubset(resource_set):
        combo_bonus += 0.25
    if {Resource.WHEAT, Resource.SHEEP}.issubset(resource_set):
        combo_bonus += 0.1

    duplicate_penalty = 0.2 * (len(resources) - unique_count)
    return base_bonus + combo_bonus - duplicate_penalty


def _port_bonus(port_type: Optional[PortType], resources: list[Resource], include_ports: bool) -> float:
    if not include_ports or port_type is None:
        return 0.0

    if port_type is PortType.ANY_3TO1:
        return 0.3 if resources else 0.15

    resource_for_port = {
        PortType.WOOD_2TO1: Resource.WOOD,
        PortType.BRICK_2TO1: Resource.BRICK,
        PortType.SHEEP_2TO1: Resource.SHEEP,
        PortType.WHEAT_2TO1: Resource.WHEAT,
        PortType.ORE_2TO1: Resource.ORE,
    }[port_type]
    return 0.55 if resource_for_port in resources else 0.1


def _risk_penalty(board: BoardState, vertex_id: int) -> float:
    target_hexes = set(board.vertices[vertex_id].adjacent_hex_ids)
    max_overlap_pips = 0

    for other_id, other_vertex in board.vertices.items():
        if other_id == vertex_id:
            continue
        shared_hexes = target_hexes.intersection(other_vertex.adjacent_hex_ids)
        if not shared_hexes:
            continue
        overlap_pips = sum(pip_value(board.get_tile(tile_id).token_number) for tile_id in shared_hexes)
        max_overlap_pips = max(max_overlap_pips, overlap_pips)

    return max_overlap_pips * 0.1


def _synergy_bonus(
    board: BoardState,
    current_resources: list[Resource],
    player_existing_vertices: Iterable[int],
) -> float:
    existing_vertices = list(player_existing_vertices)
    if not existing_vertices:
        return 0.0

    existing_resources: list[Resource] = []
    for vertex_id in existing_vertices:
        for tile in board.vertex_adjacent_tiles(vertex_id):
            if tile.resource is not Resource.DESERT:
                existing_resources.append(tile.resource)

    existing_set = set(existing_resources)
    current_set = set(current_resources)
    new_unique_resources = len(current_set - existing_set)
    overlap = len(current_set.intersection(existing_set))

    bonus = (0.6 * new_unique_resources) - (0.15 * overlap)

    wheat_ore_cross = (Resource.WHEAT in current_set and Resource.ORE in existing_set) or (
        Resource.ORE in current_set and Resource.WHEAT in existing_set
    )
    if wheat_ore_cross:
        bonus += 0.2

    wood_brick_cross = (Resource.WOOD in current_set and Resource.BRICK in existing_set) or (
        Resource.BRICK in current_set and Resource.WOOD in existing_set
    )
    if wood_brick_cross:
        bonus += 0.2

    return bonus
