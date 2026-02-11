from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from catan_analyzer.domain.board import BoardState, EdgeKey

from .state import GameState


def _other_vertex(edge: EdgeKey, vertex_id: int) -> int:
    if edge[0] == vertex_id:
        return edge[1]
    return edge[0]


def longest_road_length_for_player(
    board: BoardState,
    *,
    player_roads: Iterable[EdgeKey],
    blocked_vertices: set[int],
) -> int:
    edges = {tuple(sorted((int(edge[0]), int(edge[1])))) for edge in player_roads}
    if not edges:
        return 0

    incident_edges: dict[int, set[EdgeKey]] = defaultdict(set)
    for edge in edges:
        if not board.edge_exists(edge):
            continue
        incident_edges[edge[0]].add(edge)
        incident_edges[edge[1]].add(edge)

    def dfs(vertex_id: int, used_edges: set[EdgeKey], arrived_via_edge: bool) -> int:
        if arrived_via_edge and vertex_id in blocked_vertices:
            return 0

        best_depth = 0
        for edge in incident_edges.get(vertex_id, ()):
            if edge in used_edges:
                continue
            used_edges.add(edge)
            next_vertex_id = _other_vertex(edge, vertex_id)
            best_depth = max(best_depth, 1 + dfs(next_vertex_id, used_edges, True))
            used_edges.remove(edge)
        return best_depth

    best = 0
    for start_vertex_id in incident_edges:
        best = max(best, dfs(start_vertex_id, set(), False))
    return int(best)


def recompute_largest_army(state: GameState) -> None:
    knight_counts = {player_id: player.played_knights for player_id, player in state.players.items()}
    if not knight_counts:
        state.largest_army_owner = None
        state.largest_army_size = 0
        return

    max_knights = max(knight_counts.values())
    contenders = [player_id for player_id, count in knight_counts.items() if count == max_knights and count >= 3]
    current_owner = state.largest_army_owner
    if not contenders:
        state.largest_army_owner = None
        state.largest_army_size = max_knights
        return

    if current_owner in contenders:
        state.largest_army_owner = current_owner
        state.largest_army_size = max_knights
        return

    state.largest_army_owner = contenders[0] if len(contenders) == 1 else None
    state.largest_army_size = max_knights


def recompute_longest_road(state: GameState) -> None:
    lengths: dict[int, int] = {}
    for player_id, player in state.players.items():
        blocked_vertices = {
            vertex_id
            for other_id, other_player in state.players.items()
            if other_id != player_id
            for vertex_id in (other_player.settlements | other_player.cities)
        }
        lengths[player_id] = longest_road_length_for_player(
            state.board,
            player_roads=player.roads,
            blocked_vertices=blocked_vertices,
        )

    if not lengths:
        state.longest_road_owner = None
        state.longest_road_length = 0
        return

    max_length = max(lengths.values())
    contenders = [
        player_id for player_id, length in lengths.items()
        if length == max_length and length >= 5
    ]
    current_owner = state.longest_road_owner
    current_owner_length = lengths.get(current_owner, 0) if current_owner is not None else 0

    if not contenders:
        state.longest_road_owner = None
        state.longest_road_length = max_length
        return

    if current_owner in contenders:
        state.longest_road_owner = current_owner
        state.longest_road_length = max_length
        return

    if current_owner is not None and current_owner_length >= 5:
        better_contenders = [
            player_id for player_id in contenders
            if lengths[player_id] > current_owner_length
        ]
        if not better_contenders:
            state.longest_road_owner = current_owner
            state.longest_road_length = current_owner_length
            return
        top_length = max(lengths[player_id] for player_id in better_contenders)
        top_players = [player_id for player_id in better_contenders if lengths[player_id] == top_length]
        state.longest_road_owner = top_players[0] if len(top_players) == 1 else None
        state.longest_road_length = top_length
        return

    state.longest_road_owner = contenders[0] if len(contenders) == 1 else None
    state.longest_road_length = max_length


def recompute_awards(state: GameState) -> None:
    recompute_longest_road(state)
    recompute_largest_army(state)
