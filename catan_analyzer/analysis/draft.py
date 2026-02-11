from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List

from catan_analyzer.domain.board import BoardState

from .types import DraftPick, VertexScore

ScoreFn = Callable[[int, set[int], list[int]], VertexScore]
ProgressFn = Callable[[int, int], None]
CancelFn = Callable[[], None]


def snake_order(player_count: int) -> List[int]:
    if player_count < 2 or player_count > 4:
        raise ValueError("player_count must be between 2 and 4.")
    return list(range(1, player_count + 1)) + list(range(player_count, 0, -1))


def rank_vertices(
    board: BoardState,
    score_fn: ScoreFn,
    occupied_vertices: Iterable[int] | None = None,
    player_vertices: Iterable[int] | None = None,
    *,
    on_progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> List[VertexScore]:
    occupied = set(occupied_vertices or ())
    player_owned = list(player_vertices or ())
    legal_vertices = board.legal_settlement_vertices(occupied)
    total = len(legal_vertices)
    scores: list[VertexScore] = []
    for index, vertex_id in enumerate(legal_vertices, start=1):
        if cancel_check is not None:
            cancel_check()
        scores.append(score_fn(vertex_id, occupied, player_owned))
        if on_progress is not None:
            on_progress(index, total)
    return sorted(scores, key=lambda item: (-item.total_score, -item.expected_yield, item.vertex_id))


def simulate_draft(
    board: BoardState,
    player_count: int,
    score_fn: ScoreFn,
    *,
    on_progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> List[DraftPick]:
    order = snake_order(player_count)
    occupied: set[int] = set()
    player_placements: Dict[int, list[int]] = defaultdict(list)
    picks: List[DraftPick] = []
    total_turns = len(order)

    for turn_index, player_id in enumerate(order, start=1):
        if cancel_check is not None:
            cancel_check()
        legal_vertices = board.legal_settlement_vertices(occupied)
        candidate_scores: list[VertexScore] = []
        for vertex_id in legal_vertices:
            if cancel_check is not None:
                cancel_check()
            candidate_scores.append(score_fn(vertex_id, occupied, player_placements[player_id]))
        if not candidate_scores:
            break

        chosen = max(candidate_scores, key=lambda score: (score.total_score, score.expected_yield, -score.vertex_id))
        picks.append(
            DraftPick(
                turn_index=turn_index,
                player_id=player_id,
                vertex_id=chosen.vertex_id,
                score_snapshot=chosen,
            )
        )
        occupied.add(chosen.vertex_id)
        player_placements[player_id].append(chosen.vertex_id)
        if on_progress is not None:
            on_progress(turn_index, total_turns)

    return picks
