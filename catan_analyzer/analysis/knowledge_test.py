from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .types import VertexScore


@dataclass(frozen=True)
class KnowledgeTestEvaluation:
    top_n: int
    user_picks: tuple[int, ...]
    top_vertices: tuple[int, ...]
    hits: tuple[int, ...]
    missed_top_vertices: tuple[int, ...]
    extra_vertices: tuple[int, ...]
    average_rank: float | None
    numeric_score: float
    letter_grade: str

    @property
    def hit_count(self) -> int:
        return len(self.hits)


def evaluate_user_settlement_picks(
    user_picks: Sequence[int],
    ranking: Sequence[VertexScore],
    *,
    top_n: int = 4,
) -> KnowledgeTestEvaluation:
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    unique_user_picks: list[int] = []
    seen: set[int] = set()
    for raw_vertex_id in user_picks:
        vertex_id = int(raw_vertex_id)
        if vertex_id in seen:
            continue
        seen.add(vertex_id)
        unique_user_picks.append(vertex_id)

    top_vertices = tuple(score.vertex_id for score in ranking[:top_n])
    top_vertex_set = set(top_vertices)
    rank_lookup = {score.vertex_id: index for index, score in enumerate(ranking, start=1)}

    hits = tuple(vertex_id for vertex_id in unique_user_picks if vertex_id in top_vertex_set)
    missed = tuple(vertex_id for vertex_id in top_vertices if vertex_id not in seen)
    extra = tuple(vertex_id for vertex_id in unique_user_picks if vertex_id not in top_vertex_set)

    ranked_user_picks = [rank_lookup[vertex_id] for vertex_id in unique_user_picks if vertex_id in rank_lookup]
    average_rank = (
        sum(ranked_user_picks) / len(ranked_user_picks)
        if ranked_user_picks
        else None
    )
    numeric_score = _knowledge_test_numeric_score(
        user_picks=unique_user_picks,
        rank_lookup=rank_lookup,
        top_n=top_n,
        hit_count=len(hits),
    )

    return KnowledgeTestEvaluation(
        top_n=top_n,
        user_picks=tuple(unique_user_picks),
        top_vertices=top_vertices,
        hits=hits,
        missed_top_vertices=missed,
        extra_vertices=extra,
        average_rank=average_rank,
        numeric_score=numeric_score,
        letter_grade=_letter_grade(numeric_score),
    )


def _knowledge_test_numeric_score(
    *,
    user_picks: Sequence[int],
    rank_lookup: dict[int, int],
    top_n: int,
    hit_count: int,
) -> float:
    hit_component = 70.0 * (hit_count / top_n)
    rank_component = 30.0 * _average_rank_quality(
        user_picks=user_picks,
        rank_lookup=rank_lookup,
        top_n=top_n,
    )
    return round(hit_component + rank_component, 1)


def _average_rank_quality(
    *,
    user_picks: Sequence[int],
    rank_lookup: dict[int, int],
    top_n: int,
) -> float:
    if top_n <= 0:
        return 0.0

    quality_sum = 0.0
    picks_to_score = list(user_picks[:top_n])
    if len(picks_to_score) < top_n:
        picks_to_score.extend([-1] * (top_n - len(picks_to_score)))

    for vertex_id in picks_to_score:
        rank = rank_lookup.get(vertex_id)
        if rank is None:
            continue
        if rank <= top_n:
            quality_sum += 1.0
        else:
            quality_sum += top_n / rank

    return quality_sum / top_n


def _letter_grade(score: float) -> str:
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"
