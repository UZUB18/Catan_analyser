from __future__ import annotations

import math
import multiprocessing
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional

from catan_analyzer.domain.board import BoardState

from .draft import rank_vertices, simulate_draft
from .fullgame_rollout import FullGameAnalyzer
from .mcts_lite import MctsLiteAnalyzer
from .phase_rollout import PhaseRolloutEvaluator
from .runtime import AnalysisRuntime
from .scoring import pip_value, score_vertex
from .seeding import analysis_seed
from .types import (
    AnalysisConfig,
    AnalysisMode,
    AnalysisResult,
    MctsLineExplanation,
    MctsSummary,
    VertexScore,
)


class Analyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> AnalysisResult:
        raise NotImplementedError


class HeuristicAnalyzer(Analyzer):
    def analyze(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> AnalysisResult:
        _validate_config(config)
        if runtime is not None:
            runtime.report_progress("Scoring legal settlement vertices…", 0.02, force=True)
        score_fn = self._make_score_fn(board, config, expected_override=None)
        global_ranking = rank_vertices(
            board,
            score_fn,
            on_progress=(
                (lambda done, total: runtime.report_progress(
                    "Scoring legal settlement vertices…",
                    0.02 + (0.63 * (done / max(1, total))),
                ))
                if runtime is not None
                else None
            ),
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )
        predicted_sequence = simulate_draft(
            board,
            config.player_count,
            score_fn,
            on_progress=(
                (lambda done, total: runtime.report_progress(
                    "Simulating draft order…",
                    0.65 + (0.30 * (done / max(1, total))),
                ))
                if runtime is not None
                else None
            ),
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )
        top_recommendations = global_ranking[: 2 * config.player_count]
        if runtime is not None:
            runtime.report_progress("Heuristic analysis complete.", 1.0, force=True)
        return AnalysisResult(
            global_ranking=global_ranking,
            predicted_sequence=predicted_sequence,
            top_recommendations=top_recommendations,
        )

    def _make_score_fn(
        self,
        board: BoardState,
        config: AnalysisConfig,
        expected_override: Optional[Dict[int, float]],
    ):
        def score_fn(vertex_id: int, occupied: set[int], player_vertices: list[int]) -> VertexScore:
            override = expected_override[vertex_id] if expected_override else None
            score = score_vertex(
                board,
                vertex_id,
                include_ports=config.include_ports,
                occupied_vertices=occupied,
                player_existing_vertices=player_vertices,
                expected_yield_override=override,
                expected_yield_overrides=expected_override,
            )
            if occupied:
                overlap_penalty = _occupied_overlap_penalty(board, vertex_id, occupied)
                score.risk_penalty = round(score.risk_penalty + overlap_penalty, 4)
                score.total_score = round(score.total_score - overlap_penalty, 4)
            return score

        return score_fn


class MonteCarloAnalyzer(HeuristicAnalyzer):
    def analyze(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> AnalysisResult:
        _validate_config(config)
        if runtime is not None:
            runtime.report_progress("Running Monte Carlo yield simulations…", 0.01, force=True)
        expected_override = self._simulate_expected_yield(board, config, runtime=runtime)
        score_fn = self._make_score_fn(board, config, expected_override=expected_override)
        global_ranking = rank_vertices(
            board,
            score_fn,
            on_progress=(
                (lambda done, total: runtime.report_progress(
                    "Building ranking from Monte Carlo estimates…",
                    0.62 + (0.23 * (done / max(1, total))),
                ))
                if runtime is not None
                else None
            ),
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )
        predicted_sequence = simulate_draft(
            board,
            config.player_count,
            score_fn,
            on_progress=(
                (lambda done, total: runtime.report_progress(
                    "Simulating draft order…",
                    0.85 + (0.13 * (done / max(1, total))),
                ))
                if runtime is not None
                else None
            ),
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )
        top_recommendations = global_ranking[: 2 * config.player_count]
        if runtime is not None:
            runtime.report_progress("Monte Carlo analysis complete.", 1.0, force=True)
        return AnalysisResult(
            global_ranking=global_ranking,
            predicted_sequence=predicted_sequence,
            top_recommendations=top_recommendations,
        )

    def _simulate_expected_yield(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> Dict[int, float]:
        base_seed = analysis_seed(board, config.mc_seed, salt="monte_carlo_expected_yield")
        iterations = max(1, int(config.mc_iterations))
        rolls_per_game = max(1, int(config.mc_rolls_per_game))

        contributions_by_roll: dict[int, tuple[tuple[int, int], ...]] = {roll: tuple() for roll in range(2, 13)}
        mutable_contributions: Dict[int, list[tuple[int, int]]] = {roll: [] for roll in range(2, 13)}
        for vertex_id, vertex in board.vertices.items():
            hit_counts: Dict[int, int] = defaultdict(int)
            for tile_id in vertex.adjacent_hex_ids:
                token = board.get_tile(tile_id).token_number
                if token is not None:
                    hit_counts[token] += 1
            for token, count in hit_counts.items():
                mutable_contributions[token].append((vertex_id, count))
        contributions_by_roll = {
            roll: tuple(entries)
            for roll, entries in mutable_contributions.items()
        }

        vertex_ids = tuple(sorted(board.vertices.keys()))
        worker_count = _resolve_worker_count(config, iterations, rolls_per_game)
        if worker_count <= 1:
            totals = _monte_carlo_expected_yield_worker(
                seed=base_seed,
                iterations=iterations,
                rolls_per_game=rolls_per_game,
                contributions_by_roll=contributions_by_roll,
                vertex_ids=vertex_ids,
                progress_callback=(
                    (lambda done, total: runtime.report_progress(
                        "Running Monte Carlo yield simulations…",
                        0.02 + (0.58 * (done / max(1, total))),
                    ))
                    if runtime is not None
                    else None
                ),
                cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
            )
        else:
            totals = _parallel_monte_carlo_expected_yield(
                base_seed=base_seed,
                iterations=iterations,
                rolls_per_game=rolls_per_game,
                worker_count=worker_count,
                contributions_by_roll=contributions_by_roll,
                vertex_ids=vertex_ids,
            )

        scaling = 36.0 / rolls_per_game
        return {
            vertex_id: (total / iterations) * scaling
            for vertex_id, total in totals.items()
        }


class PhaseRolloutAnalyzer(Analyzer):
    def analyze(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> AnalysisResult:
        _validate_config(config)
        if runtime is not None:
            runtime.report_progress("Preparing phase rollout evaluator…", 0.02, force=True)
        evaluator = PhaseRolloutEvaluator(board, config, runtime=runtime)
        score_fn = self._make_score_fn(board, config, evaluator)
        global_ranking = rank_vertices(
            board,
            score_fn,
            on_progress=(
                (lambda done, total: runtime.report_progress(
                    "Evaluating vertices with phase rollouts…",
                    0.04 + (0.76 * (done / max(1, total))),
                ))
                if runtime is not None
                else None
            ),
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )
        predicted_sequence = simulate_draft(
            board,
            config.player_count,
            score_fn,
            on_progress=(
                (lambda done, total: runtime.report_progress(
                    "Simulating draft order with phase scores…",
                    0.80 + (0.18 * (done / max(1, total))),
                ))
                if runtime is not None
                else None
            ),
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )
        top_recommendations = global_ranking[: 2 * config.player_count]
        if runtime is not None:
            runtime.report_progress("Phase rollout analysis complete.", 1.0, force=True)
        return AnalysisResult(
            global_ranking=global_ranking,
            predicted_sequence=predicted_sequence,
            top_recommendations=top_recommendations,
        )

    def _make_score_fn(
        self,
        board: BoardState,
        config: AnalysisConfig,
        evaluator: PhaseRolloutEvaluator,
    ):
        def score_fn(vertex_id: int, occupied: set[int], player_vertices: list[int]) -> VertexScore:
            base = score_vertex(
                board,
                vertex_id,
                include_ports=config.include_ports,
                occupied_vertices=occupied,
                player_existing_vertices=player_vertices,
            )
            phase = evaluator.evaluate_vertex(
                vertex_id,
                occupied_vertices=occupied,
                player_existing_vertices=player_vertices,
            )

            combined = VertexScore(
                vertex_id=vertex_id,
                total_score=round(base.total_score + phase.total_score, 4),
                expected_yield=base.expected_yield,
                diversity_score=base.diversity_score,
                port_score=base.port_score,
                risk_penalty=base.risk_penalty,
                synergy_score=base.synergy_score,
                frontier_score=base.frontier_score,
                best_path_score=base.best_path_score,
                tempo_score=phase.tempo_score,
                recipe_coverage_score=phase.recipe_coverage_score,
                fragility_penalty=phase.fragility_penalty,
                port_conversion_score=phase.port_conversion_score,
                robber_penalty=phase.robber_penalty,
            )
            if occupied:
                overlap_penalty = _occupied_overlap_penalty(board, vertex_id, occupied)
                combined.risk_penalty = round(combined.risk_penalty + overlap_penalty, 4)
                combined.total_score = round(combined.total_score - overlap_penalty, 4)
            return combined

        return score_fn


class HybridOpeningAnalyzer(Analyzer):
    """Blend opening analyzers through normalized weighted consensus."""

    _RANK_BLEND_WEIGHT = 0.35
    _DYNAMIC_HEURISTIC_WEIGHT = 0.62
    _CONSENSUS_SCORE_SCALE = 6.0

    def analyze(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> AnalysisResult:
        _validate_config(config)
        started = time.perf_counter()
        if runtime is not None:
            runtime.report_progress("Starting hybrid consensus analysis…", 0.01, force=True)

        components = self._collect_component_results(board, config, runtime=runtime)
        active_components = [(name, weight, result) for name, weight, result in components if weight > 0.0]
        if not active_components:
            raise ValueError("Hybrid mode requires at least one positive component weight.")

        legal_vertices = sorted(board.legal_settlement_vertices(set()))
        if not legal_vertices:
            if runtime is not None:
                runtime.report_progress("No legal vertices found.", 1.0, force=True)
            return AnalysisResult(
                global_ranking=[],
                predicted_sequence=[],
                top_recommendations=[],
            )
        if runtime is not None:
            runtime.report_progress("Combining component scores…", 0.83)

        normalized_component_scores = {
            name: self._normalize_component_scores(result.global_ranking, legal_vertices)
            for name, _, result in active_components
        }
        component_score_index = {
            name: {score.vertex_id: score for score in result.global_ranking}
            for name, _, result in active_components
        }

        total_weight = sum(weight for _, weight, _ in active_components)
        consensus_before_penalty: dict[int, float] = {}
        stability_penalties: dict[int, float] = {}
        component_breakdown: dict[int, dict[str, float]] = {}

        ranking: list[VertexScore] = []
        total_vertices = max(1, len(legal_vertices))
        for vertex_index, vertex_id in enumerate(legal_vertices, start=1):
            if runtime is not None:
                runtime.raise_if_cancelled()
            weighted_mean = 0.0
            per_component: dict[str, float] = {}
            for name, weight, _ in active_components:
                component_signal = normalized_component_scores[name].get(vertex_id, 0.0)
                per_component[name] = component_signal
                weighted_mean += weight * component_signal
            weighted_mean /= total_weight

            weighted_variance = 0.0
            for name, weight, _ in active_components:
                delta = per_component[name] - weighted_mean
                weighted_variance += weight * (delta * delta)
            weighted_variance /= total_weight
            spread = math.sqrt(max(0.0, weighted_variance))
            stability_penalty = config.hybrid_stability_penalty_weight * spread
            consensus_score = weighted_mean - stability_penalty

            consensus_before_penalty[vertex_id] = weighted_mean
            stability_penalties[vertex_id] = stability_penalty
            component_breakdown[vertex_id] = per_component
            ranking.append(
                self._build_consensus_vertex_score(
                    vertex_id=vertex_id,
                    consensus_score=consensus_score,
                    component_score_index=component_score_index,
                    active_components=active_components,
                )
            )
            if runtime is not None:
                runtime.report_progress(
                    "Combining component scores…",
                    0.83 + (0.08 * (vertex_index / total_vertices)),
                )

        ranking.sort(key=lambda item: (-item.total_score, -item.expected_yield, item.vertex_id))
        top_recommendations = ranking[: 2 * config.player_count]

        score_index = {score.vertex_id: score for score in ranking}

        def score_fn(vertex_id: int, occupied: set[int], player_vertices: list[int]) -> VertexScore:
            base = score_index.get(vertex_id)
            dynamic = score_vertex(
                board,
                vertex_id,
                include_ports=config.include_ports,
                occupied_vertices=occupied,
                player_existing_vertices=player_vertices,
            )
            if base is None:
                return dynamic
            dynamic.total_score = round(
                (self._DYNAMIC_HEURISTIC_WEIGHT * dynamic.total_score)
                + (self._CONSENSUS_SCORE_SCALE * base.total_score),
                4,
            )
            dynamic.tempo_score = base.tempo_score
            dynamic.recipe_coverage_score = base.recipe_coverage_score
            dynamic.fragility_penalty = base.fragility_penalty
            dynamic.port_conversion_score = base.port_conversion_score
            dynamic.robber_penalty = base.robber_penalty
            if occupied:
                overlap_penalty = _occupied_overlap_penalty(board, vertex_id, occupied)
                dynamic.risk_penalty = round(dynamic.risk_penalty + overlap_penalty, 4)
                dynamic.total_score = round(dynamic.total_score - overlap_penalty, 4)
            return dynamic

        predicted_sequence = simulate_draft(
            board,
            config.player_count,
            score_fn,
            on_progress=(
                (lambda done, total: runtime.report_progress(
                    "Simulating hybrid draft sequence…",
                    0.91 + (0.06 * (done / max(1, total))),
                ))
                if runtime is not None
                else None
            ),
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )

        component_names = [name for name, _, _ in active_components]
        mcts_result = next(result for name, _, result in active_components if name == "mcts_lite")
        explain_lines = self._build_explain_lines(
            ranking=ranking,
            component_names=component_names,
            component_breakdown=component_breakdown,
            consensus_before_penalty=consensus_before_penalty,
            stability_penalties=stability_penalties,
            mcts_lines=mcts_result.explain_lines,
        )

        best_score = ranking[0].total_score if ranking else 0.0
        alt_gap = (ranking[0].total_score - ranking[1].total_score) if len(ranking) > 1 else best_score
        mcts_root_visits = (
            mcts_result.mcts_summary.root_visits
            if mcts_result.mcts_summary is not None
            else max(1, config.mcts_iterations)
        )
        summary = MctsSummary(
            root_visits=mcts_root_visits,
            best_line_score=round(best_score, 4),
            alt_line_score_gap=round(alt_gap, 4),
            runtime_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        if runtime is not None:
            runtime.report_progress("Hybrid analysis complete.", 1.0, force=True)

        return AnalysisResult(
            global_ranking=ranking,
            predicted_sequence=predicted_sequence,
            top_recommendations=top_recommendations,
            explain_lines=explain_lines,
            mcts_summary=summary,
        )

    def _collect_component_results(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> list[tuple[str, float, AnalysisResult]]:
        component_specs = [
            ("heuristic", float(config.hybrid_weight_heuristic), AnalysisMode.HEURISTIC.value),
            ("phase_rollout", float(config.hybrid_weight_phase_rollout), AnalysisMode.PHASE_ROLLOUT_MC.value),
            ("mcts_lite", float(config.hybrid_weight_mcts_lite), AnalysisMode.MCTS_LITE_OPENING.value),
        ]
        if config.hybrid_include_monte_carlo:
            component_specs.append(
                ("monte_carlo", float(config.hybrid_weight_monte_carlo), AnalysisMode.MONTE_CARLO.value)
            )

        return _collect_hybrid_component_results(board, config, component_specs, runtime=runtime)

    def _normalize_component_scores(
        self,
        ranking: list[VertexScore],
        legal_vertices: list[int],
    ) -> dict[int, float]:
        if not legal_vertices:
            return {}
        if not ranking:
            return {vertex_id: 0.5 for vertex_id in legal_vertices}

        score_by_vertex = {score.vertex_id: score.total_score for score in ranking}
        rank_by_vertex = {score.vertex_id: index for index, score in enumerate(ranking)}

        scores = [score_by_vertex.get(vertex_id, 0.0) for vertex_id in legal_vertices]
        min_score = min(scores)
        max_score = max(scores)
        span = max_score - min_score
        rank_denominator = max(1, len(legal_vertices) - 1)

        normalized: dict[int, float] = {}
        for vertex_id in legal_vertices:
            raw_total = score_by_vertex.get(vertex_id, min_score)
            if span <= 1e-9:
                normalized_total = 0.5
            else:
                normalized_total = (raw_total - min_score) / span

            rank_position = rank_by_vertex.get(vertex_id, len(legal_vertices) - 1)
            normalized_rank = 1.0 - (rank_position / rank_denominator)
            normalized[vertex_id] = (
                ((1.0 - self._RANK_BLEND_WEIGHT) * normalized_total)
                + (self._RANK_BLEND_WEIGHT * normalized_rank)
            )

        return normalized

    def _build_consensus_vertex_score(
        self,
        *,
        vertex_id: int,
        consensus_score: float,
        component_score_index: dict[str, dict[int, VertexScore]],
        active_components: list[tuple[str, float, AnalysisResult]],
    ) -> VertexScore:
        return VertexScore(
            vertex_id=vertex_id,
            total_score=round(consensus_score, 4),
            expected_yield=round(
                self._weighted_metric("expected_yield", vertex_id, component_score_index, active_components), 4
            ),
            diversity_score=round(
                self._weighted_metric("diversity_score", vertex_id, component_score_index, active_components), 4
            ),
            port_score=round(
                self._weighted_metric("port_score", vertex_id, component_score_index, active_components), 4
            ),
            risk_penalty=round(
                self._weighted_metric("risk_penalty", vertex_id, component_score_index, active_components), 4
            ),
            synergy_score=round(
                self._weighted_metric("synergy_score", vertex_id, component_score_index, active_components), 4
            ),
            frontier_score=round(
                self._weighted_metric("frontier_score", vertex_id, component_score_index, active_components), 4
            ),
            best_path_score=round(
                self._weighted_metric("best_path_score", vertex_id, component_score_index, active_components), 4
            ),
            tempo_score=round(
                self._weighted_metric("tempo_score", vertex_id, component_score_index, active_components), 4
            ),
            recipe_coverage_score=round(
                self._weighted_metric("recipe_coverage_score", vertex_id, component_score_index, active_components), 4
            ),
            fragility_penalty=round(
                self._weighted_metric("fragility_penalty", vertex_id, component_score_index, active_components), 4
            ),
            port_conversion_score=round(
                self._weighted_metric("port_conversion_score", vertex_id, component_score_index, active_components), 4
            ),
            robber_penalty=round(
                self._weighted_metric("robber_penalty", vertex_id, component_score_index, active_components), 4
            ),
        )

    @staticmethod
    def _weighted_metric(
        metric_name: str,
        vertex_id: int,
        component_score_index: dict[str, dict[int, VertexScore]],
        active_components: list[tuple[str, float, AnalysisResult]],
    ) -> float:
        total_weight = 0.0
        weighted_value = 0.0
        for component_name, component_weight, _ in active_components:
            score = component_score_index[component_name].get(vertex_id)
            if score is None:
                continue
            weighted_value += component_weight * float(getattr(score, metric_name, 0.0))
            total_weight += component_weight

        if total_weight <= 0.0:
            return 0.0
        return weighted_value / total_weight

    def _build_explain_lines(
        self,
        *,
        ranking: list[VertexScore],
        component_names: list[str],
        component_breakdown: dict[int, dict[str, float]],
        consensus_before_penalty: dict[int, float],
        stability_penalties: dict[int, float],
        mcts_lines: list[MctsLineExplanation],
    ) -> list[MctsLineExplanation]:
        component_label = {
            "heuristic": "H",
            "phase_rollout": "P",
            "mcts_lite": "M",
            "monte_carlo": "MC",
        }
        lines: list[MctsLineExplanation] = []
        explain_limit = min(4, len(ranking))
        for rank_index, score in enumerate(ranking[:explain_limit], start=1):
            vertex_id = score.vertex_id
            per_component = component_breakdown[vertex_id]
            breakdown_text = " ".join(
                f"{component_label[name]}:{per_component.get(name, 0.0):.2f}"
                for name in component_names
            )
            lines.append(
                MctsLineExplanation(
                    ply_index=rank_index,
                    actor=1,
                    action=f"HYB#{rank_index} S{vertex_id} {breakdown_text}",
                    self_value=round(consensus_before_penalty[vertex_id], 4),
                    blocking_delta=round(stability_penalties[vertex_id], 4),
                    uct_value=round(score.total_score, 4),
                    visits=max(1, int(round(consensus_before_penalty[vertex_id] * 1_000))),
                )
            )

        for offset, line in enumerate(mcts_lines[:2], start=1):
            lines.append(
                MctsLineExplanation(
                    ply_index=200 + offset,
                    actor=line.actor,
                    action=f"MCTS {line.action}",
                    self_value=line.self_value,
                    blocking_delta=line.blocking_delta,
                    uct_value=line.uct_value,
                    visits=line.visits,
                )
            )

        return lines


_MONTE_CARLO_PARALLEL_MIN_WORK = 200_000
_HYBRID_PARALLEL_MIN_WORK = 3_000


def _resolve_worker_count(config: AnalysisConfig, iterations: int, rolls_per_game: int) -> int:
    if multiprocessing.current_process().daemon:
        return 1

    total_work = iterations * rolls_per_game
    if total_work < _MONTE_CARLO_PARALLEL_MIN_WORK:
        return 1

    available = max(1, os.cpu_count() or 1)
    configured = max(0, int(config.parallel_workers))
    if configured == 0:
        workers = available
    else:
        workers = min(configured, available)

    return max(1, workers)


def _resolve_parallel_worker_budget(config: AnalysisConfig) -> int:
    available = max(1, os.cpu_count() or 1)
    configured = max(0, int(config.parallel_workers))
    if configured == 0:
        return available
    return min(configured, available)


def _collect_hybrid_component_results(
    board: BoardState,
    config: AnalysisConfig,
    component_specs: list[tuple[str, float, str]],
    runtime: AnalysisRuntime | None = None,
) -> list[tuple[str, float, AnalysisResult]]:
    if runtime is not None:
        return _collect_hybrid_component_results_sequential(board, config, component_specs, runtime=runtime)

    if not _should_parallelize_hybrid(config):
        return _collect_hybrid_component_results_sequential(board, config, component_specs)

    weighted_specs = [spec for spec in component_specs if spec[1] > 0.0]
    if len(weighted_specs) <= 1:
        return _collect_hybrid_component_results_sequential(board, config, component_specs)

    if multiprocessing.current_process().daemon:
        return _collect_hybrid_component_results_sequential(board, config, component_specs)

    max_workers = min(_resolve_parallel_worker_budget(config), len(weighted_specs))
    if max_workers <= 1:
        return _collect_hybrid_component_results_sequential(board, config, component_specs)

    futures = []
    weighted_results: dict[str, AnalysisResult] = {}
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for component_name, _component_weight, mode_value in weighted_specs:
                futures.append(
                    (
                        component_name,
                        executor.submit(_run_analysis_mode, mode_value, board, config),
                    )
                )
            for component_name, future in futures:
                weighted_results[component_name] = future.result()
    except Exception:
        return _collect_hybrid_component_results_sequential(board, config, component_specs)

    ordered_results: list[tuple[str, float, AnalysisResult]] = []
    for component_name, component_weight, mode_value in component_specs:
        if component_weight > 0.0 and component_name in weighted_results:
            ordered_results.append((component_name, component_weight, weighted_results[component_name]))
        else:
            ordered_results.append(
                (
                    component_name,
                    component_weight,
                    _run_analysis_mode(mode_value, board, config),
                )
            )
    return ordered_results


def _collect_hybrid_component_results_sequential(
    board: BoardState,
    config: AnalysisConfig,
    component_specs: list[tuple[str, float, str]],
    runtime: AnalysisRuntime | None = None,
) -> list[tuple[str, float, AnalysisResult]]:
    if runtime is None:
        return [
            (
                component_name,
                component_weight,
                _run_analysis_mode(mode_value, board, config),
            )
            for component_name, component_weight, mode_value in component_specs
        ]

    results: list[tuple[str, float, AnalysisResult]] = []
    weighted_specs = [spec for spec in component_specs if spec[1] > 0.0]
    total_weighted = max(1, len(weighted_specs))
    stage_names = {
        AnalysisMode.HEURISTIC.value: "heuristic component",
        AnalysisMode.PHASE_ROLLOUT_MC.value: "phase rollout component",
        AnalysisMode.MCTS_LITE_OPENING.value: "MCTS-lite component",
        AnalysisMode.MONTE_CARLO.value: "Monte Carlo component",
    }

    completed_weighted = 0
    for component_name, component_weight, mode_value in component_specs:
        runtime.raise_if_cancelled()
        if component_weight <= 0.0:
            results.append(
                (
                    component_name,
                    component_weight,
                    AnalysisResult(global_ranking=[], predicted_sequence=[], top_recommendations=[]),
                )
            )
            continue

        segment_start = 0.04 + (0.76 * (completed_weighted / total_weighted))
        segment_end = 0.04 + (0.76 * ((completed_weighted + 1) / total_weighted))
        stage_label = stage_names.get(mode_value, f"{component_name} component")
        component_runtime = runtime.subrange(segment_start, segment_end, stage_prefix=f"Running {stage_label}: ")
        component_result = _run_analysis_mode(mode_value, board, config, runtime=component_runtime)
        results.append((component_name, component_weight, component_result))
        completed_weighted += 1

    return results


def _run_analysis_mode(
    mode_value: str,
    board: BoardState,
    config: AnalysisConfig,
    runtime: AnalysisRuntime | None = None,
) -> AnalysisResult:
    analyzer = create_analyzer(mode_value)
    return analyzer.analyze(board, config, runtime=runtime)


def _should_parallelize_hybrid(config: AnalysisConfig) -> bool:
    total_work_estimate = (
        max(1, int(config.phase_rollout_count)) * max(1, int(config.phase_turn_horizon))
        + max(1, int(config.mcts_iterations)) * max(1, int(config.mcts_max_plies))
    )
    if config.hybrid_include_monte_carlo:
        total_work_estimate += max(1, int(config.mc_iterations)) * max(1, int(config.mc_rolls_per_game))
    return total_work_estimate >= _HYBRID_PARALLEL_MIN_WORK


def _parallel_monte_carlo_expected_yield(
    *,
    base_seed: int,
    iterations: int,
    rolls_per_game: int,
    worker_count: int,
    contributions_by_roll: dict[int, tuple[tuple[int, int], ...]],
    vertex_ids: tuple[int, ...],
) -> dict[int, float]:
    chunks = _split_iterations(iterations, worker_count)
    if len(chunks) <= 1:
        return _monte_carlo_expected_yield_worker(
            seed=base_seed,
            iterations=iterations,
            rolls_per_game=rolls_per_game,
            contributions_by_roll=contributions_by_roll,
            vertex_ids=vertex_ids,
        )

    futures = []
    totals = {vertex_id: 0.0 for vertex_id in vertex_ids}
    try:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            for worker_index, chunk_iterations in enumerate(chunks):
                worker_seed = base_seed + (worker_index + 1) * 104_729
                futures.append(
                    executor.submit(
                        _monte_carlo_expected_yield_worker,
                        worker_seed,
                        chunk_iterations,
                        rolls_per_game,
                        contributions_by_roll,
                        vertex_ids,
                    )
                )
            for future in futures:
                partial_totals = future.result()
                for vertex_id, value in partial_totals.items():
                    totals[vertex_id] += value
        return totals
    except Exception:
        return _monte_carlo_expected_yield_worker(
            seed=base_seed,
            iterations=iterations,
            rolls_per_game=rolls_per_game,
            contributions_by_roll=contributions_by_roll,
            vertex_ids=vertex_ids,
        )


def _split_iterations(total_iterations: int, worker_count: int) -> list[int]:
    workers = max(1, worker_count)
    base = total_iterations // workers
    remainder = total_iterations % workers
    chunks = [
        base + (1 if worker_index < remainder else 0)
        for worker_index in range(workers)
    ]
    return [chunk for chunk in chunks if chunk > 0]


def _monte_carlo_expected_yield_worker(
    seed: int,
    iterations: int,
    rolls_per_game: int,
    contributions_by_roll: dict[int, tuple[tuple[int, int], ...]],
    vertex_ids: tuple[int, ...],
    progress_callback=None,
    cancel_check=None,
) -> dict[int, float]:
    rng = random.Random(seed)
    totals = {vertex_id: 0.0 for vertex_id in vertex_ids}
    randint = rng.randint

    progress_step = max(1, iterations // 120)
    for iteration_index in range(iterations):
        if cancel_check is not None:
            cancel_check()
        for _ in range(rolls_per_game):
            roll = randint(1, 6) + randint(1, 6)
            for vertex_id, contribution in contributions_by_roll[roll]:
                totals[vertex_id] += contribution
        if progress_callback is not None and (
            iteration_index == iterations - 1 or (iteration_index + 1) % progress_step == 0
        ):
            progress_callback(iteration_index + 1, iterations)
    return totals


def create_analyzer(mode: AnalysisMode | str) -> Analyzer:
    normalized = AnalysisMode(mode)
    if normalized is AnalysisMode.FULL_GAME:
        return FullGameAnalyzer()
    if normalized is AnalysisMode.HYBRID_OPENING:
        return HybridOpeningAnalyzer()
    if normalized is AnalysisMode.MCTS_LITE_OPENING:
        return MctsLiteAnalyzer()
    if normalized is AnalysisMode.PHASE_ROLLOUT_MC:
        return PhaseRolloutAnalyzer()
    if normalized is AnalysisMode.MONTE_CARLO:
        return MonteCarloAnalyzer()
    return HeuristicAnalyzer()


def _validate_config(config: AnalysisConfig) -> None:
    if config.player_count < 2 or config.player_count > 4:
        raise ValueError("player_count must be between 2 and 4.")
    if config.mc_iterations < 1:
        raise ValueError("mc_iterations must be >= 1.")
    if config.mc_rolls_per_game < 1:
        raise ValueError("mc_rolls_per_game must be >= 1.")
    if config.phase_rollout_count < 1:
        raise ValueError("phase_rollout_count must be >= 1.")
    if config.phase_turn_horizon < 1:
        raise ValueError("phase_turn_horizon must be >= 1.")
    if config.mcts_iterations < 1:
        raise ValueError("mcts_iterations must be >= 1.")
    if config.mcts_max_plies < 1:
        raise ValueError("mcts_max_plies must be >= 1.")
    if config.mcts_candidate_settlements < 1:
        raise ValueError("mcts_candidate_settlements must be >= 1.")
    if config.mcts_candidate_road_directions < 1:
        raise ValueError("mcts_candidate_road_directions must be >= 1.")
    if not (0.0 <= config.opponent_block_weight <= 1.0):
        raise ValueError("opponent_block_weight must be between 0.0 and 1.0.")
    if config.hybrid_stability_penalty_weight < 0.0:
        raise ValueError("hybrid_stability_penalty_weight must be >= 0.0.")
    hybrid_weights = (
        config.hybrid_weight_heuristic,
        config.hybrid_weight_phase_rollout,
        config.hybrid_weight_mcts_lite,
    )
    if any(weight < 0.0 for weight in hybrid_weights):
        raise ValueError("Hybrid component weights must be >= 0.0.")
    if config.hybrid_weight_heuristic <= 0.0:
        raise ValueError("hybrid_weight_heuristic must be > 0.0.")
    if config.hybrid_weight_phase_rollout <= 0.0:
        raise ValueError("hybrid_weight_phase_rollout must be > 0.0.")
    if config.hybrid_weight_mcts_lite <= 0.0:
        raise ValueError("hybrid_weight_mcts_lite must be > 0.0.")
    if config.hybrid_weight_monte_carlo < 0.0:
        raise ValueError("hybrid_weight_monte_carlo must be >= 0.0.")
    if config.hybrid_include_monte_carlo and config.hybrid_weight_monte_carlo <= 0.0:
        raise ValueError("hybrid_weight_monte_carlo must be > 0.0 when monte is enabled.")
    if config.parallel_workers < 0:
        raise ValueError("parallel_workers must be >= 0.")
    if config.full_game_rollouts < 1:
        raise ValueError("full_game_rollouts must be >= 1.")
    if config.full_game_max_turns < 20:
        raise ValueError("full_game_max_turns must be >= 20.")
    if config.full_game_candidate_vertices < 1:
        raise ValueError("full_game_candidate_vertices must be >= 1.")
    if config.full_game_trade_offer_limit < 0:
        raise ValueError("full_game_trade_offer_limit must be >= 0.")


def _occupied_overlap_penalty(board: BoardState, vertex_id: int, occupied: set[int]) -> float:
    target_hexes = set(board.vertices[vertex_id].adjacent_hex_ids)
    overlap = 0
    for occupied_vertex_id in occupied:
        shared_hexes = target_hexes.intersection(board.vertices[occupied_vertex_id].adjacent_hex_ids)
        overlap += sum(pip_value(board.get_tile(tile_id).token_number) for tile_id in shared_hexes)
    return overlap * 0.08
