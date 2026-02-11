from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

from catan_analyzer.domain.board import BoardState, Resource

from .blocking import blocking_externality_delta, best_expansion_snapshot, static_vertex_value
from .draft import simulate_draft, snake_order
from .phase_rollout import PhaseRolloutEvaluator
from .runtime import AnalysisRuntime
from .scoring import score_vertex
from .seeding import analysis_seed
from .topology import TOPOLOGY_DISTANCE_LAMBDA, build_settlement_blocked_vertices, candidate_expansion_vertices, road_distance_map
from .types import (
    AnalysisConfig,
    AnalysisResult,
    MctsLineExplanation,
    MctsSummary,
    VertexScore,
)


@dataclass(frozen=True)
class OpeningAction:
    kind: str
    actor: int
    vertex_id: int | None = None
    road_target_id: int | None = None
    road_distance: int = 0
    self_gain: float = 0.0
    blocking_delta: float = 0.0
    description: str = ""


@dataclass
class OpeningState:
    turn_index: int
    phase: str
    ply_index: int
    occupied: set[int]
    player_vertices: dict[int, list[int]]
    player_road_targets: dict[int, list[int]]
    pending_actor: int | None = None
    pending_settlement_vertex: int | None = None
    blocking_harm_to_focal: float = 0.0

    def clone(self) -> "OpeningState":
        return OpeningState(
            turn_index=self.turn_index,
            phase=self.phase,
            ply_index=self.ply_index,
            occupied=set(self.occupied),
            player_vertices={player_id: list(vertices) for player_id, vertices in self.player_vertices.items()},
            player_road_targets={
                player_id: list(targets) for player_id, targets in self.player_road_targets.items()
            },
            pending_actor=self.pending_actor,
            pending_settlement_vertex=self.pending_settlement_vertex,
            blocking_harm_to_focal=self.blocking_harm_to_focal,
        )

    def key(self) -> tuple:
        return (
            self.turn_index,
            self.phase,
            self.ply_index,
            tuple(sorted(self.occupied)),
            tuple((player_id, tuple(self.player_vertices[player_id])) for player_id in sorted(self.player_vertices)),
            tuple((player_id, tuple(self.player_road_targets[player_id])) for player_id in sorted(self.player_road_targets)),
            self.pending_actor,
            self.pending_settlement_vertex,
            round(self.blocking_harm_to_focal, 4),
        )


@dataclass
class TreeNode:
    state: OpeningState
    parent: "TreeNode | None" = None
    action: OpeningAction | None = None
    children: list["TreeNode"] = field(default_factory=list)
    untried_actions: list[OpeningAction] | None = None
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits


class MctsLiteAnalyzer:
    """Tree search over opening settlements + road-direction intent."""

    def analyze(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> AnalysisResult:
        started = time.perf_counter()
        if runtime is not None:
            runtime.report_progress("Initializing MCTS-lite analysis…", 0.01, force=True)
        focal_player_id = 1
        turn_order = snake_order(config.player_count)
        max_plies = min(max(1, config.mcts_max_plies), len(turn_order) * 2)
        rng = random.Random(analysis_seed(board, config.mc_seed, salt="mcts_lite_opening"))

        phase_evaluator = PhaseRolloutEvaluator(board, config)
        state_eval_cache: dict[tuple, float] = {}

        root_state = OpeningState(
            turn_index=0,
            phase="settlement",
            ply_index=0,
            occupied=set(),
            player_vertices={player_id: [] for player_id in range(1, config.player_count + 1)},
            player_road_targets={player_id: [] for player_id in range(1, config.player_count + 1)},
        )
        root = TreeNode(state=root_state)

        iteration_count = max(1, int(config.mcts_iterations))
        for iteration_index in range(iteration_count):
            if runtime is not None:
                runtime.raise_if_cancelled()
            node = root
            # selection
            while True:
                if self._is_terminal(node.state, turn_order, max_plies):
                    break
                if node.untried_actions is None:
                    node.untried_actions = self._generate_actions(
                        board,
                        node.state,
                        config,
                        turn_order,
                        focal_player_id=focal_player_id,
                    )
                if node.untried_actions:
                    break
                if not node.children:
                    break
                node = self._select_child(
                    node,
                    config=config,
                    turn_order=turn_order,
                    focal_player_id=focal_player_id,
                )

            # expansion
            if not self._is_terminal(node.state, turn_order, max_plies):
                if node.untried_actions is None:
                    node.untried_actions = self._generate_actions(
                        board,
                        node.state,
                        config,
                        turn_order,
                        focal_player_id=focal_player_id,
                    )
                if node.untried_actions:
                    pick_span = min(3, len(node.untried_actions))
                    pick_index = 0 if pick_span <= 1 else rng.randrange(pick_span)
                    action = node.untried_actions.pop(pick_index)
                    child_state = self._apply_action(
                        board,
                        node.state,
                        action,
                        config=config,
                        focal_player_id=focal_player_id,
                    )
                    child = TreeNode(state=child_state, parent=node, action=action)
                    node.children.append(child)
                    node = child

            # simulation
            simulation_state = node.state.clone()
            simulation_value = self._rollout(
                board,
                simulation_state,
                config=config,
                turn_order=turn_order,
                max_plies=max_plies,
                focal_player_id=focal_player_id,
                phase_evaluator=phase_evaluator,
                state_eval_cache=state_eval_cache,
                rng=rng,
            )

            # backpropagation
            cursor: TreeNode | None = node
            while cursor is not None:
                cursor.visits += 1
                cursor.value_sum += simulation_value
                cursor = cursor.parent

            if runtime is not None:
                runtime.report_progress(
                    "Running MCTS-lite search…",
                    0.06 + (0.66 * ((iteration_index + 1) / iteration_count)),
                )

        root_settlement_children = [
            child
            for child in root.children
            if child.action is not None and child.action.kind == "settlement"
        ]
        root_stats: dict[int, TreeNode] = {
            child.action.vertex_id: child
            for child in root_settlement_children
            if child.action and child.action.vertex_id is not None
        }

        ranking = self._build_ranking(
            board,
            config=config,
            root_stats=root_stats,
            phase_evaluator=phase_evaluator,
            runtime=runtime,
        )

        top_recommendations = ranking[: 2 * config.player_count]
        if runtime is not None:
            runtime.report_progress("Selecting top recommendations…", 0.78)

        score_index = {score.vertex_id: score for score in ranking}

        def score_fn(vertex_id: int, occupied: set[int], player_vertices: list[int]) -> VertexScore:
            base = score_index.get(vertex_id)
            if base is None:
                return score_vertex(
                    board,
                    vertex_id,
                    include_ports=config.include_ports,
                    occupied_vertices=occupied,
                    player_existing_vertices=player_vertices,
                )
            score = score_vertex(
                board,
                vertex_id,
                include_ports=config.include_ports,
                occupied_vertices=occupied,
                player_existing_vertices=player_vertices,
            )
            score.total_score = round(score.total_score + (0.45 * base.total_score), 4)
            score.tempo_score = base.tempo_score
            score.recipe_coverage_score = base.recipe_coverage_score
            score.fragility_penalty = base.fragility_penalty
            score.port_conversion_score = base.port_conversion_score
            score.robber_penalty = base.robber_penalty
            return score

        predicted_sequence = simulate_draft(
            board,
            config.player_count,
            score_fn,
            on_progress=(
                (lambda done, total: runtime.report_progress(
                    "Simulating opening draft…",
                    0.78 + (0.14 * (done / max(1, total))),
                ))
                if runtime is not None
                else None
            ),
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )
        explain_lines = self._extract_explain_lines(
            root,
            config=config,
            turn_order=turn_order,
            focal_player_id=focal_player_id,
        )
        if runtime is not None:
            runtime.report_progress("Preparing explainability output…", 0.96)

        best_root_mean = max((child.mean_value for child in root_settlement_children), default=0.0)
        second_best_root_mean = sorted((child.mean_value for child in root_settlement_children), reverse=True)
        alt_gap = (
            best_root_mean - second_best_root_mean[1]
            if len(second_best_root_mean) > 1
            else best_root_mean
        )
        runtime_ms = (time.perf_counter() - started) * 1000.0
        summary = MctsSummary(
            root_visits=root.visits,
            best_line_score=round(best_root_mean, 4),
            alt_line_score_gap=round(alt_gap, 4),
            runtime_ms=round(runtime_ms, 2),
        )

        if runtime is not None:
            runtime.report_progress("MCTS-lite analysis complete.", 1.0, force=True)

        return AnalysisResult(
            global_ranking=ranking,
            predicted_sequence=predicted_sequence,
            top_recommendations=top_recommendations,
            explain_lines=explain_lines,
            mcts_summary=summary,
        )

    def _build_ranking(
        self,
        board: BoardState,
        *,
        config: AnalysisConfig,
        root_stats: Mapping[int, TreeNode],
        phase_evaluator: PhaseRolloutEvaluator,
        runtime: AnalysisRuntime | None = None,
    ) -> list[VertexScore]:
        legal_vertices = board.legal_settlement_vertices(set())
        max_root_visits = max((node.visits for node in root_stats.values()), default=1)

        preliminary: list[tuple[int, float, VertexScore]] = []
        total_vertices = max(1, len(legal_vertices))
        for index, vertex_id in enumerate(legal_vertices, start=1):
            if runtime is not None:
                runtime.raise_if_cancelled()
            base = score_vertex(
                board,
                vertex_id,
                include_ports=config.include_ports,
                occupied_vertices=(),
                player_existing_vertices=(),
            )
            root_node = root_stats.get(vertex_id)
            mean_value = root_node.mean_value if root_node else 0.0
            visit_boost = (root_node.visits / max_root_visits) if root_node else 0.0
            first_pick_penalty = self._first_settlement_port_penalty(board, vertex_id, base)
            preliminary_total = base.total_score + (0.5 * mean_value) + (1.6 * visit_boost) - first_pick_penalty
            preliminary.append((vertex_id, preliminary_total, base))
            if runtime is not None:
                runtime.report_progress(
                    "Scoring vertices from MCTS search…",
                    0.72 + (0.12 * (index / total_vertices)),
                )

        preliminary.sort(key=lambda item: item[1], reverse=True)
        detailed_ids = {vertex_id for vertex_id, *_ in preliminary[: max(14, 2 * config.player_count + 8)]}

        ranking: list[VertexScore] = []
        detail_total = max(1, len(preliminary))
        for detail_index, (vertex_id, preliminary_total, base) in enumerate(preliminary, start=1):
            if runtime is not None:
                runtime.raise_if_cancelled()
            root_node = root_stats.get(vertex_id)
            mean_value = root_node.mean_value if root_node else 0.0
            visit_boost = (root_node.visits / max_root_visits) if root_node else 0.0

            if vertex_id in detailed_ids:
                phase = phase_evaluator.evaluate_vertex(
                    vertex_id,
                    occupied_vertices=set(),
                    player_existing_vertices=(),
                )
                first_pick_penalty = self._first_settlement_port_penalty(board, vertex_id, base)
                total = (
                    base.total_score
                    + phase.total_score
                    + (0.25 * mean_value)
                    + (1.1 * visit_boost)
                    - first_pick_penalty
                )
                score = VertexScore(
                    vertex_id=vertex_id,
                    total_score=round(total, 4),
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
            else:
                score = VertexScore(
                    vertex_id=vertex_id,
                    total_score=round(preliminary_total, 4),
                    expected_yield=base.expected_yield,
                    diversity_score=base.diversity_score,
                    port_score=base.port_score,
                    risk_penalty=base.risk_penalty,
                    synergy_score=base.synergy_score,
                    frontier_score=base.frontier_score,
                    best_path_score=base.best_path_score,
                )
            ranking.append(score)
            if runtime is not None and detail_index % 6 == 0:
                runtime.report_progress(
                    "Refining ranking details…",
                    0.84 + (0.06 * (detail_index / detail_total)),
                )

        return sorted(ranking, key=lambda item: (-item.total_score, -item.expected_yield, item.vertex_id))

    def _extract_explain_lines(
        self,
        root: TreeNode,
        *,
        config: AnalysisConfig,
        turn_order: list[int],
        focal_player_id: int,
    ) -> list[MctsLineExplanation]:
        lines: list[MctsLineExplanation] = []
        node = root
        ply = 1
        while node.children and ply <= min(config.mcts_max_plies, len(turn_order) * 2):
            actor = self._actor_for_state(node.state, turn_order)
            children = [child for child in node.children if child.action is not None]
            if not children:
                break
            if actor == focal_player_id:
                best_child = max(children, key=lambda child: (child.visits, child.mean_value))
            else:
                block_weight = float(config.opponent_block_weight)
                best_child = max(
                    children,
                    key=lambda child: (
                        ((1.0 - block_weight) * child.action.self_gain)
                        + (block_weight * child.action.blocking_delta),
                        child.visits,
                    ),
                )

            action = best_child.action
            if action is None:
                break
            lines.append(
                MctsLineExplanation(
                    ply_index=ply,
                    actor=action.actor,
                    action=action.description or action.kind,
                    self_value=round(action.self_gain, 4),
                    blocking_delta=round(action.blocking_delta, 4),
                    uct_value=round(best_child.mean_value, 4),
                    visits=best_child.visits,
                )
            )
            node = best_child
            ply += 1

        # Add top alternatives at root for context.
        root_alternatives = [
            child for child in root.children if child.action is not None and child.action.kind == "settlement"
        ]
        root_alternatives.sort(key=lambda child: (child.mean_value, child.visits), reverse=True)
        for alt_rank, child in enumerate(root_alternatives[1:3], start=1):
            action = child.action
            if action is None or action.vertex_id is None:
                continue
            lines.append(
                MctsLineExplanation(
                    ply_index=100 + alt_rank,
                    actor=action.actor,
                    action=f"ALT#{alt_rank} S{action.vertex_id}",
                    self_value=round(action.self_gain, 4),
                    blocking_delta=round(action.blocking_delta, 4),
                    uct_value=round(child.mean_value, 4),
                    visits=child.visits,
                )
            )

        return lines

    def _rollout(
        self,
        board: BoardState,
        state: OpeningState,
        *,
        config: AnalysisConfig,
        turn_order: list[int],
        max_plies: int,
        focal_player_id: int,
        phase_evaluator: PhaseRolloutEvaluator,
        state_eval_cache: dict[tuple, float],
        rng: random.Random,
    ) -> float:
        rollout_state = state.clone()
        while not self._is_terminal(rollout_state, turn_order, max_plies):
            actions = self._generate_actions(
                board,
                rollout_state,
                config,
                turn_order,
                focal_player_id=focal_player_id,
            )
            if not actions:
                break
            actor = self._actor_for_state(rollout_state, turn_order)
            if actor == focal_player_id:
                action = self._sample_best(actions, key_fn=lambda item: item.self_gain, rng=rng)
            else:
                block_weight = float(config.opponent_block_weight)
                action = self._sample_best(
                    actions,
                    key_fn=lambda item: ((1.0 - block_weight) * item.self_gain) + (block_weight * item.blocking_delta),
                    rng=rng,
                )
            rollout_state = self._apply_action(
                board,
                rollout_state,
                action,
                config=config,
                focal_player_id=focal_player_id,
            )

        return self._evaluate_terminal(
            board,
            rollout_state,
            config=config,
            focal_player_id=focal_player_id,
            phase_evaluator=phase_evaluator,
            state_eval_cache=state_eval_cache,
        )

    def _evaluate_terminal(
        self,
        board: BoardState,
        state: OpeningState,
        *,
        config: AnalysisConfig,
        focal_player_id: int,
        phase_evaluator: PhaseRolloutEvaluator,
        state_eval_cache: dict[tuple, float],
    ) -> float:
        cache_key = ("terminal", focal_player_id, state.key())
        cached = state_eval_cache.get(cache_key)
        if cached is not None:
            return cached

        focal_vertices = list(state.player_vertices[focal_player_id])
        if not focal_vertices:
            value = -30.0
            state_eval_cache[cache_key] = value
            return value

        primary_phase = phase_evaluator.evaluate_vertex(
            focal_vertices[0],
            occupied_vertices=state.occupied,
            player_existing_vertices=focal_vertices[1:],
        )
        phase_total = primary_phase.total_score

        if len(focal_vertices) >= 2:
            secondary_phase = phase_evaluator.evaluate_vertex(
                focal_vertices[1],
                occupied_vertices=state.occupied,
                player_existing_vertices=focal_vertices[:1],
            )
            phase_total = (phase_total + secondary_phase.total_score) / 2.0

        road_targets = state.player_road_targets[focal_player_id]
        if road_targets:
            road_values = [
                static_vertex_value(
                    board,
                    target_vertex_id,
                    include_ports=config.include_ports,
                )
                for target_vertex_id in road_targets
            ]
            road_bonus = 0.18 * (sum(road_values) / len(road_values))
        else:
            road_bonus = 0.0

        value = phase_total + road_bonus - state.blocking_harm_to_focal
        state_eval_cache[cache_key] = value
        return value

    def _select_child(
        self,
        node: TreeNode,
        *,
        config: AnalysisConfig,
        turn_order: list[int],
        focal_player_id: int,
    ) -> TreeNode:
        actor = self._actor_for_state(node.state, turn_order)
        exploration_c = float(config.mcts_exploration_c)
        parent_visits_log = math.log(max(2, node.visits))

        best_score = float("-inf")
        best_child = node.children[0]
        for child in node.children:
            if child.action is None:
                continue
            exploration = exploration_c * math.sqrt(parent_visits_log / max(1, child.visits))
            if actor == focal_player_id:
                score = child.mean_value + exploration
            else:
                block_weight = float(config.opponent_block_weight)
                opponent_objective = ((1.0 - block_weight) * child.action.self_gain) + (
                    block_weight * child.action.blocking_delta
                )
                score = opponent_objective + exploration
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _generate_actions(
        self,
        board: BoardState,
        state: OpeningState,
        config: AnalysisConfig,
        turn_order: list[int],
        *,
        focal_player_id: int,
    ) -> list[OpeningAction]:
        if self._is_terminal(state, turn_order, min(config.mcts_max_plies, len(turn_order) * 2)):
            return []

        actor = self._actor_for_state(state, turn_order)
        if actor is None:
            return []

        if state.phase == "settlement":
            legal_vertices = board.legal_settlement_vertices(state.occupied)
            scored_actions: list[tuple[float, OpeningAction]] = []

            for vertex_id in legal_vertices:
                actor_vertices = state.player_vertices[actor]
                quick_score = score_vertex(
                    board,
                    vertex_id,
                    include_ports=config.include_ports,
                    occupied_vertices=state.occupied,
                    player_existing_vertices=actor_vertices,
                )
                self_gain = self._opening_settlement_gain(
                    board=board,
                    vertex_id=vertex_id,
                    quick_score=quick_score,
                    existing_settlement_count=len(actor_vertices),
                )
                blocking_delta = 0.0
                if actor != focal_player_id and state.player_vertices[focal_player_id]:
                    impact = blocking_externality_delta(
                        board,
                        source_vertices=state.player_vertices[focal_player_id],
                        occupied_before=state.occupied,
                        occupied_after=set(state.occupied) | {vertex_id},
                        include_ports=config.include_ports,
                    )
                    blocking_delta = impact.delta

                action = OpeningAction(
                    kind="settlement",
                    actor=actor,
                    vertex_id=vertex_id,
                    self_gain=round(self_gain, 4),
                    blocking_delta=round(blocking_delta, 4),
                    description=f"S{vertex_id}",
                )

                if actor == focal_player_id:
                    objective = self_gain
                else:
                    block_weight = float(config.opponent_block_weight)
                    objective = ((1.0 - block_weight) * self_gain) + (block_weight * blocking_delta)
                scored_actions.append((objective, action))

            scored_actions.sort(key=lambda item: (item[0], item[1].self_gain), reverse=True)
            action_limit = max(4, int(config.mcts_candidate_settlements))
            return [action for _, action in scored_actions[:action_limit]]

        # road direction phase
        actor_vertices = state.player_vertices.get(actor, [])
        if not actor_vertices:
            return [
                OpeningAction(
                    kind="road",
                    actor=actor,
                    self_gain=0.0,
                    blocking_delta=0.0,
                    description="R-hold",
                )
            ]

        occupied = set(state.occupied)
        settlement_blocked = build_settlement_blocked_vertices(board, occupied)
        target_scores: dict[int, tuple[float, int]] = {}
        for source_vertex_id in actor_vertices:
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
                path_value = static_vertex_value(
                    board,
                    candidate_vertex_id,
                    include_ports=config.include_ports,
                ) - (TOPOLOGY_DISTANCE_LAMBDA * distance)
                previous = target_scores.get(candidate_vertex_id)
                if previous is None or path_value > previous[0]:
                    target_scores[candidate_vertex_id] = (path_value, distance)

        if not target_scores:
            return [
                OpeningAction(
                    kind="road",
                    actor=actor,
                    self_gain=0.0,
                    blocking_delta=0.0,
                    description="R-hold",
                )
            ]

        focal_snapshot = (
            best_expansion_snapshot(
                board,
                source_vertices=state.player_vertices[focal_player_id],
                occupied_vertices=occupied,
                include_ports=config.include_ports,
            )
            if actor != focal_player_id and state.player_vertices[focal_player_id]
            else None
        )
        scored_actions: list[tuple[float, OpeningAction]] = []
        for target_vertex_id, (path_value, distance) in target_scores.items():
            blocking_delta = 0.0
            if focal_snapshot and focal_snapshot.best_vertex_id == target_vertex_id:
                blocking_delta += 0.8
            action = OpeningAction(
                kind="road",
                actor=actor,
                road_target_id=target_vertex_id,
                road_distance=distance,
                self_gain=round(path_value, 4),
                blocking_delta=round(blocking_delta, 4),
                description=f"R->{target_vertex_id} (d{distance})",
            )

            if actor == focal_player_id:
                objective = path_value
            else:
                block_weight = float(config.opponent_block_weight)
                objective = ((1.0 - block_weight) * path_value) + (block_weight * blocking_delta)
            scored_actions.append((objective, action))

        scored_actions.sort(key=lambda item: (item[0], item[1].self_gain), reverse=True)
        action_limit = max(1, int(config.mcts_candidate_road_directions))
        return [action for _, action in scored_actions[:action_limit]]

    def _apply_action(
        self,
        board: BoardState,
        state: OpeningState,
        action: OpeningAction,
        *,
        config: AnalysisConfig,
        focal_player_id: int,
    ) -> OpeningState:
        next_state = state.clone()
        next_state.ply_index += 1

        if action.kind == "settlement":
            if action.vertex_id is not None:
                next_state.occupied.add(action.vertex_id)
                next_state.player_vertices[action.actor].append(action.vertex_id)
                if action.actor != focal_player_id:
                    next_state.blocking_harm_to_focal += action.blocking_delta
            next_state.phase = "road"
            next_state.pending_actor = action.actor
            next_state.pending_settlement_vertex = action.vertex_id
            return next_state

        # road direction
        if action.road_target_id is not None:
            next_state.player_road_targets[action.actor].append(action.road_target_id)
        next_state.phase = "settlement"
        next_state.pending_actor = None
        next_state.pending_settlement_vertex = None
        next_state.turn_index += 1
        return next_state

    @staticmethod
    def _sample_best(
        actions: list[OpeningAction],
        *,
        key_fn,
        rng: random.Random,
    ) -> OpeningAction:
        if len(actions) == 1:
            return actions[0]
        sorted_actions = sorted(actions, key=key_fn, reverse=True)
        top_bucket = sorted_actions[: min(3, len(sorted_actions))]
        if rng.random() < 0.85:
            return top_bucket[0]
        return rng.choice(top_bucket)

    @staticmethod
    def _actor_for_state(state: OpeningState, turn_order: list[int]) -> int | None:
        if state.phase == "settlement":
            if state.turn_index >= len(turn_order):
                return None
            return turn_order[state.turn_index]
        return state.pending_actor

    @staticmethod
    def _is_terminal(state: OpeningState, turn_order: list[int], max_plies: int) -> bool:
        if state.turn_index >= len(turn_order):
            return True
        return state.ply_index >= max_plies

    def _opening_settlement_gain(
        self,
        *,
        board: BoardState,
        vertex_id: int,
        quick_score: VertexScore,
        existing_settlement_count: int,
    ) -> float:
        gain = quick_score.total_score
        if existing_settlement_count != 0:
            return gain

        port_penalty = self._first_settlement_port_penalty(board, vertex_id, quick_score)
        return gain - port_penalty

    def _first_settlement_port_penalty(
        self,
        board: BoardState,
        vertex_id: int,
        quick_score: VertexScore,
    ) -> float:
        port_type = board.vertices[vertex_id].port_type
        if port_type is None:
            return 0.0

        expected_yield = quick_score.expected_yield
        low_yield_penalty = max(0.0, 11.0 - expected_yield) * 0.35

        resources = {
            tile.resource
            for tile in board.vertex_adjacent_tiles(vertex_id)
            if tile.resource is not Resource.DESERT
        }
        road_recipe_bonus = 0.0 if {Resource.WOOD, Resource.BRICK}.issubset(resources) else 0.6

        return 2.75 + low_yield_penalty + road_recipe_bonus
