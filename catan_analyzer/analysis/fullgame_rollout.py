from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from catan_analyzer.domain.board import BoardState, Resource
from catan_analyzer.game import (
    CITY_COST,
    DEV_COST,
    ROAD_COST,
    SETTLEMENT_COST,
    ACTION_BUILD_CITY,
    ACTION_BUILD_ROAD,
    ACTION_BUILD_SETTLEMENT,
    ACTION_BUY_DEV_CARD,
    ACTION_DISCARD_RESOURCES,
    ACTION_END_BUILD_PHASE,
    ACTION_END_TRADE_PHASE,
    ACTION_END_TURN,
    ACTION_MOVE_ROBBER,
    ACTION_PLAY_KNIGHT,
    ACTION_PLAY_MONOPOLY,
    ACTION_PLAY_ROAD_BUILDING,
    ACTION_PLAY_YEAR_OF_PLENTY,
    ACTION_REVEAL_VP,
    ACTION_ROLL_DICE,
    ACTION_STEAL_RESOURCE,
    DevCardType,
    GameAction,
    GamePhase,
    GameState,
    apply_action,
    end_build_phase,
    end_trade_phase,
    end_turn,
    initialize_game_state,
    list_legal_actions,
    player_visible_victory_points,
    roll_dice,
)

from .draft import rank_vertices, simulate_draft
from .runtime import AnalysisRuntime
from .scoring import pip_value, score_vertex
from .seeding import analysis_seed, derive_seed
from .types import (
    AnalysisConfig,
    AnalysisResult,
    FullGameCandidateReport,
    FullGameSummary,
    MctsLineExplanation,
    VertexScore,
)


@dataclass
class _CandidateRolloutStats:
    wins_for_focal: float = 0.0
    games: int = 0
    winner_turn_sum: float = 0.0
    focal_win_turn_sum: float = 0.0
    game_length_sum: float = 0.0
    robber_loss_sum: float = 0.0

    def record(self, *, focal_won: bool, winner_turn: int, game_length_turns: int, robber_loss_to_focal: float) -> None:
        self.games += 1
        self.winner_turn_sum += float(winner_turn)
        self.game_length_sum += float(game_length_turns)
        self.robber_loss_sum += float(robber_loss_to_focal)
        if focal_won:
            self.wins_for_focal += 1.0
            self.focal_win_turn_sum += float(winner_turn)

    @property
    def win_rate(self) -> float:
        if self.games <= 0:
            return 0.0
        return self.wins_for_focal / self.games

    @property
    def avg_winner_turn(self) -> float:
        if self.games <= 0:
            return 0.0
        return self.winner_turn_sum / self.games

    @property
    def avg_game_length(self) -> float:
        if self.games <= 0:
            return 0.0
        return self.game_length_sum / self.games

    @property
    def avg_robber_loss(self) -> float:
        if self.games <= 0:
            return 0.0
        return self.robber_loss_sum / self.games

    @property
    def avg_focal_win_turn(self) -> float:
        if self.wins_for_focal <= 0:
            return 0.0
        return self.focal_win_turn_sum / self.wins_for_focal

    def beta_interval_95(self) -> tuple[float, float]:
        mean = self.win_rate
        n = max(1, self.games)
        variance = (mean * (1.0 - mean)) / n
        margin = 1.96 * math.sqrt(max(0.0, variance))
        return max(0.0, mean - margin), min(1.0, mean + margin)

class FullGameAnalyzer:
    """Full-game simulator with a strict turn/phase state machine and policy bots."""

    def analyze(
        self,
        board: BoardState,
        config: AnalysisConfig,
        runtime: AnalysisRuntime | None = None,
    ) -> AnalysisResult:
        if runtime is not None:
            runtime.report_progress("Preparing full-game evaluator...", 0.01, force=True)

        baseline_scores = self._baseline_ranking(board)
        candidate_count = max(2, min(len(baseline_scores), int(config.full_game_candidate_vertices)))
        prefilter_count = min(len(baseline_scores), max(candidate_count, candidate_count * 3))
        candidates = self._select_simulation_candidates(
            board,
            baseline_scores=baseline_scores,
            candidate_count=candidate_count,
            prefilter_count=prefilter_count,
        )

        rollout_count = max(1, int(config.full_game_rollouts))
        total_rollout_budget = max(candidate_count, rollout_count * candidate_count)
        base_seed = analysis_seed(board, config.mc_seed, salt="full_game_rollout")
        allocator_rng = random.Random(derive_seed(base_seed, "allocator"))

        aggregate_player_wins = {player_id: 0.0 for player_id in range(1, config.player_count + 1)}
        aggregate_player_turns = {player_id: 0.0 for player_id in range(1, config.player_count + 1)}
        aggregate_game_lengths = 0.0
        aggregate_games = 0

        candidate_stats: dict[int, _CandidateRolloutStats] = {vertex_id: _CandidateRolloutStats() for vertex_id in candidates}
        candidate_rollout_index = {vertex_id: 0 for vertex_id in candidates}

        def _run_rollout_for(vertex_id: int) -> None:
            nonlocal aggregate_games, aggregate_game_lengths
            rollout_index = candidate_rollout_index[vertex_id]
            candidate_rollout_index[vertex_id] += 1
            rollout_seed = derive_seed(base_seed, "candidate", vertex_id, rollout_index)
            state = self._initialize_opening_state(
                board,
                player_count=config.player_count,
                seed=rollout_seed,
                forced_first_vertex_id=vertex_id,
            )
            outcome = self._simulate_game(
                state,
                config=config,
                max_turns=max(40, int(config.full_game_max_turns)),
            )
            winner_id = int(outcome["winner_id"])
            winner_turn = int(outcome["winner_turn"])
            game_length_turns = int(outcome.get("game_length_turns", winner_turn))
            aggregate_games += 1
            aggregate_game_lengths += float(game_length_turns)

            for player_id in aggregate_player_wins:
                if winner_id == player_id:
                    aggregate_player_wins[player_id] += 1.0
                aggregate_player_turns[player_id] += float(
                    outcome["turns_to_victory"].get(player_id, config.full_game_max_turns)
                )

            candidate_stats[vertex_id].record(
                focal_won=(winner_id == 1),
                winner_turn=winner_turn,
                game_length_turns=game_length_turns,
                robber_loss_to_focal=float(outcome["robber_loss_to_focal"]),
            )

        # Initial exploration pass: at least one rollout for every candidate.
        for vertex_id in candidates:
            if runtime is not None:
                runtime.raise_if_cancelled()
            _run_rollout_for(vertex_id)
            if runtime is not None:
                runtime.report_progress(
                    "Running full-game rollouts...",
                    0.04 + (0.78 * (aggregate_games / max(1, total_rollout_budget))),
                )

        # Thompson sampling allocates the remaining budget.
        while aggregate_games < total_rollout_budget:
            if runtime is not None:
                runtime.raise_if_cancelled()
            sampled_vertex = self._thompson_pick_vertex(candidates, candidate_stats, allocator_rng)
            _run_rollout_for(sampled_vertex)
            if runtime is not None:
                runtime.report_progress(
                    "Running full-game rollouts...",
                    0.04 + (0.78 * (aggregate_games / max(1, total_rollout_budget))),
                )

        candidate_signals: dict[int, tuple[float, float, float]] = {}
        for vertex_id, stats in candidate_stats.items():
            avg_focal_win_turn = stats.avg_focal_win_turn if stats.avg_focal_win_turn > 0 else float(config.full_game_max_turns)
            candidate_signals[vertex_id] = (stats.win_rate, avg_focal_win_turn, stats.avg_robber_loss)

        ranking: list[VertexScore] = []
        for score in baseline_scores:
            stats = candidate_stats.get(score.vertex_id)
            if stats is None:
                ranking.append(score)
                continue
            avg_focal_win_turn = stats.avg_focal_win_turn if stats.avg_focal_win_turn > 0 else float(config.full_game_max_turns)
            full_game_boost = (8.0 * stats.win_rate) - (avg_focal_win_turn / 60.0)
            total = (0.45 * score.total_score) + full_game_boost
            ranking.append(
                replace(
                    score,
                    total_score=round(total, 4),
                    tempo_score=round((100.0 / max(1.0, avg_focal_win_turn)), 4),
                    recipe_coverage_score=round(stats.win_rate * 5.0, 4),
                    robber_penalty=round(stats.avg_robber_loss, 4),
                )
            )

        ranking.sort(key=lambda item: (-item.total_score, -item.expected_yield, item.vertex_id))
        top_recommendations = ranking[: min(2 * config.player_count, len(ranking))]

        score_lookup = {score.vertex_id: score for score in ranking}

        def _score_fn(vertex_id: int, occupied: set[int], player_vertices: list[int]) -> VertexScore:
            if vertex_id in score_lookup:
                base = score_lookup[vertex_id]
            else:
                base = score_vertex(
                    board,
                    vertex_id,
                    include_ports=config.include_ports,
                    occupied_vertices=occupied,
                    player_existing_vertices=player_vertices,
                )
            if occupied:
                dynamic = score_vertex(
                    board,
                    vertex_id,
                    include_ports=config.include_ports,
                    occupied_vertices=occupied,
                    player_existing_vertices=player_vertices,
                )
                return replace(
                    base,
                    total_score=round((0.65 * base.total_score) + (0.35 * dynamic.total_score), 4),
                    risk_penalty=round(dynamic.risk_penalty, 4),
                )
            return base

        predicted_sequence = simulate_draft(
            board,
            config.player_count,
            _score_fn,
            cancel_check=(runtime.raise_if_cancelled if runtime is not None else None),
        )

        candidate_reports = self._build_candidate_reports(
            board=board,
            candidate_stats=candidate_stats,
            ranking=ranking,
        )

        full_game_summary = FullGameSummary(
            rollout_count=max(1, aggregate_games),
            player_win_rates={
                player_id: (aggregate_player_wins[player_id] / max(1, aggregate_games))
                for player_id in aggregate_player_wins
            },
            expected_turns_to_victory={
                player_id: (aggregate_player_turns[player_id] / max(1, aggregate_games))
                for player_id in aggregate_player_turns
            },
            average_game_length_turns=(aggregate_game_lengths / max(1, aggregate_games)),
            top_candidates=candidate_reports,
            predicted_first_round=[pick.vertex_id for pick in predicted_sequence[: config.player_count]],
        )

        explain_lines = self._build_explain_lines(ranking, candidate_signals, candidates, candidate_stats)
        if runtime is not None:
            runtime.report_progress("Full-game analysis complete.", 1.0, force=True)

        return AnalysisResult(
            global_ranking=ranking,
            predicted_sequence=predicted_sequence,
            top_recommendations=top_recommendations,
            explain_lines=explain_lines,
            full_game_summary=full_game_summary,
        )

    def _baseline_ranking(self, board: BoardState) -> list[VertexScore]:
        def _score(vertex_id: int, occupied: set[int], player_vertices: list[int]) -> VertexScore:
            return score_vertex(
                board,
                vertex_id,
                include_ports=True,
                occupied_vertices=occupied,
                player_existing_vertices=player_vertices,
            )

        return rank_vertices(board, _score)

    def _select_simulation_candidates(
        self,
        board: BoardState,
        *,
        baseline_scores: Sequence[VertexScore],
        candidate_count: int,
        prefilter_count: int,
    ) -> list[int]:
        baseline_rank = {score.vertex_id: index for index, score in enumerate(baseline_scores)}
        static_ranked = sorted(
            baseline_scores,
            key=lambda score: (
                -self._opening_static_signal(board, score.vertex_id),
                baseline_rank[score.vertex_id],
            ),
        )
        prefiltered = static_ranked[: max(1, prefilter_count)]
        prefiltered.sort(key=lambda score: baseline_rank[score.vertex_id])
        return [score.vertex_id for score in prefiltered[: max(1, candidate_count)]]

    def _opening_static_signal(self, board: BoardState, vertex_id: int) -> float:
        tiles = list(board.vertex_adjacent_tiles(vertex_id))
        pip_total = float(sum(pip_value(tile.token_number) for tile in tiles))
        resource_weights: dict[Resource, float] = {}
        for tile in tiles:
            if tile.resource is Resource.DESERT:
                continue
            resource_weights[tile.resource] = resource_weights.get(tile.resource, 0.0) + float(pip_value(tile.token_number))
        total_weight = sum(resource_weights.values())
        if total_weight <= 0:
            diversity = 0.0
        else:
            probs = [weight / total_weight for weight in resource_weights.values() if weight > 0]
            entropy = -sum(prob * math.log(prob) for prob in probs)
            max_entropy = math.log(5.0)
            diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        missing_key_penalty = 0.0
        if Resource.BRICK not in resource_weights:
            missing_key_penalty += 1.2
        if Resource.WOOD not in resource_weights:
            missing_key_penalty += 1.2

        vertex = board.vertices[vertex_id]
        port_bonus = 0.0
        if vertex.port_type is not None:
            port_bonus += 0.9

        expansion_options = 0.0
        for neighbor_id in vertex.adjacent_vertex_ids:
            neighbor_tiles = list(board.vertex_adjacent_tiles(neighbor_id))
            expansion_options += 0.2 * sum(pip_value(tile.token_number) for tile in neighbor_tiles)

        return (0.45 * pip_total) + (2.6 * diversity) + port_bonus + expansion_options - missing_key_penalty

    def _thompson_pick_vertex(
        self,
        candidates: Sequence[int],
        candidate_stats: dict[int, _CandidateRolloutStats],
        rng: random.Random,
    ) -> int:
        best_vertex = int(candidates[0])
        best_sample = -1.0
        for vertex_id in candidates:
            stats = candidate_stats[vertex_id]
            alpha = 1.0 + stats.wins_for_focal
            beta = 1.0 + max(0.0, stats.games - stats.wins_for_focal)
            sample = rng.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_vertex = int(vertex_id)
        return best_vertex

    def _build_candidate_reports(
        self,
        *,
        board: BoardState,
        candidate_stats: dict[int, _CandidateRolloutStats],
        ranking: Sequence[VertexScore],
    ) -> list[FullGameCandidateReport]:
        rank_lookup = {score.vertex_id: index for index, score in enumerate(ranking)}
        ordered_candidates = sorted(candidate_stats.keys(), key=lambda vertex_id: rank_lookup.get(vertex_id, 10_000))
        reports: list[FullGameCandidateReport] = []
        for vertex_id in ordered_candidates[:8]:
            stats = candidate_stats[vertex_id]
            ci_low, ci_high = stats.beta_interval_95()
            reports.append(
                FullGameCandidateReport(
                    vertex_id=vertex_id,
                    simulations=max(1, stats.games),
                    win_rate=round(stats.win_rate, 4),
                    win_rate_ci_low=round(ci_low, 4),
                    win_rate_ci_high=round(ci_high, 4),
                    avg_win_turn=round(stats.avg_focal_win_turn if stats.avg_focal_win_turn > 0 else 0.0, 3),
                    avg_game_length_turns=round(stats.avg_game_length, 3),
                    avg_robber_loss_to_focal=round(stats.avg_robber_loss, 4),
                    total_pips=int(sum(pip_value(tile.token_number) for tile in board.vertex_adjacent_tiles(vertex_id))),
                    on_port=(board.vertices[vertex_id].port_type is not None),
                )
            )
        return reports

    def _initialize_opening_state(
        self,
        board: BoardState,
        *,
        player_count: int,
        seed: int,
        forced_first_vertex_id: int,
    ) -> GameState:
        state = initialize_game_state(board, player_count=player_count, seed=seed)
        # Force focal player's first settlement.
        state = apply_action(state, GameAction("place_setup_settlement", {"vertex_id": forced_first_vertex_id}))
        road_actions = list_legal_actions(state)
        if road_actions:
            state = apply_action(state, self._best_setup_road_action(state, road_actions))

        while state.phase in {GamePhase.SETUP_SETTLEMENT, GamePhase.SETUP_ROAD}:
            actions = list_legal_actions(state)
            if not actions:
                break
            if state.phase is GamePhase.SETUP_SETTLEMENT:
                state = apply_action(state, self._best_setup_settlement_action(state, actions))
            else:
                state = apply_action(state, self._best_setup_road_action(state, actions))
        return state

    def _best_setup_settlement_action(self, state: GameState, actions: Sequence[GameAction]) -> GameAction:
        best = actions[0]
        best_value = float("-inf")
        occupied = state.all_occupied_vertices()
        player = state.players[state.current_player_id]
        existing = list(player.settlements | player.cities)
        for action in actions:
            vertex_id = int(action.data["vertex_id"])
            value = score_vertex(
                state.board,
                vertex_id,
                include_ports=True,
                occupied_vertices=occupied,
                player_existing_vertices=existing,
            ).total_score
            if value > best_value:
                best = action
                best_value = value
        return best

    def _best_setup_road_action(self, state: GameState, actions: Sequence[GameAction]) -> GameAction:
        best = actions[0]
        best_value = float("-inf")
        for action in actions:
            edge = tuple(action.data["edge"])
            anchor = state.pending_setup_vertex_id
            if anchor is None:
                continue
            other = int(edge[0]) if int(edge[1]) == anchor else int(edge[1])
            value = score_vertex(state.board, other, include_ports=True).total_score
            if value > best_value:
                best = action
                best_value = value
        return best

    def _simulate_game(self, state: GameState, *, config: AnalysisConfig, max_turns: int) -> dict[str, float | int | dict[int, int]]:
        robber_loss_to_focal = 0.0
        turn_limit = max(20, int(max_turns))
        while state.phase is not GamePhase.GAME_OVER and state.turn_number <= turn_limit:
            if state.phase in {GamePhase.TURN_START, GamePhase.TRADE, GamePhase.BUILD, GamePhase.DEV_PLAY}:
                if not state.turn_has_rolled:
                    state = self._execute_turn_before_roll(state)
                else:
                    state = self._execute_post_roll_turn(state, config)
                continue

            if state.phase is GamePhase.ROBBER_DISCARD:
                actions = list_legal_actions(state)
                state = apply_action(state, actions[0])
                continue

            if state.phase is GamePhase.ROBBER_MOVE:
                action = self._best_robber_move_action(state)
                if action is None:
                    legal = list_legal_actions(state)
                    action = legal[0]
                state = apply_action(state, action)
                continue

            if state.phase is GamePhase.ROBBER_STEAL:
                action = self._best_steal_action(state)
                before = state.players[1].card_count()
                acting_player_id = state.current_player_id
                state = apply_action(state, action)
                after = state.players[1].card_count()
                if acting_player_id != 1 and after < before:
                    robber_loss_to_focal += float(before - after)
                continue

            break

        winner_id = int(state.winner_id or 0)
        winner_turn = int(state.turn_number if winner_id != 0 else (turn_limit + 20))
        game_length_turns = int(min(state.turn_number, turn_limit))
        turns_to_victory = {
            player_id: (winner_turn if player_id == winner_id and winner_id != 0 else turn_limit + 20)
            for player_id in state.players
        }
        return {
            "winner_id": winner_id,
            "winner_turn": winner_turn,
            "game_length_turns": game_length_turns,
            "turns_to_victory": turns_to_victory,
            "robber_loss_to_focal": robber_loss_to_focal,
        }

    def _execute_turn_before_roll(self, state: GameState) -> GameState:
        action = self._choose_dev_action(state, include_vp_reveal=True)
        if action is not None:
            return apply_action(state, action)
        legal = list_legal_actions(state)
        roll_actions = [candidate for candidate in legal if candidate.kind == ACTION_ROLL_DICE]
        if roll_actions:
            return apply_action(state, roll_actions[0])
        return state

    def _execute_post_roll_turn(self, state: GameState, config: AnalysisConfig) -> GameState:
        action_budget = 10
        trades_remaining = max(0, int(config.full_game_trade_offer_limit))
        while (
            action_budget > 0
            and state.phase in {GamePhase.TURN_START, GamePhase.TRADE, GamePhase.BUILD, GamePhase.DEV_PLAY}
            and state.turn_has_rolled
            and state.phase is not GamePhase.GAME_OVER
        ):
            action = self._choose_post_roll_action(state, trades_remaining=trades_remaining)
            if action is None:
                break
            if action.kind in {"trade_bank", "trade_player"}:
                trades_remaining = max(0, trades_remaining - 1)
            state = apply_action(state, action)
            if state.phase is GamePhase.GAME_OVER:
                return state
            action_budget -= 1

        if state.phase in {GamePhase.TURN_START, GamePhase.TRADE, GamePhase.BUILD, GamePhase.DEV_PLAY} and state.turn_has_rolled:
            legal = list_legal_actions(state)
            end_turn_actions = [candidate for candidate in legal if candidate.kind == ACTION_END_TURN]
            if end_turn_actions:
                return apply_action(state, end_turn_actions[0])
            return apply_action(state, end_turn())
        return state

    def _choose_post_roll_action(self, state: GameState, *, trades_remaining: int) -> GameAction | None:
        legal = list_legal_actions(state)
        if not legal:
            return None

        reveal_actions = [action for action in legal if action.kind == ACTION_REVEAL_VP]
        if reveal_actions:
            return reveal_actions[0]

        build_action = self._choose_build_action(state)
        trade_action, trade_utility = self._choose_best_trade_action(state)
        dev_action = self._choose_dev_action(state, include_vp_reveal=False)

        if build_action is not None:
            return build_action
        if trades_remaining > 0 and trade_action is not None and trade_utility > 0.1:
            return trade_action
        if dev_action is not None:
            return dev_action
        if trades_remaining > 0 and trade_action is not None and trade_utility > 0.0:
            return trade_action
        return None

    def _best_robber_move_action(self, state: GameState) -> GameAction | None:
        legal = list_legal_actions(state)
        if not legal:
            return None
        best = legal[0]
        best_score = float("-inf")
        acting_player_id = state.current_player_id
        for action in legal:
            tile_id = int(action.data["tile_id"])
            score = 0.0
            for player_id, player in state.players.items():
                for vertex_id in player.settlements:
                    if tile_id not in state.board.vertices[vertex_id].adjacent_hex_ids:
                        continue
                    pips = sum(
                        pip_value(state.board.get_tile(adj).token_number)
                        for adj in state.board.vertices[vertex_id].adjacent_hex_ids
                    )
                    if player_id == acting_player_id:
                        score -= 0.45 * pips
                    else:
                        score += 1.00 * pips
                for vertex_id in player.cities:
                    if tile_id not in state.board.vertices[vertex_id].adjacent_hex_ids:
                        continue
                    pips = sum(
                        pip_value(state.board.get_tile(adj).token_number)
                        for adj in state.board.vertices[vertex_id].adjacent_hex_ids
                    )
                    if player_id == acting_player_id:
                        score -= 0.85 * pips
                    else:
                        score += 1.40 * pips
            targets = [
                player_id
                for player_id in state.players
                if player_id != acting_player_id
                and state.players[player_id].card_count() > 0
                and any(tile_id in state.board.vertices[v].adjacent_hex_ids for v in state.players[player_id].settlements | state.players[player_id].cities)
            ]
            if targets:
                score += 2.0
            if score > best_score:
                best_score = score
                best = action
        return best

    def _best_steal_action(self, state: GameState) -> GameAction:
        legal = list_legal_actions(state)
        if not legal:
            return GameAction("skip_steal", {})
        def _target_cards(action: GameAction) -> int:
            target_id = int(action.data.get("target_player_id", -1))
            if target_id not in state.players:
                return -1
            return state.players[target_id].card_count()
        ranked = sorted(
            legal,
            key=_target_cards,
            reverse=True,
        )
        return ranked[0]

    def _execute_trade_phase(self, state: GameState, config: AnalysisConfig) -> GameState:
        offers_remaining = max(0, int(config.full_game_trade_offer_limit))
        while offers_remaining > 0 and state.phase is GamePhase.TRADE:
            action, _utility = self._choose_best_trade_action(state)
            if action is None:
                break
            state = apply_action(state, action)
            offers_remaining -= 1
        if state.phase is GamePhase.TRADE:
            state = apply_action(state, end_trade_phase())
        return state

    def _choose_best_trade_action(self, state: GameState) -> tuple[GameAction | None, float]:
        player = state.players[state.current_player_id]
        legal = [action for action in list_legal_actions(state) if action.kind in { "trade_bank", "trade_player" }]
        if not legal:
            return None, 0.0
        baseline = self._build_access_score(player.hand)
        baseline_vp = player_visible_victory_points(state, player.player_id)
        best_action: GameAction | None = None
        best_delta = 0.0
        for action in legal:
            if action.kind == "trade_bank":
                simulated = self._simulate_trade_effect(player.hand, action)
                delta = self._build_access_score(simulated) - baseline
                if delta > best_delta:
                    best_delta = delta
                    best_action = action
                continue

            target_player_id = int(action.data["target_player_id"])
            target = state.players[target_player_id]
            our_after = self._simulate_trade_effect(player.hand, action)
            their_after = self._simulate_trade_effect(target.hand, action, perspective="target")
            our_delta = self._build_access_score(our_after) - baseline
            their_delta = self._build_access_score(their_after) - self._build_access_score(target.hand)
            target_vp = player_visible_victory_points(state, target_player_id)
            if target_vp >= baseline_vp + 2 and their_delta > 0:
                continue
            if our_delta <= 0:
                continue
            if their_delta < -0.35:
                continue
            utility = our_delta - max(0.0, their_delta * 0.45)
            if utility > best_delta:
                best_delta = utility
                best_action = action
        return best_action, best_delta

    def _simulate_trade_effect(
        self,
        hand: dict[Resource, int],
        action: GameAction,
        *,
        perspective: str = "self",
    ) -> dict[Resource, int]:
        simulated = dict(hand)
        if action.kind == "trade_bank":
            give_resource = action.data["give_resource"]
            receive_resource = action.data["receive_resource"]
            give_amount = int(action.data.get("give_amount", 4))
            if isinstance(give_resource, Resource) and isinstance(receive_resource, Resource):
                simulated[give_resource] = max(0, simulated.get(give_resource, 0) - give_amount)
                simulated[receive_resource] = simulated.get(receive_resource, 0) + 1
            return simulated

        give = action.data.get("give", {})
        receive = action.data.get("receive", {})
        give_map = {
            (resource if isinstance(resource, Resource) else Resource(str(resource))): int(amount)
            for resource, amount in give.items()
        }
        receive_map = {
            (resource if isinstance(resource, Resource) else Resource(str(resource))): int(amount)
            for resource, amount in receive.items()
        }

        if perspective == "self":
            for resource, amount in give_map.items():
                simulated[resource] = max(0, simulated.get(resource, 0) - amount)
            for resource, amount in receive_map.items():
                simulated[resource] = simulated.get(resource, 0) + amount
            return simulated

        for resource, amount in receive_map.items():
            simulated[resource] = max(0, simulated.get(resource, 0) - amount)
        for resource, amount in give_map.items():
            simulated[resource] = simulated.get(resource, 0) + amount
        return simulated

    def _build_access_score(self, hand: dict[Resource, int]) -> float:
        def missing(cost: dict[Resource, int]) -> int:
            return int(sum(max(0, amount - hand.get(resource, 0)) for resource, amount in cost.items()))

        score = 0.0
        score += 3.0 / (1.0 + missing(SETTLEMENT_COST))
        score += 3.2 / (1.0 + missing(CITY_COST))
        score += 2.2 / (1.0 + missing(DEV_COST))
        score += 1.0 / (1.0 + missing(ROAD_COST))
        return score

    def _execute_build_phase(self, state: GameState) -> GameState:
        action_budget = 8
        while action_budget > 0 and state.phase is GamePhase.BUILD:
            action = self._choose_build_action(state)
            if action is None:
                break
            state = apply_action(state, action)
            if state.phase is GamePhase.GAME_OVER:
                return state
            action_budget -= 1
        if state.phase is GamePhase.BUILD:
            state = apply_action(state, end_build_phase())
        return state

    def _choose_build_action(self, state: GameState) -> GameAction | None:
        legal = list_legal_actions(state)
        if not legal:
            return None
        player = state.players[state.current_player_id]

        settlement_actions = [action for action in legal if action.kind == ACTION_BUILD_SETTLEMENT]
        if settlement_actions:
            return max(
                settlement_actions,
                key=lambda action: score_vertex(
                    state.board,
                    int(action.data["vertex_id"]),
                    include_ports=True,
                    occupied_vertices=state.all_occupied_vertices(),
                    player_existing_vertices=list(player.settlements | player.cities),
                ).total_score,
            )

        city_actions = [action for action in legal if action.kind == ACTION_BUILD_CITY]
        if city_actions:
            return max(city_actions, key=lambda action: self._vertex_yield(state.board, int(action.data["vertex_id"])))

        dev_actions = [action for action in legal if action.kind == ACTION_BUY_DEV_CARD]
        if dev_actions:
            return dev_actions[0]

        road_actions = [action for action in legal if action.kind == ACTION_BUILD_ROAD]
        if road_actions:
            return max(road_actions, key=lambda action: self._road_expansion_value(state, action))

        return None

    def _road_expansion_value(self, state: GameState, action: GameAction) -> float:
        edge = tuple(action.data["edge"])
        player = state.players[state.current_player_id]
        vertices = set(player.settlements) | set(player.cities)
        other = int(edge[0])
        if other in vertices:
            other = int(edge[1])
        return float(self._vertex_yield(state.board, other))

    def _vertex_yield(self, board: BoardState, vertex_id: int) -> int:
        return sum(pip_value(tile.token_number) for tile in board.vertex_adjacent_tiles(vertex_id))

    def _execute_dev_phase(self, state: GameState) -> GameState:
        action = self._choose_dev_action(state)
        if action is not None:
            state = apply_action(state, action)
            if state.phase is GamePhase.GAME_OVER:
                return state
        if state.phase is GamePhase.DEV_PLAY:
            state = apply_action(state, end_turn())
        return state

    def _choose_dev_action(self, state: GameState, *, include_vp_reveal: bool = True) -> GameAction | None:
        legal = list_legal_actions(state)
        if not legal:
            return None

        player = state.players[state.current_player_id]
        visible_vp = player_visible_victory_points(state, player.player_id)
        hidden_vp = player.hidden_vp_cards()
        if include_vp_reveal and hidden_vp > 0 and visible_vp + hidden_vp >= 10:
            reveal_actions = [action for action in legal if action.kind == ACTION_REVEAL_VP]
            if reveal_actions:
                return reveal_actions[0]

        knight_actions = [action for action in legal if action.kind == ACTION_PLAY_KNIGHT]
        if knight_actions:
            return max(knight_actions, key=lambda action: self._robber_tile_pressure(state, int(action.data["tile_id"])))

        road_building_actions = [action for action in legal if action.kind == ACTION_PLAY_ROAD_BUILDING]
        if road_building_actions:
            return road_building_actions[0]

        plenty_actions = [action for action in legal if action.kind == ACTION_PLAY_YEAR_OF_PLENTY]
        if plenty_actions:
            target_cost = self._best_cost_target(player.hand)
            if target_cost:
                needed = [
                    resource
                    for resource, amount in target_cost.items()
                    if player.hand.get(resource, 0) < amount
                ]
                if needed:
                    for action in plenty_actions:
                        first = action.data["resource_one"]
                        second = action.data["resource_two"]
                        if first in needed or second in needed:
                            return action

        monopoly_actions = [action for action in legal if action.kind == ACTION_PLAY_MONOPOLY]
        if monopoly_actions:
            best_action = None
            best_total = 0
            for action in monopoly_actions:
                resource = action.data["resource"]
                total = sum(
                    other.hand.get(resource, 0)
                    for other_id, other in state.players.items()
                    if other_id != state.current_player_id
                )
                if total > best_total:
                    best_total = total
                    best_action = action
            if best_action is not None and best_total >= 3:
                return best_action

        return None

    def _best_cost_target(self, hand: dict[Resource, int]) -> dict[Resource, int] | None:
        candidates = [SETTLEMENT_COST, CITY_COST, DEV_COST]
        best_cost = None
        best_missing = 999
        for cost in candidates:
            missing = sum(max(0, amount - hand.get(resource, 0)) for resource, amount in cost.items())
            if missing < best_missing:
                best_missing = missing
                best_cost = cost
        return best_cost

    def _robber_tile_pressure(self, state: GameState, tile_id: int) -> float:
        score = 0.0
        acting = state.current_player_id
        for player_id, player in state.players.items():
            for vertex_id in player.settlements | player.cities:
                if tile_id not in state.board.vertices[vertex_id].adjacent_hex_ids:
                    continue
                pips = self._vertex_yield(state.board, vertex_id)
                if player_id == acting:
                    score -= 0.4 * pips
                else:
                    score += 1.0 * pips
        return score

    def _build_explain_lines(
        self,
        ranking: Sequence[VertexScore],
        candidate_signals: dict[int, tuple[float, float, float]],
        candidates: Iterable[int],
        candidate_stats: dict[int, _CandidateRolloutStats],
    ) -> list[MctsLineExplanation]:
        candidate_set = set(candidates)
        lines: list[MctsLineExplanation] = []
        for rank, score in enumerate(ranking[:8], start=1):
            win_rate, avg_turn, robber_loss = candidate_signals.get(
                score.vertex_id,
                (0.0, 0.0, 0.0),
            )
            suffix = "sim" if score.vertex_id in candidate_set else "base"
            stats = candidate_stats.get(score.vertex_id)
            interval = stats.beta_interval_95() if stats is not None else (0.0, 0.0)
            ci_text = f"CI[{interval[0]:.2f},{interval[1]:.2f}]"
            lines.append(
                MctsLineExplanation(
                    ply_index=rank,
                    actor=1,
                    action=f"FG#{rank} V{score.vertex_id} ({suffix}) {ci_text}",
                    self_value=round(win_rate, 4),
                    blocking_delta=round(robber_loss, 4),
                    uct_value=round(score.total_score, 4),
                    visits=max(1, int(round(avg_turn))) if avg_turn > 0 else 1,
                )
            )
        return lines
