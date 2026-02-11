from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from catan_analyzer.domain.board import BoardState, PortType, Resource

from .scoring import pip_value
from .seeding import analysis_seed, derive_seed
from .runtime import AnalysisRuntime
from .topology import (
    TOPOLOGY_DISTANCE_LAMBDA,
    build_settlement_blocked_vertices,
    candidate_expansion_vertices,
    road_distance_map,
)
from .types import AnalysisConfig, RobberPolicy

ROAD_COST = {Resource.WOOD: 1, Resource.BRICK: 1}
SETTLEMENT_COST = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
    Resource.WHEAT: 1,
    Resource.SHEEP: 1,
}
RESOURCE_TYPES = [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]


@dataclass(frozen=True)
class PhaseEvaluation:
    total_score: float
    tempo_score: float
    recipe_coverage_score: float
    fragility_penalty: float
    port_conversion_score: float
    robber_penalty: float
    best_partner_vertex_id: int | None = None


class PhaseRolloutEvaluator:
    """Evaluate settlement vertices through pair-aware phase rollouts."""

    def __init__(
        self,
        board: BoardState,
        config: AnalysisConfig,
        *,
        expected_yield_overrides: Mapping[int, float] | None = None,
        runtime: AnalysisRuntime | None = None,
    ) -> None:
        self.board = board
        self.config = config
        self.runtime = runtime
        self.expected_yield_overrides = dict(expected_yield_overrides or {})
        self._base_seed = analysis_seed(board, config.mc_seed, salt="phase_rollout")
        self._vertex_pair_cache: Dict[tuple[int, tuple[int, ...], tuple[int, ...]], PhaseEvaluation] = {}
        self._static_vertex_value_cache: Dict[int, float] = {}
        self._vertex_rates_cache: Dict[int, tuple[dict[Resource, float], dict[Resource, list[float]]]] = {}

    def evaluate_vertex(
        self,
        vertex_id: int,
        *,
        occupied_vertices: Iterable[int],
        player_existing_vertices: Iterable[int],
    ) -> PhaseEvaluation:
        if self.runtime is not None:
            self.runtime.raise_if_cancelled()
        occupied_signature = tuple(sorted(set(occupied_vertices)))
        player_signature = tuple(sorted(set(player_existing_vertices)))
        cache_key = (vertex_id, occupied_signature, player_signature)
        cached = self._vertex_pair_cache.get(cache_key)
        if cached is not None:
            return cached

        occupied = set(occupied_signature)
        if player_signature:
            base_vertices = list(dict.fromkeys([*player_signature, vertex_id]))
            evaluation = self._evaluate_pair(
                start_vertices=base_vertices,
                occupied_vertices=occupied,
                rollout_count=max(1, self.config.phase_rollout_count),
                best_partner_vertex_id=None,
                seed_tag=("existing_pair", vertex_id, occupied_signature, player_signature),
            )
            self._vertex_pair_cache[cache_key] = evaluation
            return evaluation

        legal_partners = [
            partner_id
            for partner_id in self.board.legal_settlement_vertices(occupied | {vertex_id})
            if partner_id != vertex_id
        ]
        if not legal_partners:
            evaluation = self._evaluate_pair(
                start_vertices=[vertex_id],
                occupied_vertices=occupied,
                rollout_count=max(1, self.config.phase_rollout_count),
                best_partner_vertex_id=None,
                seed_tag=("single_vertex", vertex_id, occupied_signature),
            )
            self._vertex_pair_cache[cache_key] = evaluation
            return evaluation

        partner_pool = sorted(
            legal_partners,
            key=lambda partner_id: self._pair_seed_score(vertex_id, partner_id),
            reverse=True,
        )
        top_partner_count = min(6, len(partner_pool))
        selected_partners = partner_pool[:top_partner_count]
        rollouts_per_partner = max(1, int(self.config.phase_rollout_count) // max(1, top_partner_count))

        best_evaluation: PhaseEvaluation | None = None
        for partner_id in selected_partners:
            evaluation = self._evaluate_pair(
                start_vertices=[vertex_id, partner_id],
                occupied_vertices=occupied,
                rollout_count=rollouts_per_partner,
                best_partner_vertex_id=partner_id,
                seed_tag=("partner_pair", vertex_id, partner_id, occupied_signature),
            )
            if best_evaluation is None or evaluation.total_score > best_evaluation.total_score:
                best_evaluation = evaluation

        assert best_evaluation is not None  # by construction selected_partners non-empty
        self._vertex_pair_cache[cache_key] = best_evaluation
        return best_evaluation

    def _evaluate_pair(
        self,
        *,
        start_vertices: Sequence[int],
        occupied_vertices: set[int],
        rollout_count: int,
        best_partner_vertex_id: int | None,
        seed_tag: tuple,
    ) -> PhaseEvaluation:
        rates, contribution_shares = self._pair_expected_inflow(start_vertices)
        recipe_score = self._recipe_coverage_score(rates)
        fragility_penalty = self._fragility_penalty(contribution_shares)
        port_conversion_score = self._port_conversion_score(start_vertices, rates)

        avg_turns_to_third, rollout_robber_impact = self._run_phase_rollouts(
            start_vertices=start_vertices,
            occupied_vertices=occupied_vertices,
            rollout_count=rollout_count,
            seed_tag=seed_tag,
        )
        tempo_score = -avg_turns_to_third

        base_robber_penalty = self._base_robber_penalty(start_vertices, contribution_shares)
        robber_penalty = (base_robber_penalty * self._robber_policy_multiplier()) + rollout_robber_impact

        total_score = (
            (0.65 * tempo_score)
            + (2.2 * recipe_score)
            + (0.75 * port_conversion_score)
            - (1.25 * fragility_penalty)
            - (1.10 * robber_penalty)
        )

        return PhaseEvaluation(
            total_score=round(total_score, 4),
            tempo_score=round(tempo_score, 4),
            recipe_coverage_score=round(recipe_score, 4),
            fragility_penalty=round(fragility_penalty, 4),
            port_conversion_score=round(port_conversion_score, 4),
            robber_penalty=round(robber_penalty, 4),
            best_partner_vertex_id=best_partner_vertex_id,
        )

    def _pair_seed_score(self, first_vertex_id: int, second_vertex_id: int) -> float:
        rates, shares = self._pair_expected_inflow([first_vertex_id, second_vertex_id])
        recipe_score = self._recipe_coverage_score(rates)
        fragility = self._fragility_penalty(shares)
        port_bonus = self._port_conversion_score([first_vertex_id, second_vertex_id], rates)
        yield_bonus = sum(rates.values()) * 8.0
        return yield_bonus + (1.8 * recipe_score) + (1.2 * port_bonus) - (1.4 * fragility)

    def _pair_expected_inflow(
        self,
        vertex_ids: Sequence[int],
    ) -> tuple[dict[Resource, float], dict[Resource, list[float]]]:
        rates: dict[Resource, float] = {resource: 0.0 for resource in RESOURCE_TYPES}
        contributions: dict[Resource, list[float]] = {resource: [] for resource in RESOURCE_TYPES}

        for vertex_id in vertex_ids:
            vertex_rates, vertex_contributions = self._vertex_expected_inflow(vertex_id)
            for resource in RESOURCE_TYPES:
                rate = vertex_rates[resource]
                rates[resource] += rate
                contributions[resource].extend(vertex_contributions[resource])

        return rates, contributions

    def _vertex_expected_inflow(
        self,
        vertex_id: int,
    ) -> tuple[dict[Resource, float], dict[Resource, list[float]]]:
        cached = self._vertex_rates_cache.get(vertex_id)
        if cached is not None:
            return cached

        rates: dict[Resource, float] = {resource: 0.0 for resource in RESOURCE_TYPES}
        contributions: dict[Resource, list[float]] = {resource: [] for resource in RESOURCE_TYPES}

        for tile in self.board.vertex_adjacent_tiles(vertex_id):
            if tile.resource is Resource.DESERT:
                continue
            pip = self._tile_pip(tile.id)
            if pip <= 0:
                continue
            rate = pip / 36.0
            rates[tile.resource] += rate
            contributions[tile.resource].append(rate)

        self._vertex_rates_cache[vertex_id] = (rates, contributions)
        return rates, contributions

    def _run_phase_rollouts(
        self,
        *,
        start_vertices: Sequence[int],
        occupied_vertices: set[int],
        rollout_count: int,
        seed_tag: tuple,
    ) -> tuple[float, float]:
        total_turns = 0.0
        total_robber_impact = 0.0
        for rollout_index in range(max(1, rollout_count)):
            if self.runtime is not None:
                self.runtime.raise_if_cancelled()
            rng = random.Random(derive_seed(self._base_seed, seed_tag, rollout_index))
            turns_to_third, robber_impact = self._simulate_single_phase(
                start_vertices=start_vertices,
                occupied_vertices=occupied_vertices,
                rng=rng,
            )
            total_turns += turns_to_third
            total_robber_impact += robber_impact

        iterations = float(max(1, rollout_count))
        return (total_turns / iterations, total_robber_impact / iterations)

    def _simulate_single_phase(
        self,
        *,
        start_vertices: Sequence[int],
        occupied_vertices: set[int],
        rng: random.Random,
    ) -> tuple[float, float]:
        horizon = max(1, int(self.config.phase_turn_horizon))
        inventory: defaultdict[Resource, int] = defaultdict(int)
        owned_vertices = set(start_vertices)
        occupied_total = set(occupied_vertices).union(owned_vertices)

        target_vertex_id, target_distance = self._best_expansion_target(owned_vertices, occupied_total)
        road_progress = 0
        robber_blocked_tile_id: int | None = None
        robber_impact = 0.0
        turns_to_third = float(horizon + 8)

        for turn_index in range(1, horizon + 1):
            dice_roll = rng.randint(1, 6) + rng.randint(1, 6)

            if dice_roll == 7:
                robber_blocked_tile_id = self._select_robber_tile(owned_vertices, rng=rng)
            else:
                gained, blocked = self._apply_production_roll(
                    owned_vertices=owned_vertices,
                    dice_roll=dice_roll,
                    robber_blocked_tile_id=robber_blocked_tile_id,
                )
                for resource, amount in gained.items():
                    inventory[resource] += amount
                robber_impact += blocked * 0.35

            if target_vertex_id is not None and target_vertex_id in occupied_total:
                target_vertex_id, target_distance = self._best_expansion_target(owned_vertices, occupied_total)
                road_progress = 0

            road_progress, target_vertex_id, target_distance = self._attempt_build_sequence(
                inventory=inventory,
                owned_vertices=owned_vertices,
                occupied_total=occupied_total,
                target_vertex_id=target_vertex_id,
                target_distance=target_distance,
                road_progress=road_progress,
            )

            if len(owned_vertices) >= 3:
                turns_to_third = float(turn_index)
                break

            if target_vertex_id is None:
                target_vertex_id, target_distance = self._best_expansion_target(owned_vertices, occupied_total)
                road_progress = 0

        return (turns_to_third, robber_impact)

    def _attempt_build_sequence(
        self,
        *,
        inventory: defaultdict[Resource, int],
        owned_vertices: set[int],
        occupied_total: set[int],
        target_vertex_id: int | None,
        target_distance: int,
        road_progress: int,
    ) -> tuple[int, int | None, int]:
        ports = self._ports_for_vertices(owned_vertices)

        if target_vertex_id is not None and road_progress < target_distance:
            self._trade_towards_cost(inventory, ROAD_COST, ports)
            if self._can_afford(inventory, ROAD_COST):
                self._spend(inventory, ROAD_COST)
                road_progress += 1

        if target_vertex_id is not None and road_progress >= target_distance:
            self._trade_towards_cost(inventory, SETTLEMENT_COST, ports)
            if self._can_afford(inventory, SETTLEMENT_COST) and self.board.is_legal_settlement(
                target_vertex_id, occupied_total
            ):
                self._spend(inventory, SETTLEMENT_COST)
                owned_vertices.add(target_vertex_id)
                occupied_total.add(target_vertex_id)
                road_progress = 0
                target_vertex_id, target_distance = self._best_expansion_target(owned_vertices, occupied_total)

        return road_progress, target_vertex_id, target_distance

    def _best_expansion_target(
        self,
        owned_vertices: set[int],
        occupied_total: set[int],
    ) -> tuple[int | None, int]:
        settlement_blocked = build_settlement_blocked_vertices(self.board, occupied_total)
        best_vertex_id: int | None = None
        best_distance = 0
        best_value = float("-inf")

        for source_vertex_id in owned_vertices:
            distance_map = road_distance_map(
                self.board,
                source_vertex_id=source_vertex_id,
                blocked_vertices=occupied_total - {source_vertex_id},
            )
            candidates = candidate_expansion_vertices(
                self.board,
                source_vertex_id=source_vertex_id,
                settlement_blocked_vertices=settlement_blocked,
                distance_map=distance_map,
            )
            for candidate_vertex_id in candidates:
                distance = distance_map[candidate_vertex_id]
                value = self._static_vertex_value(candidate_vertex_id) - (TOPOLOGY_DISTANCE_LAMBDA * distance)
                if value > best_value:
                    best_value = value
                    best_vertex_id = candidate_vertex_id
                    best_distance = distance

        return best_vertex_id, best_distance

    def _apply_production_roll(
        self,
        *,
        owned_vertices: set[int],
        dice_roll: int,
        robber_blocked_tile_id: int | None,
    ) -> tuple[dict[Resource, int], int]:
        gained: dict[Resource, int] = {resource: 0 for resource in RESOURCE_TYPES}
        blocked = 0
        for vertex_id in owned_vertices:
            for tile in self.board.vertex_adjacent_tiles(vertex_id):
                if tile.resource is Resource.DESERT or tile.token_number != dice_roll:
                    continue
                if robber_blocked_tile_id is not None and tile.id == robber_blocked_tile_id:
                    blocked += 1
                    continue
                gained[tile.resource] += 1
        return gained, blocked

    def _select_robber_tile(self, owned_vertices: set[int], *, rng: random.Random) -> int | None:
        if self.config.robber_policy is RobberPolicy.RANDOM_LEGAL_MOVE:
            legal_tiles = [tile.id for tile in self.board.tiles if tile.resource is not Resource.DESERT]
            return rng.choice(legal_tiles) if legal_tiles else None

        our_tiles = {
            tile.id
            for vertex_id in owned_vertices
            for tile in self.board.vertex_adjacent_tiles(vertex_id)
            if tile.resource is not Resource.DESERT
        }
        if not our_tiles:
            return None

        if self.config.robber_policy is RobberPolicy.WORST_CASE_US:
            return max(our_tiles, key=self._tile_pip)

        # TARGET_STRONGEST_OPPONENT approximation: prefer non-owned high-production tiles.
        candidate_tiles = [
            tile.id
            for tile in self.board.tiles
            if tile.resource is not Resource.DESERT and tile.id not in our_tiles
        ]
        if candidate_tiles:
            return max(candidate_tiles, key=self._tile_pip)

        return max(our_tiles, key=self._tile_pip)

    def _recipe_coverage_score(self, rates: Mapping[Resource, float]) -> float:
        road = min(rates[Resource.WOOD], rates[Resource.BRICK]) / 0.11
        settlement = min(
            rates[Resource.WOOD],
            rates[Resource.BRICK],
            rates[Resource.WHEAT],
            rates[Resource.SHEEP],
        ) / 0.09
        city = min(rates[Resource.ORE], rates[Resource.WHEAT]) / 0.10
        dev = min(rates[Resource.ORE], rates[Resource.WHEAT], rates[Resource.SHEEP]) / 0.085
        return (
            min(1.0, road)
            + min(1.0, settlement)
            + (0.8 * min(1.0, city))
            + (0.75 * min(1.0, dev))
        )

    def _fragility_penalty(self, contributions: Mapping[Resource, list[float]]) -> float:
        penalty = 0.0
        for resource in RESOURCE_TYPES:
            values = contributions[resource]
            total = sum(values)
            if total <= 0:
                continue
            max_share = max(values) / total
            penalty += max_share * max_share
            if len(values) == 1:
                penalty += 0.2
        return penalty

    def _port_conversion_score(
        self,
        vertex_ids: Sequence[int],
        rates: Mapping[Resource, float],
    ) -> float:
        ports = self._ports_for_vertices(vertex_ids)
        if not ports:
            return 0.0

        surplus = sum(max(0.0, rates[resource] - 0.09) for resource in RESOURCE_TYPES)
        score = surplus * 0.9
        if PortType.ANY_3TO1 in ports:
            score += surplus * 0.8

        specific_port_map = {
            PortType.WOOD_2TO1: Resource.WOOD,
            PortType.BRICK_2TO1: Resource.BRICK,
            PortType.SHEEP_2TO1: Resource.SHEEP,
            PortType.WHEAT_2TO1: Resource.WHEAT,
            PortType.ORE_2TO1: Resource.ORE,
        }
        for port_type, resource in specific_port_map.items():
            if port_type in ports:
                score += max(0.0, rates[resource] - 0.08) * 1.8
        return score

    def _base_robber_penalty(
        self,
        start_vertices: Sequence[int],
        contributions: Mapping[Resource, list[float]],
    ) -> float:
        red_exposure = 0.0
        for vertex_id in start_vertices:
            for tile in self.board.vertex_adjacent_tiles(vertex_id):
                if tile.token_number in (6, 8):
                    red_exposure += self._tile_pip(tile.id) / 5.0

        single_point_failures = sum(1 for resource in RESOURCE_TYPES if len(contributions[resource]) <= 1 and contributions[resource])
        return (0.35 * red_exposure) + (0.3 * single_point_failures)

    def _robber_policy_multiplier(self) -> float:
        if self.config.robber_policy is RobberPolicy.WORST_CASE_US:
            return 1.35
        if self.config.robber_policy is RobberPolicy.RANDOM_LEGAL_MOVE:
            return 1.0
        return 0.8

    def _static_vertex_value(self, vertex_id: int) -> float:
        cached = self._static_vertex_value_cache.get(vertex_id)
        if cached is not None:
            return cached

        adjacent_tiles = self.board.vertex_adjacent_tiles(vertex_id)
        resources = [tile.resource for tile in adjacent_tiles if tile.resource is not Resource.DESERT]
        expected = self.expected_yield_overrides.get(
            vertex_id,
            float(sum(pip_value(tile.token_number) for tile in adjacent_tiles)),
        )
        unique_count = len(set(resources))
        diversity = {0: 0.0, 1: 0.2, 2: 1.0, 3: 2.0}.get(unique_count, 0.0)

        port_score = 0.0
        if self.config.include_ports:
            port_type = self.board.vertices[vertex_id].port_type
            if port_type is PortType.ANY_3TO1:
                port_score = 0.3
            elif port_type is not None:
                port_score = 0.45

        value = expected + diversity + port_score
        self._static_vertex_value_cache[vertex_id] = value
        return value

    def _tile_pip(self, tile_id: int) -> int:
        tile = self.board.get_tile(tile_id)
        return pip_value(tile.token_number)

    def _ports_for_vertices(self, vertex_ids: Iterable[int]) -> set[PortType]:
        return {
            self.board.vertices[vertex_id].port_type
            for vertex_id in vertex_ids
            if self.board.vertices[vertex_id].port_type is not None
        }

    def _trade_towards_cost(
        self,
        inventory: defaultdict[Resource, int],
        target_cost: Mapping[Resource, int],
        ports: set[PortType],
    ) -> None:
        for missing_resource in target_cost:
            while inventory[missing_resource] < target_cost[missing_resource]:
                trade_resource, trade_ratio = self._best_trade_offer(inventory, missing_resource, ports)
                if trade_resource is None:
                    break
                inventory[trade_resource] -= trade_ratio
                inventory[missing_resource] += 1

    def _best_trade_offer(
        self,
        inventory: defaultdict[Resource, int],
        missing_resource: Resource,
        ports: set[PortType],
    ) -> tuple[Resource | None, int]:
        best_resource = None
        best_ratio = 99
        for resource in RESOURCE_TYPES:
            if resource is missing_resource:
                continue
            ratio = self._trade_ratio_for_resource(resource, ports)
            if ratio is None or inventory[resource] < ratio:
                continue
            if inventory[resource] - ratio < 0:
                continue
            if ratio < best_ratio or (ratio == best_ratio and inventory[resource] > inventory.get(best_resource, 0)):
                best_resource = resource
                best_ratio = ratio
        if best_resource is None:
            return (None, 0)
        return (best_resource, best_ratio)

    def _trade_ratio_for_resource(self, resource: Resource, ports: set[PortType]) -> int | None:
        specific_map = {
            Resource.WOOD: PortType.WOOD_2TO1,
            Resource.BRICK: PortType.BRICK_2TO1,
            Resource.SHEEP: PortType.SHEEP_2TO1,
            Resource.WHEAT: PortType.WHEAT_2TO1,
            Resource.ORE: PortType.ORE_2TO1,
        }
        if specific_map[resource] in ports:
            return 2
        if PortType.ANY_3TO1 in ports:
            return 3
        if self.config.allow_bank_trading:
            return 4
        return None

    @staticmethod
    def _can_afford(inventory: Mapping[Resource, int], cost: Mapping[Resource, int]) -> bool:
        return all(inventory.get(resource, 0) >= amount for resource, amount in cost.items())

    @staticmethod
    def _spend(inventory: defaultdict[Resource, int], cost: Mapping[Resource, int]) -> None:
        for resource, amount in cost.items():
            inventory[resource] -= amount
