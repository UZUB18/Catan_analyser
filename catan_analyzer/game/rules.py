from __future__ import annotations

from typing import Iterable

from catan_analyzer.domain.board import EdgeKey, PortType, Resource

from .actions import (
    ACTION_BUILD_CITY,
    ACTION_BUILD_ROAD,
    ACTION_BUILD_SETTLEMENT,
    ACTION_BUY_DEV_CARD,
    ACTION_DISCARD_RESOURCES,
    ACTION_END_BUILD_PHASE,
    ACTION_END_TRADE_PHASE,
    ACTION_END_TURN,
    ACTION_MOVE_ROBBER,
    ACTION_PLACE_SETUP_ROAD,
    ACTION_PLACE_SETUP_SETTLEMENT,
    ACTION_PLAY_KNIGHT,
    ACTION_PLAY_MONOPOLY,
    ACTION_PLAY_ROAD_BUILDING,
    ACTION_PLAY_YEAR_OF_PLENTY,
    ACTION_REVEAL_VP,
    ACTION_ROLL_DICE,
    ACTION_SKIP_STEAL,
    ACTION_STEAL_RESOURCE,
    ACTION_TRADE_BANK,
    ACTION_TRADE_PLAYER,
    GameAction,
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    discard_resources,
    end_build_phase,
    end_trade_phase,
    end_turn,
    move_robber,
    place_setup_road,
    place_setup_settlement,
    play_knight,
    play_monopoly,
    play_road_building,
    play_year_of_plenty,
    reveal_vp,
    roll_dice,
    skip_steal,
    steal_resource,
    trade_bank,
    trade_player,
)
from .awards import recompute_awards
from .state import (
    DevCardType,
    GamePhase,
    GameState,
    PlayerState,
    count_resource_total,
    next_player_id,
    normalize_edge,
    player_total_victory_points,
    turn_order_from,
)

ROAD_COST = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
}
SETTLEMENT_COST = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
    Resource.WHEAT: 1,
    Resource.SHEEP: 1,
}
CITY_COST = {
    Resource.WHEAT: 2,
    Resource.ORE: 3,
}
DEV_COST = {
    Resource.SHEEP: 1,
    Resource.WHEAT: 1,
    Resource.ORE: 1,
}
PRODUCING_RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)
TURN_ACTION_PHASES = (
    GamePhase.TURN_START,
    GamePhase.TRADE,
    GamePhase.BUILD,
    GamePhase.DEV_PLAY,
)
ROLLED_PHASES = (
    GamePhase.TRADE,
    GamePhase.BUILD,
    GamePhase.DEV_PLAY,
    GamePhase.ROBBER_DISCARD,
    GamePhase.ROBBER_MOVE,
    GamePhase.ROBBER_STEAL,
)
MAX_DOMESTIC_TRADE_ACTIONS = 96


def _resource_map(raw: dict[Resource | str, int] | None) -> dict[Resource, int]:
    if not raw:
        return {}
    normalized: dict[Resource, int] = {}
    for key, value in raw.items():
        resource = key if isinstance(key, Resource) else Resource(str(key))
        amount = int(value)
        if amount <= 0:
            continue
        normalized[resource] = normalized.get(resource, 0) + amount
    return normalized


def _cost_affordable(player: PlayerState, cost: dict[Resource, int]) -> bool:
    return all(player.hand.get(resource, 0) >= amount for resource, amount in cost.items())


def _spend_cost(state: GameState, player: PlayerState, cost: dict[Resource, int]) -> None:
    for resource, amount in cost.items():
        player.hand[resource] -= amount
        state.bank[resource] += amount


def _grant_resource(state: GameState, player: PlayerState, resource: Resource, amount: int) -> None:
    if amount <= 0:
        return
    available = state.bank.get(resource, 0)
    if available <= 0:
        return
    granted = min(available, int(amount))
    state.bank[resource] = available - granted
    player.hand[resource] = player.hand.get(resource, 0) + granted


def _add_player_port_from_vertex(state: GameState, player: PlayerState, vertex_id: int) -> None:
    vertex = state.board.vertices[vertex_id]
    if vertex.port_type is not None:
        player.ports.add(vertex.port_type)


def _setup_settlement_count_for_player(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    return len(player.settlements) + len(player.cities)


def _collect_second_setup_settlement_resources(state: GameState, player: PlayerState, vertex_id: int) -> None:
    for tile in state.board.vertex_adjacent_tiles(vertex_id):
        if tile.resource is Resource.DESERT:
            continue
        _grant_resource(state, player, tile.resource, 1)


def _effective_trade_ratio(player: PlayerState, resource: Resource) -> int:
    specific_port = {
        Resource.WOOD: PortType.WOOD_2TO1,
        Resource.BRICK: PortType.BRICK_2TO1,
        Resource.SHEEP: PortType.SHEEP_2TO1,
        Resource.WHEAT: PortType.WHEAT_2TO1,
        Resource.ORE: PortType.ORE_2TO1,
    }[resource]
    if specific_port in player.ports:
        return 2
    if PortType.ANY_3TO1 in player.ports:
        return 3
    return 4


def _opponent_buildings(state: GameState, player_id: int) -> set[int]:
    occupied: set[int] = set()
    for other_id, other in state.players.items():
        if other_id == player_id:
            continue
        occupied.update(other.settlements)
        occupied.update(other.cities)
    return occupied


def _edge_connects_to_network(state: GameState, player_id: int, edge: EdgeKey) -> bool:
    player = state.players[player_id]
    v1, v2 = edge
    player_buildings = set(player.settlements) | set(player.cities)
    if v1 in player_buildings or v2 in player_buildings:
        return True

    opponent_vertices = _opponent_buildings(state, player_id)
    for vertex_id in edge:
        if vertex_id in opponent_vertices:
            continue
        for owned_edge in player.roads:
            if vertex_id in owned_edge:
                return True
    return False


def _is_legal_setup_road(state: GameState, edge: EdgeKey) -> bool:
    if state.pending_setup_vertex_id is None:
        return False
    edge = normalize_edge(edge)
    if not state.board.edge_exists(edge):
        return False
    if edge in state.all_occupied_edges():
        return False
    return state.pending_setup_vertex_id in edge


def _is_legal_road_build(state: GameState, player_id: int, edge: EdgeKey) -> bool:
    edge = normalize_edge(edge)
    if not state.board.edge_exists(edge):
        return False
    if edge in state.all_occupied_edges():
        return False
    return _edge_connects_to_network(state, player_id, edge)


def _is_legal_settlement_build(state: GameState, player_id: int, vertex_id: int) -> bool:
    occupied = state.all_occupied_vertices()
    if vertex_id in occupied:
        return False
    if not state.board.is_legal_settlement(vertex_id, occupied):
        return False
    player = state.players[player_id]
    return any(vertex_id in edge for edge in player.roads)


def _can_upgrade_to_city(state: GameState, player_id: int, vertex_id: int) -> bool:
    player = state.players[player_id]
    return vertex_id in player.settlements


def _player_has_playable_dev(state: GameState, player_id: int, card_type: DevCardType) -> bool:
    player = state.players[player_id]
    if card_type is DevCardType.VICTORY_POINT:
        return player.total_vp_cards() > player.revealed_vp_cards
    if player.played_non_vp_dev_this_turn:
        return False
    return player.playable_dev_count(card_type) > 0


def _draw_stolen_resource(state: GameState, victim: PlayerState, explicit_resource: Resource | None) -> Resource | None:
    if explicit_resource is not None:
        if victim.hand.get(explicit_resource, 0) > 0:
            return explicit_resource
        return None
    cards: list[Resource] = []
    for resource, amount in victim.hand.items():
        cards.extend([resource] * int(amount))
    if not cards:
        return None
    return state._rng.choice(cards)


def _production_entitlements_for_roll(state: GameState, roll_value: int) -> dict[int, dict[Resource, int]]:
    entitlements: dict[int, dict[Resource, int]] = {
        player_id: {resource: 0 for resource in PRODUCING_RESOURCES}
        for player_id in state.players
    }
    for player_id, player in state.players.items():
        for vertex_id in player.settlements:
            for tile in state.board.vertex_adjacent_tiles(vertex_id):
                if tile.id == state.robber_tile_id or tile.token_number != roll_value:
                    continue
                if tile.resource in PRODUCING_RESOURCES:
                    entitlements[player_id][tile.resource] += 1
        for vertex_id in player.cities:
            for tile in state.board.vertex_adjacent_tiles(vertex_id):
                if tile.id == state.robber_tile_id or tile.token_number != roll_value:
                    continue
                if tile.resource in PRODUCING_RESOURCES:
                    entitlements[player_id][tile.resource] += 2
    return entitlements


def _apply_roll_production(state: GameState, roll_value: int) -> None:
    entitlements = _production_entitlements_for_roll(state, roll_value)
    for resource in PRODUCING_RESOURCES:
        owed_by_player = {
            player_id: int(allocation.get(resource, 0))
            for player_id, allocation in entitlements.items()
            if int(allocation.get(resource, 0)) > 0
        }
        if not owed_by_player:
            continue
        bank_available = int(state.bank.get(resource, 0))
        if bank_available <= 0:
            continue

        total_owed = int(sum(owed_by_player.values()))
        if total_owed <= bank_available:
            for player_id, amount in owed_by_player.items():
                _grant_resource(state, state.players[player_id], resource, amount)
            continue

        if len(owed_by_player) == 1:
            only_player_id = next(iter(owed_by_player))
            _grant_resource(
                state,
                state.players[only_player_id],
                resource,
                min(bank_available, owed_by_player[only_player_id]),
            )


def _eligible_robber_targets(state: GameState, tile_id: int, current_player_id: int) -> list[int]:
    targets: list[int] = []
    for player_id, player in state.players.items():
        if player_id == current_player_id:
            continue
        if player.card_count() <= 0:
            continue
        touches_tile = False
        for vertex_id in player.settlements | player.cities:
            if tile_id in state.board.vertices[vertex_id].adjacent_hex_ids:
                touches_tile = True
                break
        if touches_tile:
            targets.append(player_id)
    return targets


def _check_for_winner(state: GameState) -> None:
    if state.phase == GamePhase.GAME_OVER:
        return
    player_id = state.current_player_id
    if player_total_victory_points(state, player_id) >= 10:
        state.winner_id = player_id
        state.phase = GamePhase.GAME_OVER


def _post_turn_reset(player: PlayerState) -> None:
    player.played_non_vp_dev_this_turn = False
    for card_type in list(player.new_dev_cards.keys()):
        player.new_dev_cards[card_type] = 0


def _auto_discard_plan(player: PlayerState, required: int) -> dict[Resource, int]:
    remaining = required
    plan: dict[Resource, int] = {}
    resources_desc = sorted(
        PRODUCING_RESOURCES,
        key=lambda resource: player.hand.get(resource, 0),
        reverse=True,
    )
    for resource in resources_desc:
        if remaining <= 0:
            break
        available = player.hand.get(resource, 0)
        if available <= 0:
            continue
        amount = min(available, remaining)
        plan[resource] = amount
        remaining -= amount
    return plan


def _record_event(state: GameState, text: str) -> None:
    state.event_log.append(text)
    if len(state.event_log) > 120:
        state.event_log = state.event_log[-120:]


def _is_turn_action_phase(phase: GamePhase) -> bool:
    return phase in TURN_ACTION_PHASES


def _can_roll_dice_now(state: GameState) -> bool:
    return _is_turn_action_phase(state.phase) and not state.turn_has_rolled


def _can_take_post_roll_actions(state: GameState) -> bool:
    return _is_turn_action_phase(state.phase) and state.turn_has_rolled


def _candidate_trade_maps(hand: dict[Resource, int], *, max_cards: int = 2) -> list[dict[Resource, int]]:
    maps: list[dict[Resource, int]] = []
    resources = [resource for resource in PRODUCING_RESOURCES if hand.get(resource, 0) > 0]
    for resource in resources:
        upper = min(int(hand.get(resource, 0)), max_cards)
        for amount in range(1, upper + 1):
            maps.append({resource: amount})
    if max_cards >= 2:
        for i, first in enumerate(resources):
            if hand.get(first, 0) < 1:
                continue
            for second in resources[i + 1 :]:
                if hand.get(second, 0) < 1:
                    continue
                maps.append({first: 1, second: 1})
    return maps


def _append_trade_actions(state: GameState, player_id: int, actions: list[GameAction]) -> None:
    player = state.players[player_id]
    for give_resource in PRODUCING_RESOURCES:
        give_ratio = _effective_trade_ratio(player, give_resource)
        if player.hand.get(give_resource, 0) < give_ratio:
            continue
        for receive_resource in PRODUCING_RESOURCES:
            if receive_resource is give_resource:
                continue
            if state.bank.get(receive_resource, 0) <= 0:
                continue
            actions.append(trade_bank(give_resource, receive_resource, give_amount=give_ratio))

    trade_count = 0
    give_options = _candidate_trade_maps(player.hand, max_cards=2)
    for target_id, target in state.players.items():
        if target_id == player_id:
            continue
        receive_options = _candidate_trade_maps(target.hand, max_cards=2)
        for give_map in give_options:
            for receive_map in receive_options:
                if set(give_map).intersection(receive_map):
                    continue
                actions.append(
                    trade_player(
                        target_id,
                        give=give_map,
                        receive=receive_map,
                    )
                )
                trade_count += 1
                if trade_count >= MAX_DOMESTIC_TRADE_ACTIONS:
                    return


def _append_build_actions(state: GameState, player_id: int, actions: list[GameAction]) -> None:
    player = state.players[player_id]

    if _cost_affordable(player, ROAD_COST):
        occupied_edges = state.all_occupied_edges()
        for edge in sorted(state.board.edges.keys()):
            edge = normalize_edge(edge)
            if edge in occupied_edges:
                continue
            if _is_legal_road_build(state, player_id, edge):
                actions.append(build_road(edge))

    if _cost_affordable(player, SETTLEMENT_COST):
        for vertex_id in sorted(state.board.vertices.keys()):
            if _is_legal_settlement_build(state, player_id, vertex_id):
                actions.append(build_settlement(vertex_id))

    if _cost_affordable(player, CITY_COST):
        for vertex_id in sorted(player.settlements):
            actions.append(build_city(vertex_id))

    if _cost_affordable(player, DEV_COST) and state.dev_deck:
        actions.append(buy_dev_card())


def _append_dev_actions(state: GameState, player_id: int, actions: list[GameAction]) -> None:
    if _player_has_playable_dev(state, player_id, DevCardType.KNIGHT):
        for tile in state.board.tiles:
            if tile.id == state.robber_tile_id:
                continue
            actions.append(play_knight(tile.id))

    if _player_has_playable_dev(state, player_id, DevCardType.ROAD_BUILDING):
        legal_edges = [
            edge
            for edge in sorted(state.board.edges.keys())
            if _is_legal_road_build(state, player_id, edge) and normalize_edge(edge) not in state.all_occupied_edges()
        ]
        for first_edge in legal_edges[:8]:
            actions.append(play_road_building(first_edge))

    if _player_has_playable_dev(state, player_id, DevCardType.YEAR_OF_PLENTY):
        available = [resource for resource in PRODUCING_RESOURCES if state.bank.get(resource, 0) > 0]
        for first in available:
            for second in available:
                actions.append(play_year_of_plenty(first, second))

    if _player_has_playable_dev(state, player_id, DevCardType.MONOPOLY):
        for resource in PRODUCING_RESOURCES:
            actions.append(play_monopoly(resource))

    if _player_has_playable_dev(state, player_id, DevCardType.VICTORY_POINT):
        actions.append(reveal_vp())


def _list_turn_actions(state: GameState, player_id: int) -> list[GameAction]:
    actions: list[GameAction] = []
    if _can_roll_dice_now(state):
        actions.append(roll_dice())

    _append_dev_actions(state, player_id, actions)

    if _can_take_post_roll_actions(state):
        _append_trade_actions(state, player_id, actions)
        _append_build_actions(state, player_id, actions)
        actions.append(end_trade_phase())
        actions.append(end_build_phase())
        actions.append(end_turn())
    return actions


def list_legal_actions(state: GameState) -> list[GameAction]:
    if state.phase is GamePhase.GAME_OVER:
        return []

    player_id = state.current_player_id

    if state.phase is GamePhase.SETUP_SETTLEMENT:
        occupied = state.all_occupied_vertices()
        return [place_setup_settlement(vertex_id) for vertex_id in state.board.legal_settlement_vertices(occupied)]

    if state.phase is GamePhase.SETUP_ROAD:
        if state.pending_setup_vertex_id is None:
            return []
        occupied_edges = state.all_occupied_edges()
        actions: list[GameAction] = []
        anchor_vertex = state.pending_setup_vertex_id
        for neighbor_id in state.board.vertices[anchor_vertex].adjacent_vertex_ids:
            edge = normalize_edge((anchor_vertex, neighbor_id))
            if edge in occupied_edges:
                continue
            if state.board.edge_exists(edge):
                actions.append(place_setup_road(edge))
        return actions

    if _is_turn_action_phase(state.phase):
        return _list_turn_actions(state, player_id)

    if state.phase is GamePhase.ROBBER_DISCARD:
        if not state.discard_queue:
            return []
        discard_player_id = state.discard_queue[0]
        discard_player = state.players[discard_player_id]
        required = discard_player.card_count() // 2
        return [discard_resources(discard_player_id, _auto_discard_plan(discard_player, required))]

    if state.phase is GamePhase.ROBBER_MOVE:
        return [
            move_robber(tile.id)
            for tile in state.board.tiles
            if tile.id != state.robber_tile_id
        ]

    if state.phase is GamePhase.ROBBER_STEAL:
        targets = [target_id for target_id in state.pending_steal_target_ids if state.players[target_id].card_count() > 0]
        if not targets:
            return [skip_steal()]
        return [steal_resource(target_id) for target_id in targets]

    return []


def apply_action(state: GameState, action: GameAction) -> GameState:
    if state.phase is GamePhase.GAME_OVER:
        return state.clone()

    next_state = state.clone()
    kind = action.kind
    data = action.data

    if kind == ACTION_PLACE_SETUP_SETTLEMENT:
        if next_state.phase is not GamePhase.SETUP_SETTLEMENT:
            raise ValueError("Setup settlement placement is only valid during setup settlement phase.")
        vertex_id = int(data["vertex_id"])
        occupied = next_state.all_occupied_vertices()
        if not next_state.board.is_legal_settlement(vertex_id, occupied):
            raise ValueError(f"Illegal setup settlement vertex: {vertex_id}.")
        player = next_state.players[next_state.current_player_id]
        player.settlements.add(vertex_id)
        _add_player_port_from_vertex(next_state, player, vertex_id)
        next_state.pending_setup_vertex_id = vertex_id
        next_state.phase = GamePhase.SETUP_ROAD
        if _setup_settlement_count_for_player(next_state, player.player_id) == 2:
            _collect_second_setup_settlement_resources(next_state, player, vertex_id)
        _record_event(next_state, f"P{player.player_id} setup settlement at V{vertex_id}.")
        return next_state

    if kind == ACTION_PLACE_SETUP_ROAD:
        if next_state.phase is not GamePhase.SETUP_ROAD:
            raise ValueError("Setup road placement is only valid during setup road phase.")
        edge = normalize_edge(tuple(data["edge"]))  # type: ignore[arg-type]
        if not _is_legal_setup_road(next_state, edge):
            raise ValueError(f"Illegal setup road edge: {edge}.")
        player = next_state.players[next_state.current_player_id]
        player.roads.add(edge)
        next_state.pending_setup_vertex_id = None
        next_state.setup_index += 1
        recompute_awards(next_state)
        if next_state.setup_index >= len(next_state.setup_order):
            next_state.phase = GamePhase.TURN_START
            next_state.current_player_id = 1
            _record_event(next_state, "Setup complete.")
            return next_state
        next_state.current_player_id = next_state.setup_order[next_state.setup_index]
        next_state.phase = GamePhase.SETUP_SETTLEMENT
        _record_event(next_state, f"P{player.player_id} setup road {edge[0]}-{edge[1]}.")
        return next_state

    if kind == ACTION_ROLL_DICE:
        if not _can_roll_dice_now(next_state):
            raise ValueError("Rolling dice is only valid once before post-roll actions.")
        value = data.get("value")
        if value is None:
            roll_value = int(next_state._rng.randint(1, 6) + next_state._rng.randint(1, 6))
        else:
            roll_value = int(value)
        if roll_value < 2 or roll_value > 12:
            raise ValueError("Dice roll must be in [2, 12].")
        next_state.dice_roll = roll_value
        next_state.turn_has_rolled = True
        _record_event(next_state, f"P{next_state.current_player_id} rolled {roll_value}.")
        if roll_value == 7:
            next_state.discard_queue = [
                player_id
                for player_id in turn_order_from(next_state.current_player_id, next_state.player_count)
                if next_state.players[player_id].card_count() > 7
            ]
            if next_state.discard_queue:
                next_state.phase = GamePhase.ROBBER_DISCARD
            else:
                next_state.phase = GamePhase.ROBBER_MOVE
        else:
            _apply_roll_production(next_state, roll_value)
            next_state.phase = GamePhase.TRADE
        return next_state

    if kind == ACTION_DISCARD_RESOURCES:
        if next_state.phase is not GamePhase.ROBBER_DISCARD:
            raise ValueError("Discard action is only valid during robber discard phase.")
        if not next_state.discard_queue:
            raise ValueError("No pending discard queue.")
        player_id = int(data["player_id"])
        if player_id != next_state.discard_queue[0]:
            raise ValueError("Discard order mismatch.")
        player = next_state.players[player_id]
        resources = _resource_map(data.get("resources"))  # type: ignore[arg-type]
        required = player.card_count() // 2
        if count_resource_total(resources) != required:
            raise ValueError(f"Discard must contain exactly {required} cards.")
        for resource, amount in resources.items():
            if player.hand.get(resource, 0) < amount:
                raise ValueError("Cannot discard resources not in hand.")
        for resource, amount in resources.items():
            player.hand[resource] -= amount
            next_state.bank[resource] += amount
        next_state.discard_queue.pop(0)
        _record_event(next_state, f"P{player_id} discarded {required} cards.")
        if next_state.discard_queue:
            next_state.phase = GamePhase.ROBBER_DISCARD
        else:
            next_state.phase = GamePhase.ROBBER_MOVE
        return next_state

    if kind == ACTION_MOVE_ROBBER:
        if next_state.phase is not GamePhase.ROBBER_MOVE:
            raise ValueError("Robber move is only valid during robber move phase.")
        tile_id = int(data["tile_id"])
        if tile_id == next_state.robber_tile_id:
            raise ValueError("Robber must move to a different tile.")
        if tile_id not in {tile.id for tile in next_state.board.tiles}:
            raise ValueError("Invalid robber destination tile.")
        next_state.robber_tile_id = tile_id
        targets = _eligible_robber_targets(next_state, tile_id, next_state.current_player_id)
        next_state.pending_steal_target_ids = targets
        next_state.phase = GamePhase.ROBBER_STEAL if targets else GamePhase.TRADE
        _record_event(next_state, f"P{next_state.current_player_id} moved robber to T{tile_id}.")
        return next_state

    if kind == ACTION_STEAL_RESOURCE:
        if next_state.phase is not GamePhase.ROBBER_STEAL:
            raise ValueError("Steal action is only valid during robber steal phase.")
        target_player_id = int(data["target_player_id"])
        if target_player_id not in next_state.pending_steal_target_ids:
            raise ValueError("Target player is not eligible for stealing.")
        target = next_state.players[target_player_id]
        acting = next_state.players[next_state.current_player_id]
        explicit_resource = data.get("resource")
        resource = _draw_stolen_resource(
            next_state,
            target,
            explicit_resource if isinstance(explicit_resource, Resource) else None,
        )
        if resource is not None and target.hand.get(resource, 0) > 0:
            target.hand[resource] -= 1
            acting.hand[resource] += 1
            _record_event(next_state, f"P{next_state.current_player_id} stole {resource.value} from P{target_player_id}.")
        next_state.pending_steal_target_ids = []
        next_state.phase = GamePhase.TRADE
        return next_state

    if kind == ACTION_SKIP_STEAL:
        if next_state.phase is not GamePhase.ROBBER_STEAL:
            raise ValueError("Skip steal is only valid during robber steal phase.")
        next_state.pending_steal_target_ids = []
        next_state.phase = GamePhase.TRADE
        return next_state

    if kind == ACTION_TRADE_BANK:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("Bank trading is only valid after rolling.")
        player = next_state.players[next_state.current_player_id]
        give_resource = data["give_resource"]
        receive_resource = data["receive_resource"]
        if not isinstance(give_resource, Resource) or not isinstance(receive_resource, Resource):
            raise ValueError("Trade resources must be valid resource enums.")
        if give_resource is receive_resource:
            raise ValueError("Bank trade must exchange different resources.")
        ratio = _effective_trade_ratio(player, give_resource)
        explicit_amount = data.get("give_amount")
        give_amount = int(explicit_amount) if explicit_amount is not None else ratio
        if give_amount != ratio:
            raise ValueError("Provided give_amount does not match effective trade ratio.")
        if player.hand.get(give_resource, 0) < give_amount:
            raise ValueError("Insufficient resources for bank trade.")
        if next_state.bank.get(receive_resource, 0) <= 0:
            raise ValueError("Bank has no requested resource for trade.")
        player.hand[give_resource] -= give_amount
        next_state.bank[give_resource] += give_amount
        next_state.bank[receive_resource] -= 1
        player.hand[receive_resource] += 1
        next_state.phase = GamePhase.TRADE
        _record_event(next_state, f"P{player.player_id} bank traded {give_amount} {give_resource.value} for {receive_resource.value}.")
        return next_state

    if kind == ACTION_TRADE_PLAYER:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("Player trading is only valid after rolling.")
        player = next_state.players[next_state.current_player_id]
        target_player_id = int(data["target_player_id"])
        if target_player_id == player.player_id:
            raise ValueError("Cannot trade with self.")
        if target_player_id not in next_state.players:
            raise ValueError("Invalid target player.")
        target = next_state.players[target_player_id]
        give = _resource_map(data.get("give"))  # type: ignore[arg-type]
        receive = _resource_map(data.get("receive"))  # type: ignore[arg-type]
        if not give or not receive:
            raise ValueError("Player trade must include both give and receive resources.")
        if set(give).intersection(receive):
            raise ValueError("Domestic trade cannot exchange the same resource type on both sides.")
        for resource, amount in give.items():
            if player.hand.get(resource, 0) < amount:
                raise ValueError("Offering player lacks resources.")
        for resource, amount in receive.items():
            if target.hand.get(resource, 0) < amount:
                raise ValueError("Target player lacks requested resources.")
        for resource, amount in give.items():
            player.hand[resource] -= amount
            target.hand[resource] += amount
        for resource, amount in receive.items():
            target.hand[resource] -= amount
            player.hand[resource] += amount
        next_state.phase = GamePhase.TRADE
        _record_event(next_state, f"P{player.player_id} traded with P{target_player_id}.")
        return next_state

    if kind == ACTION_END_TRADE_PHASE:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("End trade phase is only valid after rolling.")
        next_state.phase = GamePhase.BUILD
        return next_state

    if kind == ACTION_BUILD_ROAD:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("Road build is only valid after rolling.")
        player = next_state.players[next_state.current_player_id]
        edge = normalize_edge(tuple(data["edge"]))  # type: ignore[arg-type]
        if not _cost_affordable(player, ROAD_COST):
            raise ValueError("Cannot afford road cost.")
        if not _is_legal_road_build(next_state, player.player_id, edge):
            raise ValueError("Illegal road placement.")
        _spend_cost(next_state, player, ROAD_COST)
        player.roads.add(edge)
        next_state.phase = GamePhase.BUILD
        recompute_awards(next_state)
        _check_for_winner(next_state)
        _record_event(next_state, f"P{player.player_id} built road {edge[0]}-{edge[1]}.")
        return next_state

    if kind == ACTION_BUILD_SETTLEMENT:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("Settlement build is only valid after rolling.")
        player = next_state.players[next_state.current_player_id]
        vertex_id = int(data["vertex_id"])
        if not _cost_affordable(player, SETTLEMENT_COST):
            raise ValueError("Cannot afford settlement cost.")
        if not _is_legal_settlement_build(next_state, player.player_id, vertex_id):
            raise ValueError("Illegal settlement placement.")
        _spend_cost(next_state, player, SETTLEMENT_COST)
        player.settlements.add(vertex_id)
        _add_player_port_from_vertex(next_state, player, vertex_id)
        next_state.phase = GamePhase.BUILD
        recompute_awards(next_state)
        _check_for_winner(next_state)
        _record_event(next_state, f"P{player.player_id} built settlement at V{vertex_id}.")
        return next_state

    if kind == ACTION_BUILD_CITY:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("City build is only valid after rolling.")
        player = next_state.players[next_state.current_player_id]
        vertex_id = int(data["vertex_id"])
        if not _cost_affordable(player, CITY_COST):
            raise ValueError("Cannot afford city cost.")
        if not _can_upgrade_to_city(next_state, player.player_id, vertex_id):
            raise ValueError("City upgrade requires an owned settlement.")
        _spend_cost(next_state, player, CITY_COST)
        player.settlements.remove(vertex_id)
        player.cities.add(vertex_id)
        next_state.phase = GamePhase.BUILD
        _check_for_winner(next_state)
        _record_event(next_state, f"P{player.player_id} upgraded city at V{vertex_id}.")
        return next_state

    if kind == ACTION_BUY_DEV_CARD:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("Buying dev card is only valid after rolling.")
        player = next_state.players[next_state.current_player_id]
        if not _cost_affordable(player, DEV_COST):
            raise ValueError("Cannot afford development card cost.")
        if not next_state.dev_deck:
            raise ValueError("Development deck is empty.")
        _spend_cost(next_state, player, DEV_COST)
        card = next_state.dev_deck.pop()
        player.dev_cards[card] = player.dev_cards.get(card, 0) + 1
        player.new_dev_cards[card] = player.new_dev_cards.get(card, 0) + 1
        next_state.phase = GamePhase.BUILD
        _record_event(next_state, f"P{player.player_id} bought a development card.")
        if card is DevCardType.VICTORY_POINT and player_total_victory_points(next_state, player.player_id) >= 10:
            player.revealed_vp_cards = min(player.total_vp_cards(), player.revealed_vp_cards + 1)
            _record_event(next_state, f"P{player.player_id} revealed a victory point card.")
            _check_for_winner(next_state)
        return next_state

    if kind == ACTION_END_BUILD_PHASE:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("End build phase is only valid after rolling.")
        next_state.phase = GamePhase.DEV_PLAY
        _check_for_winner(next_state)
        return next_state

    if kind == ACTION_PLAY_KNIGHT:
        if not _is_turn_action_phase(next_state.phase):
            raise ValueError("Knight play is only valid during a player turn.")
        player = next_state.players[next_state.current_player_id]
        if not _player_has_playable_dev(next_state, player.player_id, DevCardType.KNIGHT):
            raise ValueError("Knight card is not playable.")
        tile_id = int(data["tile_id"])
        if tile_id == next_state.robber_tile_id:
            raise ValueError("Knight must move robber to a new tile.")
        if tile_id not in {tile.id for tile in next_state.board.tiles}:
            raise ValueError("Invalid robber target tile.")
        player.dev_cards[DevCardType.KNIGHT] -= 1
        player.played_knights += 1
        player.played_non_vp_dev_this_turn = True
        next_state.robber_tile_id = tile_id
        targets = _eligible_robber_targets(next_state, tile_id, player.player_id)
        target_player_id = data.get("target_player_id")
        explicit_resource = data.get("resource")
        if target_player_id is None and targets:
            target_player_id = targets[0]
        if target_player_id is not None:
            target_player_id = int(target_player_id)
        if target_player_id in targets:
            victim = next_state.players[target_player_id]
            resource = _draw_stolen_resource(
                next_state,
                victim,
                explicit_resource if isinstance(explicit_resource, Resource) else None,
            )
            if resource is not None and victim.hand.get(resource, 0) > 0:
                victim.hand[resource] -= 1
                player.hand[resource] += 1
        next_state.phase = GamePhase.DEV_PLAY if next_state.turn_has_rolled else GamePhase.TURN_START
        recompute_awards(next_state)
        _check_for_winner(next_state)
        _record_event(next_state, f"P{player.player_id} played Knight.")
        return next_state

    if kind == ACTION_PLAY_ROAD_BUILDING:
        if not _is_turn_action_phase(next_state.phase):
            raise ValueError("Road Building play is only valid during a player turn.")
        player = next_state.players[next_state.current_player_id]
        if not _player_has_playable_dev(next_state, player.player_id, DevCardType.ROAD_BUILDING):
            raise ValueError("Road Building card is not playable.")
        edge_one = normalize_edge(tuple(data["edge_one"]))  # type: ignore[arg-type]
        edge_two_raw = data.get("edge_two")
        edge_two = normalize_edge(tuple(edge_two_raw)) if edge_two_raw is not None else None
        if not _is_legal_road_build(next_state, player.player_id, edge_one):
            raise ValueError("First Road Building edge is illegal.")
        player.dev_cards[DevCardType.ROAD_BUILDING] -= 1
        player.played_non_vp_dev_this_turn = True
        player.roads.add(edge_one)
        if edge_two is not None and _is_legal_road_build(next_state, player.player_id, edge_two):
            player.roads.add(edge_two)
        next_state.phase = GamePhase.DEV_PLAY if next_state.turn_has_rolled else GamePhase.TURN_START
        recompute_awards(next_state)
        _check_for_winner(next_state)
        _record_event(next_state, f"P{player.player_id} played Road Building.")
        return next_state

    if kind == ACTION_PLAY_YEAR_OF_PLENTY:
        if not _is_turn_action_phase(next_state.phase):
            raise ValueError("Year of Plenty play is only valid during a player turn.")
        player = next_state.players[next_state.current_player_id]
        if not _player_has_playable_dev(next_state, player.player_id, DevCardType.YEAR_OF_PLENTY):
            raise ValueError("Year of Plenty card is not playable.")
        resource_one = data["resource_one"]
        resource_two = data["resource_two"]
        if not isinstance(resource_one, Resource) or not isinstance(resource_two, Resource):
            raise ValueError("Year of Plenty resources must be valid resource enums.")
        if next_state.bank.get(resource_one, 0) <= 0:
            raise ValueError("Bank does not have first requested resource.")
        if next_state.bank.get(resource_two, 0) <= 0:
            raise ValueError("Bank does not have second requested resource.")
        player.dev_cards[DevCardType.YEAR_OF_PLENTY] -= 1
        player.played_non_vp_dev_this_turn = True
        _grant_resource(next_state, player, resource_one, 1)
        _grant_resource(next_state, player, resource_two, 1)
        next_state.phase = GamePhase.DEV_PLAY if next_state.turn_has_rolled else GamePhase.TURN_START
        _check_for_winner(next_state)
        _record_event(next_state, f"P{player.player_id} played Year of Plenty.")
        return next_state

    if kind == ACTION_PLAY_MONOPOLY:
        if not _is_turn_action_phase(next_state.phase):
            raise ValueError("Monopoly play is only valid during a player turn.")
        player = next_state.players[next_state.current_player_id]
        if not _player_has_playable_dev(next_state, player.player_id, DevCardType.MONOPOLY):
            raise ValueError("Monopoly card is not playable.")
        resource = data["resource"]
        if not isinstance(resource, Resource):
            raise ValueError("Monopoly resource must be a valid resource enum.")
        player.dev_cards[DevCardType.MONOPOLY] -= 1
        player.played_non_vp_dev_this_turn = True
        total_taken = 0
        for target_id, target in next_state.players.items():
            if target_id == player.player_id:
                continue
            amount = target.hand.get(resource, 0)
            if amount <= 0:
                continue
            target.hand[resource] -= amount
            player.hand[resource] += amount
            total_taken += amount
        next_state.phase = GamePhase.DEV_PLAY if next_state.turn_has_rolled else GamePhase.TURN_START
        _check_for_winner(next_state)
        _record_event(next_state, f"P{player.player_id} played Monopoly ({resource.value}, {total_taken}).")
        return next_state

    if kind == ACTION_REVEAL_VP:
        if not _is_turn_action_phase(next_state.phase):
            raise ValueError("Reveal VP is only valid during a player turn.")
        player = next_state.players[next_state.current_player_id]
        if player.total_vp_cards() <= player.revealed_vp_cards:
            raise ValueError("No hidden victory point cards available to reveal.")
        count = int(data.get("count", 1))
        count = max(1, count)
        available = player.total_vp_cards() - player.revealed_vp_cards
        player.revealed_vp_cards += min(available, count)
        next_state.phase = GamePhase.DEV_PLAY if next_state.turn_has_rolled else GamePhase.TURN_START
        _check_for_winner(next_state)
        _record_event(next_state, f"P{player.player_id} revealed a victory point card.")
        return next_state

    if kind == ACTION_END_TURN:
        if not _can_take_post_roll_actions(next_state):
            raise ValueError("End turn is only valid after rolling.")
        current_player = next_state.players[next_state.current_player_id]
        _post_turn_reset(current_player)
        next_state.current_player_id = next_player_id(next_state.current_player_id, next_state.player_count)
        next_state.turn_number += 1
        next_state.phase = GamePhase.TURN_START
        next_state.dice_roll = None
        next_state.turn_has_rolled = False
        _record_event(next_state, f"Turn {next_state.turn_number} start (P{next_state.current_player_id}).")
        return next_state

    raise ValueError(f"Unsupported action kind: {kind}")


def run_forced_action(state: GameState, action_kind: str) -> GameState:
    legal = list_legal_actions(state)
    matches = [action for action in legal if action.kind == action_kind]
    if not matches:
        raise ValueError(f"No legal action of kind {action_kind} available in phase {state.phase}.")
    return apply_action(state, matches[0])
