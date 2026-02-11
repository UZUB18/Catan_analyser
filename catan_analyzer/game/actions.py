from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from catan_analyzer.domain.board import EdgeKey, Resource


@dataclass(frozen=True)
class GameAction:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


def make_action(kind: str, **data: Any) -> GameAction:
    return GameAction(kind=kind, data=dict(data))


ACTION_PLACE_SETUP_SETTLEMENT = "place_setup_settlement"
ACTION_PLACE_SETUP_ROAD = "place_setup_road"
ACTION_ROLL_DICE = "roll_dice"
ACTION_DISCARD_RESOURCES = "discard_resources"
ACTION_MOVE_ROBBER = "move_robber"
ACTION_STEAL_RESOURCE = "steal_resource"
ACTION_SKIP_STEAL = "skip_steal"
ACTION_TRADE_BANK = "trade_bank"
ACTION_TRADE_PLAYER = "trade_player"
ACTION_END_TRADE_PHASE = "end_trade_phase"
ACTION_BUILD_ROAD = "build_road"
ACTION_BUILD_SETTLEMENT = "build_settlement"
ACTION_BUILD_CITY = "build_city"
ACTION_BUY_DEV_CARD = "buy_dev_card"
ACTION_END_BUILD_PHASE = "end_build_phase"
ACTION_PLAY_KNIGHT = "play_knight"
ACTION_PLAY_ROAD_BUILDING = "play_road_building"
ACTION_PLAY_YEAR_OF_PLENTY = "play_year_of_plenty"
ACTION_PLAY_MONOPOLY = "play_monopoly"
ACTION_REVEAL_VP = "reveal_vp"
ACTION_END_TURN = "end_turn"


def place_setup_settlement(vertex_id: int) -> GameAction:
    return make_action(ACTION_PLACE_SETUP_SETTLEMENT, vertex_id=int(vertex_id))


def place_setup_road(edge: EdgeKey) -> GameAction:
    return make_action(ACTION_PLACE_SETUP_ROAD, edge=tuple(sorted((int(edge[0]), int(edge[1])))))


def roll_dice(value: int | None = None) -> GameAction:
    payload: dict[str, Any] = {}
    if value is not None:
        payload["value"] = int(value)
    return make_action(ACTION_ROLL_DICE, **payload)


def discard_resources(player_id: int, resources: dict[Resource, int]) -> GameAction:
    return make_action(
        ACTION_DISCARD_RESOURCES,
        player_id=int(player_id),
        resources={resource: int(amount) for resource, amount in resources.items()},
    )


def move_robber(tile_id: int) -> GameAction:
    return make_action(ACTION_MOVE_ROBBER, tile_id=int(tile_id))


def steal_resource(target_player_id: int, resource: Resource | None = None) -> GameAction:
    payload: dict[str, Any] = {"target_player_id": int(target_player_id)}
    if resource is not None:
        payload["resource"] = resource
    return make_action(ACTION_STEAL_RESOURCE, **payload)


def skip_steal() -> GameAction:
    return make_action(ACTION_SKIP_STEAL)


def trade_bank(give_resource: Resource, receive_resource: Resource, *, give_amount: int | None = None) -> GameAction:
    payload: dict[str, Any] = {
        "give_resource": give_resource,
        "receive_resource": receive_resource,
    }
    if give_amount is not None:
        payload["give_amount"] = int(give_amount)
    return make_action(ACTION_TRADE_BANK, **payload)


def trade_player(
    target_player_id: int,
    *,
    give: dict[Resource, int],
    receive: dict[Resource, int],
) -> GameAction:
    return make_action(
        ACTION_TRADE_PLAYER,
        target_player_id=int(target_player_id),
        give={resource: int(amount) for resource, amount in give.items()},
        receive={resource: int(amount) for resource, amount in receive.items()},
    )


def end_trade_phase() -> GameAction:
    return make_action(ACTION_END_TRADE_PHASE)


def build_road(edge: EdgeKey) -> GameAction:
    return make_action(ACTION_BUILD_ROAD, edge=tuple(sorted((int(edge[0]), int(edge[1])))))


def build_settlement(vertex_id: int) -> GameAction:
    return make_action(ACTION_BUILD_SETTLEMENT, vertex_id=int(vertex_id))


def build_city(vertex_id: int) -> GameAction:
    return make_action(ACTION_BUILD_CITY, vertex_id=int(vertex_id))


def buy_dev_card() -> GameAction:
    return make_action(ACTION_BUY_DEV_CARD)


def end_build_phase() -> GameAction:
    return make_action(ACTION_END_BUILD_PHASE)


def play_knight(tile_id: int, target_player_id: int | None = None, resource: Resource | None = None) -> GameAction:
    payload: dict[str, Any] = {"tile_id": int(tile_id)}
    if target_player_id is not None:
        payload["target_player_id"] = int(target_player_id)
    if resource is not None:
        payload["resource"] = resource
    return make_action(ACTION_PLAY_KNIGHT, **payload)


def play_road_building(edge_one: EdgeKey, edge_two: EdgeKey | None = None) -> GameAction:
    payload: dict[str, Any] = {"edge_one": tuple(sorted((int(edge_one[0]), int(edge_one[1]))))}
    if edge_two is not None:
        payload["edge_two"] = tuple(sorted((int(edge_two[0]), int(edge_two[1]))))
    return make_action(ACTION_PLAY_ROAD_BUILDING, **payload)


def play_year_of_plenty(resource_one: Resource, resource_two: Resource) -> GameAction:
    return make_action(
        ACTION_PLAY_YEAR_OF_PLENTY,
        resource_one=resource_one,
        resource_two=resource_two,
    )


def play_monopoly(resource: Resource) -> GameAction:
    return make_action(ACTION_PLAY_MONOPOLY, resource=resource)


def reveal_vp(count: int = 1) -> GameAction:
    return make_action(ACTION_REVEAL_VP, count=max(1, int(count)))


def end_turn() -> GameAction:
    return make_action(ACTION_END_TURN)
