from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable

from catan_analyzer.domain.board import BoardState, EdgeKey, PortType, Resource


class GamePhase(str, Enum):
    SETUP_SETTLEMENT = "setup_settlement"
    SETUP_ROAD = "setup_road"
    TURN_START = "turn_start"
    ROBBER_DISCARD = "robber_discard"
    ROBBER_MOVE = "robber_move"
    ROBBER_STEAL = "robber_steal"
    TRADE = "trade"
    BUILD = "build"
    DEV_PLAY = "dev_play"
    GAME_OVER = "game_over"


class DevCardType(str, Enum):
    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"


DEV_DECK_COMPOSITION: dict[DevCardType, int] = {
    DevCardType.KNIGHT: 14,
    DevCardType.VICTORY_POINT: 5,
    DevCardType.ROAD_BUILDING: 2,
    DevCardType.YEAR_OF_PLENTY: 2,
    DevCardType.MONOPOLY: 2,
}

RESOURCE_BANK_START = 19


@dataclass
class PlayerState:
    player_id: int
    hand: Dict[Resource, int] = field(default_factory=dict)
    settlements: set[int] = field(default_factory=set)
    cities: set[int] = field(default_factory=set)
    roads: set[EdgeKey] = field(default_factory=set)
    ports: set[PortType] = field(default_factory=set)
    dev_cards: Dict[DevCardType, int] = field(default_factory=dict)
    new_dev_cards: Dict[DevCardType, int] = field(default_factory=dict)
    revealed_vp_cards: int = 0
    played_knights: int = 0
    played_non_vp_dev_this_turn: bool = False

    def clone(self) -> "PlayerState":
        return PlayerState(
            player_id=self.player_id,
            hand=dict(self.hand),
            settlements=set(self.settlements),
            cities=set(self.cities),
            roads=set(self.roads),
            ports=set(self.ports),
            dev_cards=dict(self.dev_cards),
            new_dev_cards=dict(self.new_dev_cards),
            revealed_vp_cards=int(self.revealed_vp_cards),
            played_knights=int(self.played_knights),
            played_non_vp_dev_this_turn=bool(self.played_non_vp_dev_this_turn),
        )

    def card_count(self) -> int:
        return int(sum(self.hand.values()))

    def total_vp_cards(self) -> int:
        return int(self.dev_cards.get(DevCardType.VICTORY_POINT, 0))

    def hidden_vp_cards(self) -> int:
        return max(0, self.total_vp_cards() - int(self.revealed_vp_cards))

    def playable_dev_count(self, card_type: DevCardType) -> int:
        return max(
            0,
            int(self.dev_cards.get(card_type, 0)) - int(self.new_dev_cards.get(card_type, 0)),
        )


@dataclass
class GameState:
    board: BoardState
    player_count: int
    players: Dict[int, PlayerState]
    phase: GamePhase
    current_player_id: int
    turn_number: int
    robber_tile_id: int
    bank: Dict[Resource, int]
    dev_deck: list[DevCardType]
    setup_order: tuple[int, ...]
    setup_index: int = 0
    pending_setup_vertex_id: int | None = None
    discard_queue: list[int] = field(default_factory=list)
    pending_steal_target_ids: list[int] = field(default_factory=list)
    dice_roll: int | None = None
    turn_has_rolled: bool = False
    longest_road_owner: int | None = None
    longest_road_length: int = 0
    largest_army_owner: int | None = None
    largest_army_size: int = 0
    winner_id: int | None = None
    event_log: list[str] = field(default_factory=list)
    _rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    def clone(self) -> "GameState":
        cloned = GameState(
            board=self.board,
            player_count=self.player_count,
            players={player_id: player.clone() for player_id, player in self.players.items()},
            phase=self.phase,
            current_player_id=self.current_player_id,
            turn_number=self.turn_number,
            robber_tile_id=self.robber_tile_id,
            bank=dict(self.bank),
            dev_deck=list(self.dev_deck),
            setup_order=tuple(self.setup_order),
            setup_index=self.setup_index,
            pending_setup_vertex_id=self.pending_setup_vertex_id,
            discard_queue=list(self.discard_queue),
            pending_steal_target_ids=list(self.pending_steal_target_ids),
            dice_roll=self.dice_roll,
            turn_has_rolled=self.turn_has_rolled,
            longest_road_owner=self.longest_road_owner,
            longest_road_length=self.longest_road_length,
            largest_army_owner=self.largest_army_owner,
            largest_army_size=self.largest_army_size,
            winner_id=self.winner_id,
            event_log=list(self.event_log),
            _rng=random.Random(),
        )
        cloned._rng.setstate(self._rng.getstate())
        return cloned

    def all_occupied_vertices(self) -> set[int]:
        occupied: set[int] = set()
        for player in self.players.values():
            occupied.update(player.settlements)
            occupied.update(player.cities)
        return occupied

    def all_occupied_edges(self) -> set[EdgeKey]:
        occupied: set[EdgeKey] = set()
        for player in self.players.values():
            occupied.update(player.roads)
        return occupied


def initial_bank_counts() -> Dict[Resource, int]:
    return {
        Resource.WOOD: RESOURCE_BANK_START,
        Resource.BRICK: RESOURCE_BANK_START,
        Resource.SHEEP: RESOURCE_BANK_START,
        Resource.WHEAT: RESOURCE_BANK_START,
        Resource.ORE: RESOURCE_BANK_START,
    }


def make_setup_order(player_count: int) -> tuple[int, ...]:
    if player_count < 2 or player_count > 4:
        raise ValueError("player_count must be between 2 and 4.")
    forward = tuple(range(1, player_count + 1))
    backward = tuple(range(player_count, 0, -1))
    return forward + backward


def create_shuffled_dev_deck(rng: random.Random) -> list[DevCardType]:
    deck: list[DevCardType] = []
    for card_type, count in DEV_DECK_COMPOSITION.items():
        deck.extend([card_type] * count)
    rng.shuffle(deck)
    return deck


def initialize_game_state(
    board: BoardState,
    *,
    player_count: int = 4,
    seed: int | None = None,
) -> GameState:
    rng = random.Random(seed)
    players = {
        player_id: PlayerState(
            player_id=player_id,
            hand={
                Resource.WOOD: 0,
                Resource.BRICK: 0,
                Resource.SHEEP: 0,
                Resource.WHEAT: 0,
                Resource.ORE: 0,
            },
            dev_cards={card_type: 0 for card_type in DevCardType},
            new_dev_cards={card_type: 0 for card_type in DevCardType},
        )
        for player_id in range(1, player_count + 1)
    }
    desert_tile = next((tile.id for tile in board.tiles if tile.resource is Resource.DESERT), board.tiles[0].id)
    setup_order = make_setup_order(player_count)
    return GameState(
        board=board,
        player_count=player_count,
        players=players,
        phase=GamePhase.SETUP_SETTLEMENT,
        current_player_id=setup_order[0],
        turn_number=1,
        robber_tile_id=desert_tile,
        bank=initial_bank_counts(),
        dev_deck=create_shuffled_dev_deck(rng),
        setup_order=setup_order,
        _rng=rng,
    )


def player_visible_victory_points(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    score = 0
    score += len(player.settlements)
    score += 2 * len(player.cities)
    score += int(player.revealed_vp_cards)
    if state.longest_road_owner == player_id:
        score += 2
    if state.largest_army_owner == player_id:
        score += 2
    return int(score)


def player_total_victory_points(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    return int(player_visible_victory_points(state, player_id) + player.hidden_vp_cards())


def next_player_id(current_player_id: int, player_count: int) -> int:
    return (current_player_id % player_count) + 1


def turn_order_from(start_player_id: int, player_count: int) -> list[int]:
    order: list[int] = []
    player_id = start_player_id
    for _ in range(player_count):
        order.append(player_id)
        player_id = next_player_id(player_id, player_count)
    return order


def normalize_edge(edge: EdgeKey) -> EdgeKey:
    return tuple(sorted((int(edge[0]), int(edge[1]))))


def count_resource_total(resources: Dict[Resource, int] | Iterable[tuple[Resource, int]]) -> int:
    if isinstance(resources, dict):
        return int(sum(int(amount) for amount in resources.values()))
    return int(sum(int(amount) for _, amount in resources))
