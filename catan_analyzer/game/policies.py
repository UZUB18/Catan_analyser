from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .actions import (
    ACTION_BUILD_CITY,
    ACTION_BUILD_SETTLEMENT,
    ACTION_BUY_DEV_CARD,
    ACTION_REVEAL_VP,
    GameAction,
)
from .rules import apply_action
from .state import GameState, player_total_victory_points, player_visible_victory_points


class RandomPolicy:
    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def decide(self, state: GameState, legal_actions: Sequence[GameAction]) -> GameAction:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        return self._rng.choice(list(legal_actions))


@dataclass
class WeightedRandomPolicy:
    """
    Catanatron-inspired weighted random policy:
    prioritize high-impact macro actions while keeping exploration.
    """

    weights_by_kind: Mapping[str, int] = field(
        default_factory=lambda: {
            ACTION_BUILD_CITY: 10_000,
            ACTION_BUILD_SETTLEMENT: 1_000,
            ACTION_BUY_DEV_CARD: 100,
        }
    )
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def decide(self, state: GameState, legal_actions: Sequence[GameAction]) -> GameAction:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        weights = [
            max(1, int(self.weights_by_kind.get(action.kind, 1)))
            for action in legal_actions
        ]
        return self._rng.choices(list(legal_actions), weights=weights, k=1)[0]


@dataclass
class GreedyVisibleVpPolicy:
    """
    One-ply greedy policy:
    choose the action that maximizes immediate visible VP (then total VP).
    """

    prioritize_reveal_vp: bool = True

    def decide(self, state: GameState, legal_actions: Sequence[GameAction]) -> GameAction:
        if not legal_actions:
            raise ValueError("No legal actions available.")

        player_id = state.current_player_id
        best_action = legal_actions[0]
        best_tuple = (-1_000_000.0, -1_000_000.0, -1_000_000.0)

        for action in legal_actions:
            next_state = apply_action(state, action)
            visible_vp = float(player_visible_victory_points(next_state, player_id))
            total_vp = float(player_total_victory_points(next_state, player_id))
            reveal_bonus = (
                1.0
                if self.prioritize_reveal_vp and action.kind == ACTION_REVEAL_VP
                else 0.0
            )
            candidate = (visible_vp, total_vp, reveal_bonus)
            if candidate > best_tuple:
                best_tuple = candidate
                best_action = action

        return best_action

