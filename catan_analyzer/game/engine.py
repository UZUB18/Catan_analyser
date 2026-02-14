from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .actions import (
    ACTION_BUY_DEV_CARD,
    ACTION_ROLL_DICE,
    ACTION_STEAL_RESOURCE,
    GameAction,
    roll_dice,
    steal_resource,
)
from .rules import apply_action, list_legal_actions
from .state import GamePhase, GameState

ROLL_SUM_PROBABILITIES: dict[int, float] = {
    2: 1.0 / 36.0,
    3: 2.0 / 36.0,
    4: 3.0 / 36.0,
    5: 4.0 / 36.0,
    6: 5.0 / 36.0,
    7: 6.0 / 36.0,
    8: 5.0 / 36.0,
    9: 4.0 / 36.0,
    10: 3.0 / 36.0,
    11: 2.0 / 36.0,
    12: 1.0 / 36.0,
}


@dataclass(frozen=True)
class ActionOutcome:
    state: GameState
    probability: float


class EnginePolicy(Protocol):
    def decide(self, state: GameState, legal_actions: Sequence[GameAction]) -> GameAction:
        ...


class GameAccumulator(Protocol):
    def before(self, engine: "GameEngine") -> None:
        ...

    def step(self, engine_before_action: "GameEngine", action: GameAction) -> None:
        ...

    def after(self, engine: "GameEngine") -> None:
        ...


class FirstLegalPolicy:
    """Deterministic fallback policy used when no policy is provided."""

    def decide(self, state: GameState, legal_actions: Sequence[GameAction]) -> GameAction:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        return legal_actions[0]


class GameEngine:
    """
    Thin game-loop wrapper around GameState + legal/apply functions.

    Inspired by catanatron's Game wrapper API:
    - cached legal actions
    - per-tick policy decision
    - optional accumulator hooks
    """

    def __init__(
        self,
        state: GameState,
        *,
        policies: Mapping[int, EnginePolicy] | None = None,
        turn_limit: int = 1_000,
    ) -> None:
        self.state = state
        self.policies = dict(policies or {})
        self.turn_limit = max(1, int(turn_limit))
        self._default_policy: EnginePolicy = FirstLegalPolicy()

    def copy(self) -> "GameEngine":
        return GameEngine(
            self.state.clone(),
            policies=self.policies,
            turn_limit=self.turn_limit,
        )

    @property
    def legal_actions(self) -> list[GameAction]:
        return list_legal_actions(self.state)

    @property
    def winning_player_id(self) -> int | None:
        return self.state.winner_id

    def is_finished(self) -> bool:
        return (
            self.state.phase is GamePhase.GAME_OVER
            or self.state.winner_id is not None
            or self.state.turn_number > self.turn_limit
        )

    def execute(self, action: GameAction, *, validate_action: bool = True) -> GameState:
        if validate_action and action not in self.legal_actions:
            raise ValueError(f"{action} is not legal in phase {self.state.phase}.")
        self.state = apply_action(self.state, action)
        return self.state

    def play_tick(self, *, accumulators: Sequence[GameAccumulator] = ()) -> GameAction | None:
        if self.is_finished():
            return None
        legal_actions = self.legal_actions
        if not legal_actions:
            return None

        current_player_id = self.state.current_player_id
        policy = self.policies.get(current_player_id, self._default_policy)
        action = policy.decide(self.state, legal_actions)
        if action not in legal_actions:
            raise ValueError(
                f"Policy for player {current_player_id} returned illegal action: {action}."
            )

        if accumulators:
            before = self.copy()
            for accumulator in accumulators:
                accumulator.step(before, action)
        self.execute(action, validate_action=False)
        return action

    def play(
        self,
        *,
        accumulators: Sequence[GameAccumulator] = (),
        max_ticks: int | None = None,
    ) -> int | None:
        for accumulator in accumulators:
            accumulator.before(self)

        ticks = 0
        tick_cap = max_ticks if max_ticks is not None else self.turn_limit * 16
        while not self.is_finished() and ticks < tick_cap:
            action = self.play_tick(accumulators=accumulators)
            if action is None:
                break
            ticks += 1

        for accumulator in accumulators:
            accumulator.after(self)
        return self.winning_player_id


def action_outcome_spectrum(state: GameState, action: GameAction) -> list[ActionOutcome]:
    """
    Expand one legal action into probabilistic outcomes.

    This enables expected-value engines (MCTS/minimax) to reason over
    stochastic actions similarly to catanatron's action-spectrum expansion.
    """

    if action.kind == ACTION_ROLL_DICE and "value" not in action.data:
        return _roll_outcomes(state)
    if action.kind == ACTION_STEAL_RESOURCE and "resource" not in action.data:
        return _steal_outcomes(state, action)
    if action.kind == ACTION_BUY_DEV_CARD:
        return _buy_dev_card_outcomes(state, action)
    return [ActionOutcome(state=apply_action(state, action), probability=1.0)]


def expand_action_spectrum(
    state: GameState,
    actions: Sequence[GameAction],
) -> dict[GameAction, list[ActionOutcome]]:
    return {action: action_outcome_spectrum(state, action) for action in actions}


def _roll_outcomes(state: GameState) -> list[ActionOutcome]:
    outcomes: list[ActionOutcome] = []
    for roll_sum, probability in ROLL_SUM_PROBABILITIES.items():
        next_state = apply_action(state, roll_dice(roll_sum))
        outcomes.append(ActionOutcome(state=next_state, probability=probability))
    return outcomes


def _steal_outcomes(state: GameState, action: GameAction) -> list[ActionOutcome]:
    target_player_id = int(action.data["target_player_id"])
    target = state.players[target_player_id]
    total_cards = target.card_count()
    if total_cards <= 0:
        return [ActionOutcome(state=apply_action(state, action), probability=1.0)]

    outcomes: list[ActionOutcome] = []
    for resource, amount in target.hand.items():
        if amount <= 0:
            continue
        probability = float(amount) / float(total_cards)
        next_state = apply_action(state, steal_resource(target_player_id, resource))
        outcomes.append(ActionOutcome(state=next_state, probability=probability))
    return outcomes


def _buy_dev_card_outcomes(state: GameState, action: GameAction) -> list[ActionOutcome]:
    if not state.dev_deck:
        return [ActionOutcome(state=apply_action(state, action), probability=1.0)]

    counts = Counter(state.dev_deck)
    deck_size = float(len(state.dev_deck))
    outcomes: list[ActionOutcome] = []
    for card, count in counts.items():
        forced_state = state.clone()
        deck = list(forced_state.dev_deck)
        card_index = deck.index(card)
        deck.append(deck.pop(card_index))
        forced_state.dev_deck = deck
        next_state = apply_action(forced_state, action)
        outcomes.append(ActionOutcome(state=next_state, probability=float(count) / deck_size))
    return outcomes

