import unittest

from catan_analyzer.domain.board import Resource
from catan_analyzer.domain.randomizer import generate_randomized_board
from catan_analyzer.game import (
    ACTION_REVEAL_VP,
    ActionOutcome,
    DevCardType,
    GameEngine,
    GamePhase,
    GreedyVisibleVpPolicy,
    action_outcome_spectrum,
    apply_action,
    initialize_game_state,
    list_legal_actions,
    roll_dice,
    steal_resource,
)


class EngineIntegrationTests(unittest.TestCase):
    def _complete_setup(self, state):
        while state.phase.value.startswith("setup"):
            actions = list_legal_actions(state)
            self.assertGreater(len(actions), 0)
            state = apply_action(state, actions[0])
        return state

    def test_roll_action_outcomes_cover_all_roll_sums(self) -> None:
        board = generate_randomized_board(seed=801)
        state = initialize_game_state(board, player_count=4, seed=91)
        state = self._complete_setup(state)
        state.phase = GamePhase.TURN_START
        state.turn_has_rolled = False

        outcomes = action_outcome_spectrum(state, roll_dice())
        self.assertEqual(len(outcomes), 11)
        self.assertTrue(all(isinstance(outcome, ActionOutcome) for outcome in outcomes))
        self.assertAlmostEqual(sum(outcome.probability for outcome in outcomes), 1.0, places=6)
        observed_rolls = sorted({outcome.state.dice_roll for outcome in outcomes})
        self.assertEqual(observed_rolls, list(range(2, 13)))

    def test_steal_outcomes_follow_victim_hand_distribution(self) -> None:
        board = generate_randomized_board(seed=802)
        state = initialize_game_state(board, player_count=3, seed=64)
        state = self._complete_setup(state)
        state.phase = GamePhase.ROBBER_STEAL
        state.current_player_id = 1
        state.pending_steal_target_ids = [2]
        for resource in Resource:
            if resource is Resource.DESERT:
                continue
            state.players[2].hand[resource] = 0
        state.players[2].hand[Resource.WOOD] = 2
        state.players[2].hand[Resource.BRICK] = 1

        outcomes = action_outcome_spectrum(state, steal_resource(2))
        self.assertEqual(len(outcomes), 2)
        probabilities = sorted(outcome.probability for outcome in outcomes)
        self.assertAlmostEqual(probabilities[0], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(probabilities[1], 2.0 / 3.0, places=6)

    def test_greedy_policy_prefers_reveal_vp_when_it_wins(self) -> None:
        board = generate_randomized_board(seed=803)
        state = initialize_game_state(board, player_count=3, seed=65)
        state = self._complete_setup(state)
        state.current_player_id = 1
        state.phase = GamePhase.DEV_PLAY
        state.turn_has_rolled = True
        state.winner_id = None

        vertex_ids = list(state.board.vertices.keys())
        player = state.players[1]
        player.settlements = set(vertex_ids[:3])  # 3 VP
        player.cities = set(vertex_ids[3:6])  # +6 VP => 9 visible
        player.dev_cards[DevCardType.VICTORY_POINT] = 1
        player.new_dev_cards[DevCardType.VICTORY_POINT] = 0
        player.revealed_vp_cards = 0

        legal = list_legal_actions(state)
        self.assertTrue(any(action.kind == ACTION_REVEAL_VP for action in legal))
        policy = GreedyVisibleVpPolicy()
        chosen = policy.decide(state, legal)
        self.assertEqual(chosen.kind, ACTION_REVEAL_VP)

    def test_game_engine_uses_policy_and_applies_action(self) -> None:
        board = generate_randomized_board(seed=804)
        state = initialize_game_state(board, player_count=3, seed=66)
        state = self._complete_setup(state)
        state.current_player_id = 1
        state.phase = GamePhase.BUILD
        state.turn_has_rolled = True
        player = state.players[1]
        # Enable a city build while keeping other action branches limited.
        for resource in Resource:
            if resource is Resource.DESERT:
                continue
            player.hand[resource] = 0
        player.hand[Resource.WHEAT] = 2
        player.hand[Resource.ORE] = 3

        engine = GameEngine(
            state,
            policies={1: GreedyVisibleVpPolicy()},
        )
        action = engine.play_tick()
        self.assertIsNotNone(action)
        self.assertEqual(engine.state.current_player_id, 1)


if __name__ == "__main__":
    unittest.main()
