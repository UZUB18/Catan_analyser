import unittest

from catan_analyzer.domain.board import Resource
from catan_analyzer.domain.randomizer import generate_randomized_board
from catan_analyzer.game import (
    DevCardType,
    GamePhase,
    apply_action,
    discard_resources,
    end_build_phase,
    end_turn,
    initialize_game_state,
    list_legal_actions,
    move_robber,
    place_setup_road,
    place_setup_settlement,
    play_knight,
    roll_dice,
    trade_bank,
    trade_player,
)
from catan_analyzer.game.awards import longest_road_length_for_player


class GameRulesTests(unittest.TestCase):
    def _complete_setup(self, state):
        while state.phase.value.startswith("setup"):
            actions = list_legal_actions(state)
            self.assertGreater(len(actions), 0)
            state = apply_action(state, actions[0])
        return state

    def test_second_setup_settlement_grants_starting_resources(self) -> None:
        board = generate_randomized_board(seed=501)
        state = initialize_game_state(board, player_count=4, seed=33)

        # Force first settlement + road for player 1.
        first_vertex = int(list_legal_actions(state)[0].data["vertex_id"])
        state = apply_action(state, place_setup_settlement(first_vertex))
        state = apply_action(state, place_setup_road(tuple(list_legal_actions(state)[0].data["edge"])))

        # Advance setup until player 1's second settlement placement.
        while not (state.phase.value == "setup_settlement" and state.current_player_id == 1 and len(state.players[1].settlements) == 1):
            state = apply_action(state, list_legal_actions(state)[0])

        cards_before = state.players[1].card_count()
        second_vertex = int(list_legal_actions(state)[0].data["vertex_id"])
        state = apply_action(state, place_setup_settlement(second_vertex))
        cards_after = state.players[1].card_count()
        self.assertGreater(cards_after, cards_before)

    def test_roll_seven_triggers_discard_then_robber_move(self) -> None:
        board = generate_randomized_board(seed=502)
        state = initialize_game_state(board, player_count=4, seed=10)
        state = self._complete_setup(state)

        for resource in Resource:
            if resource is Resource.DESERT:
                continue
            state.players[2].hand[resource] = 0
        state.players[2].hand[Resource.WOOD] = 8
        bank_before = state.bank[Resource.WOOD]
        state = apply_action(state, roll_dice(7))
        self.assertEqual(state.phase.value, "robber_discard")
        self.assertIn(2, state.discard_queue)
        required = state.players[2].card_count() // 2
        state = apply_action(state, discard_resources(2, {Resource.WOOD: required}))
        self.assertEqual(state.players[2].hand[Resource.WOOD], 8 - required)
        self.assertEqual(state.bank[Resource.WOOD], bank_before + required)
        self.assertEqual(state.phase.value, "robber_move")

    def test_cannot_play_newly_bought_dev_card_same_turn(self) -> None:
        board = generate_randomized_board(seed=503)
        state = initialize_game_state(board, player_count=3, seed=19)
        state = self._complete_setup(state)

        state.phase = GamePhase.TRADE
        state.turn_has_rolled = True
        state.current_player_id = 1
        player = state.players[1]
        player.hand[Resource.SHEEP] = 5
        player.hand[Resource.WHEAT] = 5
        player.hand[Resource.ORE] = 5
        state.dev_deck = [DevCardType.KNIGHT]

        end_trade = next(action for action in list_legal_actions(state) if action.kind == "end_trade_phase")
        state = apply_action(state, end_trade)
        buy_action = next(action for action in list_legal_actions(state) if action.kind == "buy_dev_card")
        state = apply_action(state, buy_action)
        state = apply_action(state, end_build_phase())
        robber_target = next(tile.id for tile in state.board.tiles if tile.id != state.robber_tile_id)

        with self.assertRaises(ValueError):
            apply_action(state, play_knight(robber_target))

        # End turn to unlock newly bought cards.
        state = apply_action(state, end_turn())
        state.current_player_id = 1
        state.phase = GamePhase.DEV_PLAY
        state.players[1].new_dev_cards[DevCardType.KNIGHT] = 0
        state = apply_action(state, play_knight(robber_target))
        self.assertEqual(state.players[1].played_knights, 1)

    def test_specific_port_sets_two_to_one_bank_trade_ratio(self) -> None:
        board = generate_randomized_board(seed=504)
        state = initialize_game_state(board, player_count=3, seed=41)
        state = self._complete_setup(state)
        state.phase = GamePhase.TRADE
        state.turn_has_rolled = True
        state.current_player_id = 1
        player = state.players[1]

        # Force exact resource/port for test robustness.
        from catan_analyzer.domain.board import PortType

        player.ports = {PortType.WOOD_2TO1}
        player.hand[Resource.WOOD] = 2
        player.hand[Resource.BRICK] = 0
        bank_before = state.bank[Resource.BRICK]

        state = apply_action(state, trade_bank(Resource.WOOD, Resource.BRICK, give_amount=2))
        self.assertEqual(state.players[1].hand[Resource.WOOD], 0)
        self.assertEqual(state.players[1].hand[Resource.BRICK], 1)
        self.assertEqual(state.bank[Resource.BRICK], bank_before - 1)

    def test_longest_road_path_respects_blocking_vertices(self) -> None:
        board = generate_randomized_board(seed=505)
        state = initialize_game_state(board, player_count=3, seed=8)

        path: list[int] | None = None
        for a, vertex in board.vertices.items():
            for b in vertex.adjacent_vertex_ids:
                for c in board.vertices[b].adjacent_vertex_ids:
                    if c == a:
                        continue
                    for d in board.vertices[c].adjacent_vertex_ids:
                        if d in {a, b}:
                            continue
                        path = [a, b, c, d]
                        break
                    if path is not None:
                        break
                if path is not None:
                    break
            if path is not None:
                break
        self.assertIsNotNone(path)
        assert path is not None
        edges = {
            tuple(sorted((path[0], path[1]))),
            tuple(sorted((path[1], path[2]))),
            tuple(sorted((path[2], path[3]))),
        }
        blocked_vertices = {path[2]}
        length = longest_road_length_for_player(
            board,
            player_roads=edges,
            blocked_vertices=blocked_vertices,
        )
        self.assertEqual(length, 2)

    def test_bank_shortage_single_player_exception_pays_partial(self) -> None:
        board = generate_randomized_board(seed=506)
        state = initialize_game_state(board, player_count=3, seed=22)
        state = self._complete_setup(state)

        target_tile = next(tile for tile in board.tiles if tile.resource is not Resource.DESERT and tile.token_number is not None)
        target_vertex = next(
            vertex_id
            for vertex_id, vertex in board.vertices.items()
            if target_tile.id in vertex.adjacent_hex_ids
        )

        for player in state.players.values():
            player.settlements.clear()
            player.cities.clear()
        state.players[1].cities.add(target_vertex)  # Owed 2 cards from this tile.

        state.current_player_id = 1
        state.phase = GamePhase.TURN_START
        state.turn_has_rolled = False
        if target_tile.id == state.robber_tile_id:
            state.robber_tile_id = next(tile.id for tile in board.tiles if tile.id != target_tile.id)
        state.bank[target_tile.resource] = 1
        before = state.players[1].hand[target_tile.resource]

        state = apply_action(state, roll_dice(target_tile.token_number))
        self.assertEqual(state.players[1].hand[target_tile.resource], before + 1)

    def test_bank_shortage_multiple_players_blocks_resource(self) -> None:
        board = generate_randomized_board(seed=507)
        state = initialize_game_state(board, player_count=3, seed=23)
        state = self._complete_setup(state)

        target_tile = next(tile for tile in board.tiles if tile.resource is not Resource.DESERT and tile.token_number is not None)
        adjacent_vertices = [
            vertex_id
            for vertex_id, vertex in board.vertices.items()
            if target_tile.id in vertex.adjacent_hex_ids
        ]
        self.assertGreaterEqual(len(adjacent_vertices), 2)

        for player in state.players.values():
            player.settlements.clear()
            player.cities.clear()
        state.players[1].settlements.add(adjacent_vertices[0])
        state.players[2].settlements.add(adjacent_vertices[1])

        state.current_player_id = 1
        state.phase = GamePhase.TURN_START
        state.turn_has_rolled = False
        if target_tile.id == state.robber_tile_id:
            state.robber_tile_id = next(tile.id for tile in board.tiles if tile.id != target_tile.id)
        state.bank[target_tile.resource] = 1
        p1_before = state.players[1].hand[target_tile.resource]
        p2_before = state.players[2].hand[target_tile.resource]

        state = apply_action(state, roll_dice(target_tile.token_number))
        self.assertEqual(state.players[1].hand[target_tile.resource], p1_before)
        self.assertEqual(state.players[2].hand[target_tile.resource], p2_before)

    def test_knight_can_be_played_before_roll(self) -> None:
        board = generate_randomized_board(seed=508)
        state = initialize_game_state(board, player_count=3, seed=24)
        state = self._complete_setup(state)
        state.current_player_id = 1
        state.phase = GamePhase.TURN_START
        state.turn_has_rolled = False
        player = state.players[1]
        player.dev_cards[DevCardType.KNIGHT] = 1
        player.new_dev_cards[DevCardType.KNIGHT] = 0

        robber_target = next(tile.id for tile in state.board.tiles if tile.id != state.robber_tile_id)
        state = apply_action(state, play_knight(robber_target))
        self.assertEqual(state.players[1].played_knights, 1)
        self.assertFalse(state.turn_has_rolled)
        self.assertEqual(state.phase, GamePhase.TURN_START)

    def test_player_trade_rejects_like_for_like(self) -> None:
        board = generate_randomized_board(seed=509)
        state = initialize_game_state(board, player_count=3, seed=25)
        state = self._complete_setup(state)
        state.current_player_id = 1
        state.phase = GamePhase.TRADE
        state.turn_has_rolled = True
        state.players[1].hand[Resource.WOOD] = 2
        state.players[2].hand[Resource.WOOD] = 2

        with self.assertRaises(ValueError):
            apply_action(
                state,
                trade_player(
                    2,
                    give={Resource.WOOD: 1},
                    receive={Resource.WOOD: 1},
                ),
            )


if __name__ == "__main__":
    unittest.main()
