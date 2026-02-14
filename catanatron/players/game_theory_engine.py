"""
game_theory_engine.py

Standalone GameTheoryEngine (GT) extracted from three_engines.py.

Version: 1.0.0
Highlights:
- Uses catanatron.players.value.get_value_fn for shared evaluation.
- Uses catanatron.players.tree_search_utils for pruning and stochastic outcome expansion.
- Keeps GT-style opponent denial and trade skepticism.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from catanatron import Player
from catanatron.players.tree_search_utils import expand_spectrum, list_prunned_actions
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn
from catanatron.state_functions import (
    get_actual_victory_points,
    get_enemy_colors,
    get_player_freqdeck,
)

# Optional state helpers (exist in many Catanatron installs, guarded defensively).
try:
    from catanatron.state_functions import get_dev_cards_in_hand  # type: ignore
except Exception:  # pragma: no cover
    get_dev_cards_in_hand = None  # type: ignore

try:
    from catanatron.state_functions import get_player_buildings  # type: ignore
except Exception:  # pragma: no cover
    get_player_buildings = None  # type: ignore

# Optional simulation utilities.
try:
    from catanatron.apply_action import apply_action  # type: ignore
    from catanatron.state import State  # type: ignore
except Exception:  # pragma: no cover
    apply_action = None  # type: ignore
    State = None  # type: ignore

# Resource constants: prefer enums if present, otherwise strings.
try:
    from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE  # type: ignore
except Exception:  # pragma: no cover
    WOOD, BRICK, SHEEP, WHEAT, ORE = "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"

GT_ENGINE_VERSION = "1.0.0"

RESOURCES: List[str] = [WOOD, BRICK, SHEEP, WHEAT, ORE]
RES_IDX: Dict[str, int] = {r: i for i, r in enumerate(RESOURCES)}

DICE_PROB: Dict[int, float] = {
    2: 1 / 36,
    3: 2 / 36,
    4: 3 / 36,
    5: 4 / 36,
    6: 5 / 36,
    7: 6 / 36,
    8: 5 / 36,
    9: 4 / 36,
    10: 3 / 36,
    11: 2 / 36,
    12: 1 / 36,
}


def _safe_enum_value(x: Any) -> Any:
    return getattr(x, "value", x)


def _as_resource_name(x: Any) -> str:
    v = _safe_enum_value(x)
    if isinstance(v, str):
        if "." in v and v.split(".")[-1] in {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"}:
            return v.split(".")[-1]
        return v
    return str(v)


def _freqdeck_total(fd: Sequence[int]) -> int:
    return int(sum(fd))


def _try_get_player_buildings(state: Any, color: Any, kind: str) -> List[Any]:
    if get_player_buildings is not None:
        try:
            return list(get_player_buildings(state, color, kind))  # type: ignore
        except Exception:
            pass

    for attr in ["player_buildings", "buildings", "structures"]:
        obj = getattr(state, attr, None)
        if obj is None:
            continue
        try:
            if isinstance(obj, dict):
                sub = obj.get(color) or obj.get(str(color))
                if isinstance(sub, dict) and kind in sub:
                    return list(sub[kind])
        except Exception:
            pass
    return []


def _try_get_dev_cards_in_hand(state: Any, color: Any) -> int:
    if get_dev_cards_in_hand is not None:
        try:
            cards = get_dev_cards_in_hand(state, color)  # type: ignore
            return len(cards) if cards is not None else 0
        except Exception:
            return 0
    return 0


def _board_from_state(state: Any) -> Any:
    return getattr(state, "board", None) or getattr(state, "map", None) or getattr(state, "game_map", None)


def _tile_resource(tile: Any) -> Optional[str]:
    for attr in ["resource", "resource_type", "terrain", "type"]:
        if hasattr(tile, attr):
            r = _as_resource_name(getattr(tile, attr))
            if r in RES_IDX:
                return r

    if isinstance(tile, dict):
        for k in ["resource", "resource_type", "terrain", "type"]:
            if k in tile:
                r = _as_resource_name(tile[k])
                if r in RES_IDX:
                    return r
    return None


def _tile_number(tile: Any) -> Optional[int]:
    for attr in ["number", "dice", "token", "value", "roll"]:
        if hasattr(tile, attr):
            try:
                n = int(getattr(tile, attr))
                if 2 <= n <= 12:
                    return n
            except Exception:
                pass

    if isinstance(tile, dict):
        for k in ["number", "dice", "token", "value", "roll"]:
            if k in tile:
                try:
                    n = int(tile[k])
                    if 2 <= n <= 12:
                        return n
                except Exception:
                    pass
    return None


def _tiles_adjacent_to_node(board: Any, node_id: Any) -> List[Any]:
    if board is None:
        return []

    for name in [
        "get_tiles_adjacent_to_node",
        "get_adjacent_tiles_to_node",
        "get_node_tiles",
        "node_tiles",
        "tiles_for_node",
        "get_tiles_for_node",
    ]:
        fn = getattr(board, name, None)
        if callable(fn):
            try:
                out = fn(node_id)
                if out is None:
                    continue
                return list(out)
            except Exception:
                continue

    for attr in ["node_to_tiles", "node_tiles", "tiles_by_node", "adjacent_tiles_by_node"]:
        m = getattr(board, attr, None)
        if isinstance(m, dict) and node_id in m:
            try:
                return list(m[node_id])
            except Exception:
                return []

    return []


def _node_production_profile(board: Any, node_id: Any) -> Tuple[List[float], float, int]:
    inflow = [0.0] * 5
    variance = 0.0
    seen = set()

    tiles = _tiles_adjacent_to_node(board, node_id)
    for tile in tiles:
        r = _tile_resource(tile)
        n = _tile_number(tile)
        if r is None or n is None or n == 7:
            continue
        p = DICE_PROB.get(n, 0.0)
        idx = RES_IDX.get(r)
        if idx is None:
            continue
        inflow[idx] += p
        variance += p * (1.0 - p)
        seen.add(r)

    return inflow, variance, len(seen)


def _estimate_player_inflow(state: Any, color: Any) -> Tuple[List[float], float]:
    board = _board_from_state(state)
    settlements = _try_get_player_buildings(state, color, "SETTLEMENT")
    cities = _try_get_player_buildings(state, color, "CITY")

    inflow = [0.0] * 5
    variance = 0.0

    for node in settlements:
        node_in, node_var, _ = _node_production_profile(board, node)
        inflow = [inflow[i] + node_in[i] for i in range(5)]
        variance += node_var

    for node in cities:
        node_in, node_var, _ = _node_production_profile(board, node)
        inflow = [inflow[i] + 2.0 * node_in[i] for i in range(5)]
        variance += 4.0 * node_var

    return inflow, variance


def _simulate_if_possible(state: Any, action: Any) -> Optional[Any]:
    if apply_action is None or State is None:
        return None
    try:
        new_state = State.copy(state)  # type: ignore
        apply_action(new_state, action)  # type: ignore
        return new_state
    except Exception:
        return None


def _leader_color_and_vp(state: Any, me: Any) -> Tuple[Optional[Any], int]:
    try:
        enemies = list(get_enemy_colors(state, me))
    except Exception:
        enemies = list(get_enemy_colors(getattr(state, "colors", ()), me))

    best_c: Optional[Any] = None
    best_vp = -1
    for c in enemies:
        try:
            vp = int(get_actual_victory_points(state, c))
        except Exception:
            vp = 0
        if vp > best_vp:
            best_vp = vp
            best_c = c
    return best_c, best_vp


def _can_win_this_turn(state: Any, me: Any, playable_actions: Sequence[Any]) -> Optional[Any]:
    if apply_action is None or State is None:
        return None

    for action in playable_actions:
        ns = _simulate_if_possible(state, action)
        if ns is None:
            continue
        try:
            if int(get_actual_victory_points(ns, me)) >= 10:
                return action
        except Exception:
            continue
    return None


def _action_type(action: Any) -> str:
    action_type = getattr(action, "action_type", None)
    raw = getattr(action_type, "value", action_type)
    return str(_safe_enum_value(raw))


def _parse_trade_value(value: Any) -> Tuple[List[int], List[int], Optional[Any]]:
    give = [0, 0, 0, 0, 0]
    get = [0, 0, 0, 0, 0]
    partner = None

    if value is None:
        return give, get, partner

    if isinstance(value, tuple) and len(value) == 10 and all(isinstance(x, (int, float)) for x in value):
        return [int(x) for x in value[:5]], [int(x) for x in value[5:]], None

    if isinstance(value, tuple) and len(value) >= 11:
        for start in range(len(value) - 10):
            b1 = value[start : start + 5]
            b2 = value[start + 5 : start + 10]
            if all(isinstance(x, (int, float)) for x in b1) and all(isinstance(x, (int, float)) for x in b2):
                partner = value[0] if start != 0 else None
                return [int(x) for x in b1], [int(x) for x in b2], partner

    if isinstance(value, tuple) and len(value) == 5 and all(value[i] is not None for i in range(5)):
        given = value[:4]
        recv = value[4]
        for r in given:
            rn = _as_resource_name(r)
            if rn in RES_IDX:
                give[RES_IDX[rn]] += 1
        rn = _as_resource_name(recv)
        if rn in RES_IDX:
            get[RES_IDX[rn]] += 1
        return give, get, None

    return give, get, partner


class GameTheoryEngine(Player):
    """
    Opponent-aware utility engine with AB support modules.

    Upgrades in this standalone version:
    - Shared value function scoring via players/value.py.
    - Optional action pruning via players/tree_search_utils.py.
    - Stochastic expected-value scoring via expand_spectrum().
    """

    VERSION = GT_ENGINE_VERSION

    def __init__(self, *args: Any, **kwargs: Any):
        value_fn_builder_name = kwargs.pop("value_fn_builder_name", "base_fn")
        if value_fn_builder_name == "C":
            value_fn_builder_name = "contender_fn"
        if value_fn_builder_name not in {"base_fn", "contender_fn"}:
            value_fn_builder_name = "base_fn"

        self.value_fn_builder_name = value_fn_builder_name
        self.params = kwargs.pop("params", DEFAULT_WEIGHTS)
        self.use_pruning = str(kwargs.pop("use_pruning", True)).lower() != "false"
        self.max_candidates = int(kwargs.pop("max_candidates", 14))
        self.lookahead_weight = float(kwargs.pop("lookahead_weight", 1.0))
        self.deny_weight = float(kwargs.pop("deny_weight", 0.70))

        super().__init__(*args, **kwargs)
        self._rng = random.Random()

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(v={self.VERSION},value_fn={self.value_fn_builder_name},pruning={self.use_pruning})"
        )

    def decide(self, game: Any, playable_actions: Sequence[Any]) -> Any:
        state = game.state

        winning = _can_win_this_turn(state, self.color, playable_actions)
        if winning is not None:
            return winning

        if len(playable_actions) == 1:
            return playable_actions[0]

        first_type = _action_type(playable_actions[0])
        if "INITIAL" in first_type:
            return self._pick_initial(game, playable_actions)
        if first_type == "DISCARD":
            return self._pick_best_by_simulated_utility(state, playable_actions)
        if first_type == "MOVE_ROBBER":
            return self._pick_robber_move(state, playable_actions)

        leader, leader_vp = _leader_color_and_vp(state, self.color)
        threat_mode = leader_vp >= 8

        base_me = self._utility(state, self.color)
        base_leader = self._utility(state, leader) if leader is not None else 0.0

        value_fn = self._build_value_fn()
        base_eval = self._evaluate_game(game, value_fn)

        candidates = self._candidate_actions(game, playable_actions, leader, leader_vp, threat_mode)

        try:
            outcomes_by_action = expand_spectrum(game, candidates)
        except Exception:
            outcomes_by_action = {}

        best_action = candidates[0]
        best_score = float("-inf")

        for action in candidates:
            at = _action_type(action)
            score = self._base_bias(at, threat_mode)

            if at in {"OFFER_TRADE", "ACCEPT_TRADE", "MARITIME_TRADE"}:
                score += self._trade_stance_score(state, action, leader, leader_vp)

            used_spectrum = False
            outcomes = outcomes_by_action.get(action)
            if outcomes:
                expected_eval = 0.0
                expected_me = 0.0
                expected_leader = 0.0
                total_prob = 0.0

                for outcome_game, prob in outcomes:
                    p = float(prob)
                    total_prob += p
                    expected_eval += p * self._evaluate_game(outcome_game, value_fn)
                    expected_me += p * self._utility(outcome_game.state, self.color)
                    if leader is not None:
                        expected_leader += p * self._utility(outcome_game.state, leader)

                if total_prob > 1e-9:
                    if abs(total_prob - 1.0) > 1e-6:
                        expected_eval /= total_prob
                        expected_me /= total_prob
                        expected_leader /= total_prob

                    eval_gain = expected_eval - base_eval
                    my_gain = expected_me - base_me
                    leader_gain = expected_leader - base_leader
                    lam = self.deny_weight + (0.25 if threat_mode else 0.0)

                    score += self.lookahead_weight * eval_gain
                    score += my_gain - lam * leader_gain
                    if leader_gain < -1e-6:
                        score += (8.0 if threat_mode else 3.0) * (-leader_gain)
                    used_spectrum = True

            if not used_spectrum:
                next_state = _simulate_if_possible(state, action)
                if next_state is not None:
                    me_u = self._utility(next_state, self.color)
                    leader_u = self._utility(next_state, leader) if leader is not None else base_leader
                    my_gain = me_u - base_me
                    leader_gain = leader_u - base_leader
                    lam = self.deny_weight + (0.25 if threat_mode else 0.0)
                    score += my_gain - lam * leader_gain
                    if leader_gain < -1e-6:
                        score += (5.0 if threat_mode else 2.0) * (-leader_gain)
                else:
                    score += self._fallback_type_score(state, action, threat_mode)

            score += self._rng.random() * 0.01

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _build_value_fn(self):
        try:
            return get_value_fn(self.value_fn_builder_name, self.params)
        except Exception:
            return None

    def _evaluate_game(self, game: Any, value_fn: Any) -> float:
        if value_fn is not None:
            try:
                return float(value_fn(game, self.color))
            except Exception:
                pass
        return self._utility(game.state, self.color)

    def _candidate_actions(
        self,
        game: Any,
        playable_actions: Sequence[Any],
        leader: Any,
        leader_vp: int,
        threat_mode: bool,
    ) -> List[Any]:
        actions = list(playable_actions)

        if self.use_pruning:
            try:
                pruned = list_prunned_actions(game)
                if pruned:
                    filtered = [a for a in playable_actions if any(a == p for p in pruned)]
                    if filtered:
                        actions = filtered
                    else:
                        actions = list(pruned)
            except Exception:
                pass

        if len(actions) <= self.max_candidates:
            return actions

        scored: List[Tuple[float, Any]] = []
        for action in actions:
            at = _action_type(action)
            quick = self._base_bias(at, threat_mode)
            if at in {"OFFER_TRADE", "ACCEPT_TRADE", "MARITIME_TRADE"}:
                quick += self._trade_stance_score(game.state, action, leader, leader_vp)
            scored.append((quick, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [action for _, action in scored[: self.max_candidates]]

        for action in playable_actions:
            if action not in top:
                top.append(action)
                break

        return top

    def _utility(self, state: Any, color: Any) -> float:
        if color is None:
            return 0.0

        vp = float(get_actual_victory_points(state, color))
        hand = get_player_freqdeck(state, color)
        cards = float(_freqdeck_total(hand))

        settlements = len(_try_get_player_buildings(state, color, "SETTLEMENT"))
        cities = len(_try_get_player_buildings(state, color, "CITY"))
        roads = len(_try_get_player_buildings(state, color, "ROAD"))
        dev = float(_try_get_dev_cards_in_hand(state, color))

        robber_penalty = 2.5 * max(0.0, cards - 6.0) ** 2

        return (
            120.0 * vp
            + 18.0 * cities
            + 10.0 * settlements
            + 1.8 * roads
            + 3.0 * dev
            + 0.6 * cards
            - robber_penalty
        )

    def _base_bias(self, action_type: str, threat_mode: bool) -> float:
        base = {
            "ROLL": 1.0,
            "BUILD_CITY": 14.0,
            "BUILD_SETTLEMENT": 12.0,
            "BUY_DEVELOPMENT_CARD": 6.0,
            "PLAY_KNIGHT_CARD": 5.5,
            "PLAY_MONOPOLY": 6.5,
            "PLAY_YEAR_OF_PLENTY": 6.0,
            "PLAY_ROAD_BUILDING": 4.5,
            "BUILD_ROAD": 3.5,
            "MARITIME_TRADE": 1.0,
            "OFFER_TRADE": -1.0,
            "ACCEPT_TRADE": 0.5,
            "END_TURN": 0.0,
        }.get(action_type, 0.0)

        if threat_mode:
            if action_type in {"MOVE_ROBBER", "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY"}:
                base += 4.0
            if action_type == "OFFER_TRADE":
                base -= 6.0

        return base

    def _fallback_type_score(self, state: Any, action: Any, threat_mode: bool) -> float:
        at = _action_type(action)
        score = self._base_bias(at, threat_mode)

        my_cards = _freqdeck_total(get_player_freqdeck(state, self.color))
        if my_cards >= 7 and at in {"BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD", "BUILD_ROAD"}:
            score += 3.0

        if at == "MOVE_ROBBER":
            score += 5.0

        return score

    def _trade_stance_score(self, state: Any, action: Any, leader: Any, leader_vp: int) -> float:
        at = _action_type(action)
        give, get, partner = _parse_trade_value(getattr(action, "value", None))

        hand = get_player_freqdeck(state, self.color)
        inflow, _ = _estimate_player_inflow(state, self.color)
        need_w = self._scarcity_weights(hand, inflow)

        our_gain = sum((get[i] - give[i]) * need_w[i] for i in range(5))

        leader_need = [1.0, 1.0, 1.0, 1.3, 1.4]
        if leader_vp >= 8:
            leader_need = [1.0, 1.0, 1.0, 1.6, 1.8]
        leader_help = sum(give[i] * leader_need[i] for i in range(5))

        if partner is not None and leader is not None and partner != leader:
            leader_help *= 0.35

        if at == "OFFER_TRADE":
            penalty = 2.0 + (6.0 if leader_vp >= 8 else 1.5)
            return 0.8 * our_gain - penalty - 0.7 * leader_help
        if at == "ACCEPT_TRADE":
            return 1.2 * our_gain - 0.9 * leader_help
        if at == "MARITIME_TRADE":
            return 0.9 * our_gain
        return 0.0

    def _scarcity_weights(self, hand: Sequence[int], inflow: Sequence[float]) -> List[float]:
        weights = []
        for i in range(5):
            h = float(hand[i])
            r = float(inflow[i])
            weights.append(1.0 + 1.2 / (0.15 + r) + 0.35 / (1.0 + h))
        return weights

    def _pick_best_by_simulated_utility(self, state: Any, actions: Sequence[Any]) -> Any:
        base = self._utility(state, self.color)
        best_action = actions[0]
        best_score = float("-inf")
        for action in actions:
            ns = _simulate_if_possible(state, action)
            if ns is None:
                continue
            score = self._utility(ns, self.color) - base
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _pick_robber_move(self, state: Any, actions: Sequence[Any]) -> Any:
        leader, leader_vp = _leader_color_and_vp(state, self.color)
        threat_mode = leader_vp >= 8

        best_action = actions[0]
        best_score = float("-inf")

        for action in actions:
            value = getattr(action, "value", None)
            target = None
            if isinstance(value, tuple) and len(value) >= 2:
                target = value[1]

            score = 0.0
            if leader is not None and target == leader:
                score += 50.0 if threat_mode else 20.0

            if target is not None:
                try:
                    score += 4.0 * float(get_actual_victory_points(state, target))
                except Exception:
                    pass
                try:
                    score += 0.8 * _freqdeck_total(get_player_freqdeck(state, target))
                except Exception:
                    pass

            if target == self.color:
                score -= 100.0

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _pick_initial(self, game: Any, actions: Sequence[Any]) -> Any:
        state = game.state
        action_type = _action_type(actions[0])
        board = _board_from_state(state)

        if action_type == "BUILD_INITIAL_SETTLEMENT":
            best_action = actions[0]
            best_score = float("-inf")
            for action in actions:
                node = getattr(action, "value", None)
                inflow, variance, diversity = _node_production_profile(board, node)
                score = 100.0 * sum(inflow) + 8.0 * diversity - 5.0 * variance
                score += self._rng.random() * 0.001
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action

        if action_type == "BUILD_INITIAL_ROAD":
            best_action = actions[0]
            best_score = float("-inf")
            for action in actions:
                edge = getattr(action, "value", None)
                score = 0.0
                if isinstance(edge, tuple) and len(edge) == 2:
                    n1, n2 = edge
                    p1 = _node_production_profile(board, n1)[0]
                    p2 = _node_production_profile(board, n2)[0]
                    score = 50.0 * max(sum(p1), sum(p2))
                score += self._rng.random() * 0.001
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action

        return self._pick_best_by_simulated_utility(state, actions)
