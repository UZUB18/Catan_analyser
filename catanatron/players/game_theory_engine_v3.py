"""
game_theory_engine.py

GameTheoryEngine (GT) — an opponent-aware, stochastic expected-value engine.

Version: 1.3.0

Core ideas:
- **Single scoring model**: consistently score actions with one evaluator (value_fn),
  plus an explicit multi-opponent "table threat" term (max opponent eval).
- **Opening matters**: initial placements are chosen by simulation + evaluator
  (with a small amount of Catan-specific port awareness).
- **Robber is EV**: robber moves are chosen by expected objective, with *weighted*
  steal probabilities based on the victim's hand (not uniform 1/5).
- **Tactics matter**: optional within-turn expectimax search (SAB-style) for combo turns.
- **Safe + incremental**: everything is feature-flagged and falls back to legacy heuristics.
"""


from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from catanatron import Player
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.tree_search_utils import expand_spectrum, list_prunned_actions
from catanatron.players.value import DEFAULT_WEIGHTS, CONTENDER_WEIGHTS, base_fn, get_value_fn
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
    from catanatron.models.enums import (
        WOOD,
        BRICK,
        SHEEP,
        WHEAT,
        ORE,
        Action,
        ActionRecord,
    )  # type: ignore
except Exception:  # pragma: no cover
    WOOD, BRICK, SHEEP, WHEAT, ORE = "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"
    Action = None  # type: ignore
    ActionRecord = None  # type: ignore

GT_ENGINE_VERSION = "1.3.0"

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
            if cards is None:
                return 0
            if isinstance(cards, (int, float)):
                return int(cards)
            return len(cards)
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
    # Some actions wrap the value in enums/records; unwrap when possible.
    if hasattr(value, "value"):
        try:
            value = getattr(value, "value")
        except Exception:
            pass

    give = [0, 0, 0, 0, 0]
    get = [0, 0, 0, 0, 0]
    partner = None

    if value is None:
        return give, get, partner

    if isinstance(value, tuple) and len(value) == 10 and all(isinstance(x, (int, float)) for x in value):
        return [int(x) for x in value[:5]], [int(x) for x in value[5:]], None

    # Common canonical shapes in this codebase:
    # - CONFIRM_TRADE value is 11-tuple: (offer[5], receive[5], partner_color)
    # - state.current_trade (used for ACCEPT_TRADE / REJECT_TRADE) is often 11-tuple:
    #   (offer[5], receive[5], offering_turn_index)
    if (
        isinstance(value, tuple)
        and len(value) == 11
        and all(isinstance(x, (int, float)) for x in value[:10])
    ):
        partner = value[10]
        return [int(x) for x in value[:5]], [int(x) for x in value[5:10]], partner

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



def _current_color(state: Any) -> Any:
    """Best-effort current player color (robust across catanatron versions)."""
    try:
        fn = getattr(state, "current_color", None)
        if callable(fn):
            return fn()
    except Exception:
        pass
    try:
        return state.colors[state.current_player_index]
    except Exception:
        return None


def _winning_color(game: Any) -> Any:
    """Best-effort winner query (returns None if game is not over)."""
    try:
        fn = getattr(game, "winning_color", None)
        if callable(fn):
            return fn()
    except Exception:
        pass
    return None


def _get_playable_actions(game: Any) -> List[Any]:
    """Best-effort accessor for playable actions on a game copy."""
    try:
        actions = getattr(game, "playable_actions", None)
        if actions is None:
            return []
        return list(actions)
    except Exception:
        return []


def _try_execute_game(game: Any, action: Any, action_record: Any = None) -> Optional[Any]:
    """
    Safely copy+execute an action on a game object.

    We prefer Game.copy().execute(..., validate_action=False) (fast), and fall back to
    execute() without validate_action if needed.
    """
    try:
        game_copy = game.copy()
    except Exception:
        return None

    try:
        if action_record is None:
            game_copy.execute(action, validate_action=False)
        else:
            game_copy.execute(action, validate_action=False, action_record=action_record)
        return game_copy
    except Exception:
        # Some installs don't expose validate_action. Try again without it.
        try:
            if action_record is None:
                game_copy.execute(action)
            else:
                game_copy.execute(action, action_record=action_record)
            return game_copy
        except Exception:
            return None

class GameTheoryEngine(Player):
    """
    Opponent-aware utility engine with stochastic EV scoring.

    Compared to v1.0.0, the main strength upgrades are:
    - Robust initial placement detection (state.is_initial_build_phase).
    - Unified evaluation: one consistent evaluator for gains (value_fn), plus explicit
      multi-opponent threat (max enemy eval).
    - Expected-value robber selection with weighted steal probabilities.
    - Candidate selection uses quick simulated evaluation deltas (when safe).
    - Optional within-turn expectimax search for "combo turns" (SAB-style).
    """

    VERSION = GT_ENGINE_VERSION

    def __init__(self, *args: Any, **kwargs: Any):
        # Support CLI-style extra positional args as "key=value" tokens.
        color = None
        extra_tokens: List[Any] = []
        if args:
            color = args[0]
            extra_tokens = list(args[1:])
        else:
            color = kwargs.pop("color", None)

        for tok in extra_tokens:
            if isinstance(tok, str) and "=" in tok:
                k, v = tok.split("=", 1)
                k = k.strip()
                if k and k not in kwargs:
                    kwargs[k] = v.strip()

        # ---- Core config ----
        value_fn_builder_name = str(kwargs.pop("value_fn_builder_name", "base_fn"))
        if value_fn_builder_name in {"C", "contender", "contender_fn"}:
            value_fn_builder_name = "contender_fn"
        elif value_fn_builder_name in {"B", "base", "base_fn", "None", "none"}:
            value_fn_builder_name = "base_fn"
        if value_fn_builder_name not in {"base_fn", "contender_fn"}:
            value_fn_builder_name = "base_fn"
        self.value_fn_builder_name = value_fn_builder_name

        # contender_fn only uses CONTENDER_WEIGHTS when params is None/Falsey.
        default_params = None if value_fn_builder_name == "contender_fn" else DEFAULT_WEIGHTS
        self.params = kwargs.pop("params", default_params)

        # ---- Behavioral flags ----
        self.use_pruning = str(kwargs.pop("use_pruning", True)).lower() != "false"
        self.max_candidates = int(kwargs.pop("max_candidates", 14))

        # Scoring
        self.lookahead_weight = float(kwargs.pop("lookahead_weight", 1.0))
        # Kept for backwards compatibility; new logic uses threat_lambda_*.
        self.deny_weight = float(kwargs.pop("deny_weight", 0.70))

        # Upgrades (Stage 1/2)
        self.use_value_fn_gains = str(kwargs.pop("use_value_fn_gains", True)).lower() != "false"
        self.use_self_only_eval = str(kwargs.pop("use_self_only_eval", True)).lower() != "false"
        self.use_multi_opp_threat = str(kwargs.pop("use_multi_opp_threat", True)).lower() != "false"

        self.threat_lambda_base = float(kwargs.pop("threat_lambda_base", 0.35))
        self.threat_lambda_late = float(kwargs.pop("threat_lambda_late", 0.95))

        self.use_value_opening = str(kwargs.pop("use_value_opening", True)).lower() != "false"

        self.robber_use_ev = str(kwargs.pop("robber_use_ev", True)).lower() != "false"
        self.robber_weighted_steal = str(kwargs.pop("robber_weighted_steal", True)).lower() != "false"

        self.use_quick_sim_candidates = str(
            kwargs.pop("use_quick_sim_candidates", True)
        ).lower() != "false"
        self.offer_trade_cap = int(kwargs.pop("offer_trade_cap", 4))
        self.score_jitter = float(kwargs.pop("score_jitter", 0.0))
        # Guard same-turn search from out-of-turn prompts when prompt metadata is available.
        self.use_prompt_guard = str(kwargs.pop("use_prompt_guard", False)).lower() != "false"

        # Within-turn search
        self.use_turn_search = str(kwargs.pop("use_turn_search", True)).lower() != "false"
        self.turn_search_depth = int(kwargs.pop("turn_search_depth", 3))
        self.turn_search_width = int(kwargs.pop("turn_search_width", 8))
        self.turn_search_time_budget_ms = int(kwargs.pop("turn_search_time_budget_ms", 120))

        # Initialize Player
        super().__init__(color)
        self._rng = random.Random()

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(v={self.VERSION},value_fn={self.value_fn_builder_name},pruning={self.use_pruning},turn_search={self.use_turn_search})"
        )

    def _rand_jitter(self, scale: float = 1.0) -> float:
        try:
            j = float(self.score_jitter)
        except Exception:
            j = 0.0
        if j <= 0.0:
            return 0.0
        return self._rng.random() * j * max(0.0, float(scale))

    # -------------------------
    # Top-level decision logic
    # -------------------------

    def decide(self, game: Any, playable_actions: Sequence[Any]) -> Any:
        state = game.state

        # 0) Immediate tactical win check.
        winning = _can_win_this_turn(state, self.color, playable_actions)
        if winning is not None:
            return winning

        # 1) No choice.
        if len(playable_actions) == 1:
            return playable_actions[0]

        first_type = _action_type(playable_actions[0])

        # 2) Correct initial placement detection (v1.0.0's "INITIAL" string check is not reliable).
        if getattr(state, "is_initial_build_phase", False) or ("INITIAL" in first_type):
            return self._pick_initial(game, playable_actions)

        # 3) Special phases.
        if first_type == "DISCARD":
            return self._pick_best_by_simulated_utility(state, playable_actions)

        if first_type == "MOVE_ROBBER":
            return self._pick_robber_move(game, playable_actions)

        # 4) Build evaluator + per-decision caches.
        value_fn = self._build_value_fn()

        eval_cache: Dict[Tuple[Any, int], float] = {}
        opp_cache: Dict[int, float] = {}

        def eval_for(g: Any, color: Any) -> float:
            key = (color, id(g.state))
            if key in eval_cache:
                return eval_cache[key]
            try:
                if value_fn is None:
                    raise RuntimeError("no value_fn")
                v = float(value_fn(g, color))
            except Exception:
                v = float(self._utility(g.state, color))
            eval_cache[key] = v
            return v

        def opp_eval(g: Any) -> float:
            sid = id(g.state)
            if sid in opp_cache:
                return opp_cache[sid]
            enemies = [c for c in getattr(g.state, "colors", []) if c != self.color]
            if not enemies:
                opp_cache[sid] = 0.0
                return 0.0
            v = max(eval_for(g, c) for c in enemies)
            opp_cache[sid] = v
            return v

        def threat_lambda(s: Any) -> float:
            # Smoothly increase denial pressure as the game approaches the end.
            try:
                colors = list(getattr(s, "colors", []))
                max_vp = 0
                for c in colors:
                    try:
                        max_vp = max(max_vp, int(get_actual_victory_points(s, c)))
                    except Exception:
                        pass
                # Ramp from 0 at 6VP to 1 at 10VP.
                t = (max_vp - 6) / 4.0
                if t < 0.0:
                    t = 0.0
                if t > 1.0:
                    t = 1.0
                return self.threat_lambda_base + t * (self.threat_lambda_late - self.threat_lambda_base)
            except Exception:
                # Fallback: old deny_weight behavior.
                return self.deny_weight

        def objective(g: Any) -> float:
            me = eval_for(g, self.color)
            if not self.use_multi_opp_threat:
                # Old behavior: only worry about the VP leader (approx).
                leader, _leader_vp = _leader_color_and_vp(g.state, self.color)
                if leader is None:
                    return me
                return me - threat_lambda(g.state) * eval_for(g, leader)
            return me - threat_lambda(g.state) * opp_eval(g)

        leader, leader_vp = _leader_color_and_vp(state, self.color)
        threat_mode = leader_vp >= 8
        base_obj = objective(game)

        # 5) Within-turn search (optional): helps with combo turns (trade/build/dev chains).
        if self.use_turn_search and self.turn_search_depth > 1 and self._is_my_turn_to_play(state):
            searched = self._turn_search_best_action(
                game=game,
                root_actions=list(playable_actions),
                leader=leader,
                leader_vp=leader_vp,
                threat_mode=threat_mode,
                objective_fn=objective,
                base_obj=base_obj,
                deadline=time.time() + (self.turn_search_time_budget_ms / 1000.0),
            )
            if searched is not None:
                return searched

        # 6) Candidate reduction.
        candidates = self._candidate_actions(
            game=game,
            playable_actions=playable_actions,
            leader=leader,
            leader_vp=leader_vp,
            threat_mode=threat_mode,
            objective_fn=objective,
            base_obj=base_obj,
        )

        # 7) One-step expected-value scoring.
        try:
            outcomes_by_action = expand_spectrum(game, candidates)
        except Exception:
            outcomes_by_action = {}

        best_action = candidates[0]
        best_score = float("-inf")

        for action in candidates:
            at = _action_type(action)
            score = self._base_bias(at, threat_mode)

            if at in {"OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "MARITIME_TRADE", "CONFIRM_TRADE"}:
                score += self._trade_stance_score(state, action, leader, leader_vp)

            # Prefer our corrected robber expectation when possible.
            outcomes = None
            if at == "MOVE_ROBBER" and self.robber_weighted_steal:
                outcomes = self._expand_weighted_robber_spectrum(game, action)

            if outcomes is None:
                outcomes = outcomes_by_action.get(action)

            if outcomes:
                expected = 0.0
                total_prob = 0.0
                for outcome_game, prob in outcomes:
                    p = float(prob)
                    if p <= 0:
                        continue
                    total_prob += p
                    expected += p * objective(outcome_game)
                if total_prob > 1e-9:
                    expected /= total_prob
                    score += self.lookahead_weight * (expected - base_obj)
            else:
                # Deterministic fallback: try simulate once.
                game_copy = _try_execute_game(game, action)
                if game_copy is not None:
                    score += self.lookahead_weight * (objective(game_copy) - base_obj)
                else:
                    score += self._fallback_type_score(state, action, threat_mode)

            # Tie-break jitter.
            score += self._rand_jitter()

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # -------------------------
    # Evaluation / utilities
    # -------------------------

    def _build_value_fn(self) -> Optional[Callable[[Any, Any], float]]:
        """
        Build the evaluator used by GT.

        By default we build a *self-only* evaluator (enemy_production weight set to 0),
        because GT models denial explicitly as "max opponent eval" — this avoids double
        counting and prevents the "only P1 matters" artifact in value.py.
        """
        try:
            # Base params selection mirrors catanatron.players.value.get_value_fn behavior.
            if self.value_fn_builder_name == "contender_fn":
                params = self.params or CONTENDER_WEIGHTS
            else:
                params = self.params or DEFAULT_WEIGHTS

            if self.use_self_only_eval:
                p = dict(params)
                p["enemy_production"] = 0.0
                return base_fn(p)

            # Legacy: use the library builder directly.
            return get_value_fn(self.value_fn_builder_name, params)
        except Exception:
            return None

    def _is_my_turn_to_play(self, state: Any) -> bool:
        # Avoid running turn-search during out-of-turn prompts (discard, trade responses, etc.).
        try:
            if _current_color(state) != self.color:
                return False
            if self.use_prompt_guard:
                prompt = getattr(state, "current_prompt", None)
                if prompt is not None:
                    pv = str(getattr(prompt, "value", prompt))
                    if pv not in {"PLAY_TURN", "ActionPrompt.PLAY_TURN"}:
                        return False
            return int(state.current_player_index) == int(state.current_turn_index)
        except Exception:
            return False

    # -------------------------
    # Candidate selection
    # -------------------------

    def _candidate_actions(
        self,
        game: Any,
        playable_actions: Sequence[Any],
        leader: Any,
        leader_vp: int,
        threat_mode: bool,
        objective_fn: Callable[[Any], float],
        base_obj: float,
    ) -> List[Any]:
        actions = list(playable_actions)

        # Optional pruning (shared helper).
        if self.use_pruning:
            try:
                pruned = list_prunned_actions(game)
                pruned_set = set(pruned)
                tmp = [a for a in actions if a in pruned_set]
                if tmp:
                    actions = tmp
            except Exception:
                pass

        if len(actions) <= self.max_candidates:
            return actions

        # Keep a few trade offers; they explode branching, and most are junk.
        offer_trades = [a for a in actions if _action_type(a) == "OFFER_TRADE"]
        if offer_trades and self.offer_trade_cap >= 0:
            # Rank offers by the same scarcity-based heuristic used elsewhere.
            offer_trades.sort(
                key=lambda a: self._trade_stance_score(game.state, a, leader, leader_vp),
                reverse=True,
            )
            offer_trades = offer_trades[: max(0, self.offer_trade_cap)]
        else:
            offer_trades = []

        non_offers = [a for a in actions if _action_type(a) != "OFFER_TRADE"]

        # Quick simulated deltas for deterministic-ish actions.
        scored: List[Tuple[float, Any]] = []
        for action in non_offers:
            at = _action_type(action)
            s = self._base_bias(at, threat_mode)

            if at in {"ACCEPT_TRADE", "REJECT_TRADE", "MARITIME_TRADE", "CONFIRM_TRADE"}:
                s += self._trade_stance_score(game.state, action, leader, leader_vp)

            if self.use_quick_sim_candidates and at not in {"ROLL", "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER"}:
                game_copy = _try_execute_game(game, action)
                if game_copy is not None:
                    try:
                        s += 0.65 * self.lookahead_weight * (objective_fn(game_copy) - base_obj)
                    except Exception:
                        pass

            # small jitter to avoid brittle ties
            s += self._rand_jitter(0.01)
            scored.append((s, action))

        scored.sort(key=lambda t: t[0], reverse=True)

        # Compose final candidates.
        candidates: List[Any] = []
        # Always include a few best offers (if any).
        candidates.extend(offer_trades)

        for _s, a in scored:
            if a not in candidates:
                candidates.append(a)
            if len(candidates) >= self.max_candidates:
                break

        # Safety: never return empty.
        if not candidates:
            return actions[:1]
        return candidates

    # -------------------------
    # Turn search (within-turn expectimax)
    # -------------------------

    def _turn_search_best_action(
        self,
        game: Any,
        root_actions: List[Any],
        leader: Any,
        leader_vp: int,
        threat_mode: bool,
        objective_fn: Callable[[Any], float],
        base_obj: float,
        deadline: float,
    ) -> Optional[Any]:
        """
        Within-turn expectimax (SAB-style): only searches while it remains our turn.
        Stops when:
        - depth reaches 0
        - time budget exceeded
        - game is over
        - current player is no longer us
        """
        if time.time() >= deadline:
            return None

        # Root branching: take a small, good set.
        root_candidates = self._candidate_actions(
            game=game,
            playable_actions=root_actions,
            leader=leader,
            leader_vp=leader_vp,
            threat_mode=threat_mode,
            objective_fn=objective_fn,
            base_obj=base_obj,
        )[: max(1, self.turn_search_width)]

        def is_terminal(g: Any, depth_left: int) -> bool:
            if depth_left <= 0:
                return True
            if time.time() >= deadline:
                return True
            if _winning_color(g) is not None:
                return True
            return not self._is_my_turn_to_play(g.state)

        def action_outcomes(g: Any, action: Any):
            at = _action_type(action)
            # Custom robber spectrum with weighted steals.
            if at == "MOVE_ROBBER" and self.robber_weighted_steal:
                out = self._expand_weighted_robber_spectrum(g, action)
                if out is not None:
                    return out

            # Chance actions: use shared spectrum expander.
            if at in {"ROLL", "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER"}:
                try:
                    d = expand_spectrum(g, [action])
                    out = d.get(action)
                    if out:
                        return out
                except Exception:
                    pass

            # Deterministic fallback.
            g2 = _try_execute_game(g, action)
            if g2 is None:
                return []
            return [(g2, 1.0)]

        def search_value(g: Any, depth_left: int) -> float:
            if is_terminal(g, depth_left):
                try:
                    return float(objective_fn(g))
                except Exception:
                    return float("-inf")

            actions = _get_playable_actions(g)
            if not actions:
                try:
                    return float(objective_fn(g))
                except Exception:
                    return float("-inf")

            local_base = float(objective_fn(g))
            candidates = self._candidate_actions(
                game=g,
                playable_actions=actions,
                leader=leader,
                leader_vp=leader_vp,
                threat_mode=threat_mode,
                objective_fn=objective_fn,
                base_obj=local_base,
            )[: max(1, self.turn_search_width)]

            best = float("-inf")
            for a in candidates:
                outs = action_outcomes(g, a)
                if not outs:
                    continue
                ev = 0.0
                total = 0.0
                for og, p in outs:
                    pp = float(p)
                    if pp <= 0:
                        continue
                    total += pp
                    ev += pp * search_value(og, depth_left - 1)
                if total > 1e-9:
                    ev /= total
                    if ev > best:
                        best = ev
                if time.time() >= deadline:
                    break

            if best == float("-inf"):
                return float(objective_fn(g))
            return best

        best_action = None
        best_val = float("-inf")

        for a in root_candidates:
            outs = action_outcomes(game, a)
            if not outs:
                continue
            ev = 0.0
            total = 0.0
            for og, p in outs:
                pp = float(p)
                if pp <= 0:
                    continue
                total += pp
                ev += pp * search_value(og, self.turn_search_depth - 1)
            if total > 1e-9:
                ev /= total
                if ev > best_val:
                    best_val = ev
                    best_action = a
            if time.time() >= deadline:
                break

        # Only accept search result if it actually improves over the baseline by a margin.
        if best_action is not None and best_val > base_obj + 1e-6:
            return best_action
        return None

    # -------------------------
    # Legacy helpers (kept + lightly improved)
    # -------------------------

    def _evaluate_game(self, game: Any, value_fn: Optional[Callable[[Any, Any], float]]) -> float:
        try:
            if value_fn is None:
                raise RuntimeError("no value_fn")
            return float(value_fn(game, self.color))
        except Exception:
            return float(self._utility(game.state, self.color))

    def _utility(self, state: Any, color: Any) -> float:
        # Original coarse utility (kept as a robust fallback).
        try:
            vp = float(get_actual_victory_points(state, color))
        except Exception:
            vp = 0.0

        settlements = len(_try_get_player_buildings(state, color, "SETTLEMENT"))
        cities = len(_try_get_player_buildings(state, color, "CITY"))
        roads = len(_try_get_player_buildings(state, color, "ROAD"))

        devs = 0
        if get_dev_cards_in_hand is not None:
            try:
                devs = int(get_dev_cards_in_hand(state, color))
            except Exception:
                devs = 0

        hand = get_player_freqdeck(state, color)
        hand_size = int(sum(hand)) if hand is not None else 0
        robber_penalty = 2.0 if hand_size > 7 else 0.0

        return (
            100.0 * vp
            + 4.0 * settlements
            + 9.0 * cities
            + 0.25 * roads
            + 0.8 * devs
            + 0.1 * hand_size
            - robber_penalty
        )

    def _base_bias(self, action_type: str, threat_mode: bool) -> float:
        # Mild priors to stabilize search / break ties in evaluator plateaus.
        # (Keep these small; evaluator deltas should dominate.)
        if action_type == "ROLL":
            return 4.0
        if action_type == "END_TURN":
            return -0.4
        if action_type == "BUILD_CITY":
            return 2.0
        if action_type == "BUILD_SETTLEMENT":
            return 1.6
        if action_type == "BUY_DEVELOPMENT_CARD":
            return 0.9
        if action_type.startswith("PLAY_"):
            return 0.55
        if action_type == "BUILD_ROAD":
            return 0.25
        if action_type == "MOVE_ROBBER":
            return 0.35
        if action_type == "OFFER_TRADE":
            return -1.0 if not threat_mode else -2.0
        if action_type == "ACCEPT_TRADE":
            return 0.25 if not threat_mode else 0.05
        if action_type == "REJECT_TRADE":
            return 0.1
        if action_type == "MARITIME_TRADE":
            return 0.1
        if action_type == "CONFIRM_TRADE":
            return 0.2
        if action_type == "CANCEL_TRADE":
            return -0.1
        return 0.0

    def _fallback_type_score(self, state: Any, action: Any, threat_mode: bool) -> float:
        return self._base_bias(_action_type(action), threat_mode)

    def _trade_stance_score(self, state: Any, action: Any, leader: Any, leader_vp: int) -> float:
        at = _action_type(action)
        give, get, partner = _parse_trade_value(getattr(action, "value", None))
        scarcity = self._scarcity_weights(state, self.color)

        my_gain = sum(get[i] * scarcity[i] for i in range(5)) - sum(
            give[i] * scarcity[i] for i in range(5)
        )

        # Partner can be stored as a Color (CONFIRM_TRADE) or as an offering turn index
        # in some action prompts (DECIDE_TRADE uses state.current_trade).
        partner_color = partner
        if isinstance(partner_color, int):
            try:
                colors = list(getattr(state, "colors", []))
                if 0 <= int(partner_color) < len(colors):
                    partner_color = colors[int(partner_color)]
            except Exception:
                partner_color = partner

        # If partner is known and is the leader, be harsher.
        leader_color = leader
        feeding_penalty = 0.0
        if partner_color is not None and leader_color is not None and partner_color == leader_color:
            feeding_penalty = 0.55 if leader_vp >= 7 else 0.25

        if at == "OFFER_TRADE":
            # Offering a trade is generally negative-sum unless it unlocks a build.
            base = -0.3 - feeding_penalty
            return base + 0.25 * my_gain

        if at == "ACCEPT_TRADE":
            # Accept only if it helps and doesn't clearly feed the leader.
            return my_gain - feeding_penalty

        if at == "REJECT_TRADE":
            # Lightly prefer rejecting if the offer looks bad.
            return -0.1 * max(my_gain, 0.0)

        if at == "CONFIRM_TRADE":
            # Now the trade is actually happening; score it directly.
            return 0.75 * my_gain - feeding_penalty

        if at == "CANCEL_TRADE":
            return 0.0

        if at == "MARITIME_TRADE":
            return 0.45 * my_gain

        return 0.0

    def _scarcity_weights(self, state: Any, color: Any) -> List[float]:
        inflow, _variance = _estimate_player_inflow(state, color)
        weights = []
        for i, p in enumerate(inflow):
            w = 1.0 / (0.10 + p)
            # Discount sheep a bit late-game (often the least point-dense resource).
            if RESOURCES[i] == SHEEP:
                w *= 0.92
            weights.append(w)
        m = max(weights) if weights else 1.0
        return [w / m for w in weights]

    def _pick_best_by_simulated_utility(self, state: Any, actions: Sequence[Any]) -> Any:
        best_action = actions[0]
        best_score = float("-inf")
        for action in actions:
            next_state = _simulate_if_possible(state, action)
            if next_state is None:
                score = self._fallback_type_score(state, action, threat_mode=False)
            else:
                score = self._utility(next_state, self.color)
            score += self._rand_jitter()
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _pick_discard_action(self, game: Any, actions: Sequence[Any]) -> Any:
        """
        Pick a DISCARD action by scoring the resulting position with the engine's value function.

        This is usually stronger than `_pick_best_by_simulated_utility` because it uses the same
        evaluator as the rest of the engine (hand synergy, discard penalty, etc.).
        """
        if not actions:
            return None

        value_fn = self._build_value_fn()
        state = game.state

        best_action = actions[0]
        best_score = float("-inf")

        for action in actions:
            # Prefer full Game simulation (more faithful than State-only apply_action).
            g2 = _try_execute_game(game, action)
            if g2 is not None:
                if value_fn is not None:
                    try:
                        score = float(value_fn(g2, self.color))
                    except Exception:
                        score = float(self._utility(g2.state, self.color))
                else:
                    score = float(self._utility(g2.state, self.color))
            else:
                # Fallback: State-only simulation.
                next_state = _simulate_if_possible(state, action)
                if next_state is not None:
                    score = float(self._utility(next_state, self.color))
                else:
                    score = float(self._fallback_type_score(state, action, threat_mode=False))

            score += self._rand_jitter()
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # -------------------------
    # Robber move (EV scoring)
    # -------------------------

    def _expand_weighted_robber_spectrum(self, game: Any, action: Any) -> Optional[List[Tuple[Any, float]]]:
        """
        Weighted spectrum for MOVE_ROBBER:
        - Enumerates possible stolen cards.
        - Uses victim hand composition for probabilities (not uniform 1/5).
        """
        if Action is None or ActionRecord is None:
            return None

        try:
            (coordinate, robbed_color) = getattr(action, "value", (None, None))
        except Exception:
            return None

        if robbed_color is None:
            g2 = _try_execute_game(game, action)
            return [(g2, 1.0)] if g2 is not None else None

        try:
            victim_hand = get_player_freqdeck(game.state, robbed_color)
            total = int(sum(victim_hand))
        except Exception:
            total = 0
            victim_hand = None

        if not victim_hand or total <= 0:
            g2 = _try_execute_game(game, action)
            return [(g2, 1.0)] if g2 is not None else None

        results: List[Tuple[Any, float]] = []
        for i, card in enumerate(RESOURCES):
            try:
                count = int(victim_hand[i])
            except Exception:
                count = 0
            if count <= 0:
                continue

            prob = count / float(total)
            option_action = Action(action.color, action.action_type, (coordinate, robbed_color))
            action_record = ActionRecord(action=option_action, result=card)
            g2 = _try_execute_game(game, option_action, action_record=action_record)
            if g2 is None:
                continue
            results.append((g2, prob))

        if not results:
            g2 = _try_execute_game(game, action)
            return [(g2, 1.0)] if g2 is not None else None

        # Normalize (defensive).
        s = sum(p for _g, p in results)
        if s > 1e-9 and abs(s - 1.0) > 1e-6:
            results = [(g, p / s) for g, p in results]
        return results

    def _pick_robber_move(self, game: Any, actions: Sequence[Any]) -> Any:
        state = game.state
        if not self.robber_use_ev:
            # Heuristic robber: prioritize hurting the VP leader, but be tile-aware:
            # prefer coordinates that block the leader's highest expected production and
            # avoid blocking our own best tiles.
            leader, leader_vp = _leader_color_and_vp(state, self.color)
            board = _board_from_state(state)
            tile_map = None
            try:
                m = getattr(board, "map", None)
                tile_map = getattr(m, "tiles", None) if m is not None else None
            except Exception:
                tile_map = None

            leader_sett = set(_try_get_player_buildings(state, leader, "SETTLEMENT")) if leader is not None else set()
            leader_city = set(_try_get_player_buildings(state, leader, "CITY")) if leader is not None else set()
            my_sett = set(_try_get_player_buildings(state, self.color, "SETTLEMENT"))
            my_city = set(_try_get_player_buildings(state, self.color, "CITY"))

            def blocked_expected_production(color: Any, coordinate: Any) -> float:
                if not self.robber_heuristic_blocking:
                    return 0.0
                if tile_map is None or coordinate is None:
                    return 0.0
                try:
                    tile = tile_map.get(coordinate) if hasattr(tile_map, "get") else tile_map[coordinate]
                except Exception:
                    tile = None
                if tile is None:
                    return 0.0

                r = _tile_resource(tile)
                if r is None:
                    return 0.0
                n = _tile_number(tile)
                p = float(DICE_PROB.get(int(n), 0.0)) if n is not None else 0.0
                if p <= 0.0:
                    return 0.0

                nodes_obj = getattr(tile, "nodes", None)
                node_ids: List[Any] = []
                if nodes_obj is not None:
                    try:
                        node_ids = list(nodes_obj.values())  # type: ignore[arg-type]
                    except Exception:
                        try:
                            node_ids = list(nodes_obj)  # type: ignore[arg-type]
                        except Exception:
                            node_ids = []
                elif isinstance(tile, dict) and "nodes" in tile:
                    try:
                        node_ids = list(getattr(tile["nodes"], "values", lambda: tile["nodes"])())
                    except Exception:
                        node_ids = []

                if not node_ids:
                    return 0.0

                mult = 0
                if color == leader and leader is not None:
                    for nid in node_ids:
                        if nid in leader_sett:
                            mult += 1
                        elif nid in leader_city:
                            mult += 2
                elif color == self.color:
                    for nid in node_ids:
                        if nid in my_sett:
                            mult += 1
                        elif nid in my_city:
                            mult += 2
                else:
                    # Only needed for leader/self currently.
                    return 0.0

                if mult <= 0:
                    return 0.0

                # Wheat/ore block tends to be more impactful on VP conversion.
                res_weight = 1.25 if r in {WHEAT, ORE} else 1.0
                return res_weight * p * float(mult)

            best_action = actions[0]
            best_score = float("-inf")

            # Scale blocking pressure up in endgame.
            block_w = 12.0 + (4.0 if leader_vp >= 8 else 0.0)
            self_block_w = 9.0

            for action in actions:
                try:
                    (coordinate, robbed_color) = getattr(action, "value", (None, None))
                except Exception:
                    coordinate, robbed_color = None, None

                score = 0.0

                # Prefer stealing from the leader when possible.
                if leader is not None and robbed_color == leader:
                    score += 2.5 + (0.5 if leader_vp >= 8 else 0.0)

                # Prefer stealing from big hands (more likely to hit something useful / deny).
                if robbed_color is not None:
                    hand = get_player_freqdeck(state, robbed_color)
                    score += 0.15 * sum(hand) if hand is not None else 0.0

                # Coordinate-aware blocking.
                if leader is not None and coordinate is not None and self.robber_heuristic_blocking:
                    score += block_w * blocked_expected_production(leader, coordinate)
                    score -= self_block_w * blocked_expected_production(self.color, coordinate)

                score += self._rand_jitter()
                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action

        # EV scoring with weighted steal distribution.
        value_fn = self._build_value_fn()
        if value_fn is None:
            return actions[0]

        eval_cache: Dict[Tuple[Any, int], float] = {}
        opp_cache: Dict[int, float] = {}

        def eval_for(g: Any, color: Any) -> float:
            key = (color, id(g.state))
            if key in eval_cache:
                return eval_cache[key]
            try:
                v = float(value_fn(g, color))
            except Exception:
                v = float(self._utility(g.state, color))
            eval_cache[key] = v
            return v

        def opp_eval(g: Any) -> float:
            sid = id(g.state)
            if sid in opp_cache:
                return opp_cache[sid]
            enemies = [c for c in getattr(g.state, "colors", []) if c != self.color]
            v = max(eval_for(g, c) for c in enemies) if enemies else 0.0
            opp_cache[sid] = v
            return v

        def threat_lambda(s: Any) -> float:
            try:
                colors = list(getattr(s, "colors", []))
                max_vp = 0
                for c in colors:
                    try:
                        max_vp = max(max_vp, int(get_actual_victory_points(s, c)))
                    except Exception:
                        pass
                t = (max_vp - 6) / 4.0
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                return self.threat_lambda_base + t * (self.threat_lambda_late - self.threat_lambda_base)
            except Exception:
                return self.deny_weight

        def objective(g: Any) -> float:
            return eval_for(g, self.color) - threat_lambda(g.state) * opp_eval(g)

        best_action = actions[0]
        best_val = float("-inf")

        for action in actions:
            outcomes = self._expand_weighted_robber_spectrum(game, action) if self.robber_weighted_steal else None
            if outcomes is None:
                try:
                    outcomes = expand_spectrum(game, [action]).get(action)
                except Exception:
                    outcomes = None
            if not outcomes:
                g2 = _try_execute_game(game, action)
                if g2 is None:
                    continue
                val = objective(g2)
            else:
                ev = 0.0
                total = 0.0
                for og, p in outcomes:
                    pp = float(p)
                    if pp <= 0:
                        continue
                    total += pp
                    ev += pp * objective(og)
                if total <= 1e-9:
                    continue
                val = ev / total

            val += self._rand_jitter(0.1)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action

    # -------------------------
    # Initial placement (simulation + evaluator)
    # -------------------------

    def _pick_initial(self, game: Any, actions: Sequence[Any]) -> Any:
        state = game.state
        board = _board_from_state(state)

        # If we can't simulate, fall back to heuristic.
        if not self.use_value_opening:
            return self._pick_initial_heuristic(board, actions)

        value_fn = self._build_value_fn()
        if value_fn is None:
            return self._pick_initial_heuristic(board, actions)

        def eval_me(g: Any) -> float:
            try:
                return float(value_fn(g, self.color))
            except Exception:
                return float(self._utility(g.state, self.color))

        action_type = _action_type(actions[0])

        # In this codebase, initial actions are BUILD_SETTLEMENT / BUILD_ROAD while state.is_initial_build_phase is True.
        if action_type in {"BUILD_SETTLEMENT", "BUILD_INITIAL_SETTLEMENT"}:
            best_action = actions[0]
            best_score = float("-inf")

            for action in actions:
                g1 = _try_execute_game(game, action)
                if g1 is None:
                    continue

                # Look one step ahead for the forced initial road placement (captures immediate synergy).
                follow = _get_playable_actions(g1)
                follow_type = _action_type(follow[0]) if follow else ""
                if follow and follow_type in {"BUILD_ROAD", "BUILD_INITIAL_ROAD"}:
                    best_follow = float("-inf")
                    for road_action in follow:
                        g2 = _try_execute_game(g1, road_action)
                        if g2 is None:
                            continue
                        best_follow = max(best_follow, eval_me(g2))
                    score = best_follow if best_follow != float("-inf") else eval_me(g1)
                else:
                    score = eval_me(g1)

                # Small port awareness: having *any* port early is usually good.
                try:
                    ports = g1.state.board.get_player_port_resources(self.color)
                    if ports:
                        # 3:1 is typically weaker than a good 2:1, so reward resource ports more.
                        score += 0.12 * sum(1 for p in ports if p is None)
                        score += 0.22 * sum(1 for p in ports if p is not None)
                except Exception:
                    pass

                score += self._rand_jitter(0.1)
                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action

        if action_type in {"BUILD_ROAD", "BUILD_INITIAL_ROAD"}:
            best_action = actions[0]
            best_score = float("-inf")
            for action in actions:
                g1 = _try_execute_game(game, action)
                if g1 is None:
                    continue
                score = eval_me(g1)
                score += self._rand_jitter(0.1)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action

        # Fallback for unexpected prompts in initial phase.
        return self._pick_best_by_simulated_utility(state, actions)

    def _pick_initial_heuristic(self, board: Any, actions: Sequence[Any]) -> Any:
        # Legacy heuristic (kept as safety net).
        if not actions:
            raise ValueError("No initial actions")

        action_type = _action_type(actions[0])

        if action_type in {"BUILD_SETTLEMENT", "BUILD_INITIAL_SETTLEMENT"}:
            best_action = actions[0]
            best_score = float("-inf")
            for action in actions:
                node = getattr(action, "value", None)
                inflow, variance, diversity = _node_production_profile(board, node)

                # Strong heuristic bias: pips matter, but ore+wheat are point-dense.
                ore = inflow[RES_IDX.get(ORE, 4)]
                wheat = inflow[RES_IDX.get(WHEAT, 3)]
                pip_sum = sum(inflow)
                score = 100.0 * pip_sum + 12.0 * diversity - 6.0 * variance + 22.0 * (ore + wheat)

                score += self._rand_jitter(0.1)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action

        if action_type in {"BUILD_ROAD", "BUILD_INITIAL_ROAD"}:
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
                score += self._rand_jitter(0.1)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action

        return actions[0]


# ============================================================================
# GTv3 Hybrid Upgrades: TT + Iterative Deepening Turn Search + Opponent Reply
# ============================================================================
#
# Design intent (high signal):
# - Beat AB where it is *actually* strong (tactical sequencing) by going deeper
#   within our own turn at the same time budget: iterative deepening + caching +
#   ordering.
# - Beat AB where it is *structurally wrong* (multi-player dynamics) by keeping
#   GT's opponent-aware objective and adding a lightweight opponent-reply model
#   (not paranoid minimax, but "what happens immediately after I pass the turn?").
#
# Everything is opt-in via feature flags; defaults are conservative.

from collections import OrderedDict
from dataclasses import dataclass


def _safe_sort_key(x: Any) -> str:
    """Robust sort key for heterogeneous position IDs (ints, enums, tuples, objects)."""
    try:
        return str(int(x))
    except Exception:
        return str(x)


def _canonical_edge(edge: Any) -> Any:
    """Canonicalize an undirected road edge (n1, n2) -> (min,max) when possible."""
    if isinstance(edge, tuple) and len(edge) == 2:
        a, b = edge
        # Use string order as a safe total ordering.
        sa, sb = _safe_sort_key(a), _safe_sort_key(b)
        return (a, b) if sa <= sb else (b, a)
    return edge


def _action_sig(action: Any) -> str:
    """
    Stable action signature across game copies.
    Avoid relying on object identity.
    """
    at = _action_type(action)
    v = getattr(action, "value", None)
    # repr() is more stable than str() for tuples/enums.
    return f"{at}:{repr(v)}"


@dataclass
class _TTEntry:
    value: float
    depth: int
    best_action_sig: Optional[str]
    age: int


class _LRUTranspositionTable:
    """
    Lightweight, stable-key transposition cache.

    Keyed by (stable_state_key, mode), storing the deepest value found.
    Uses an LRU eviction policy to keep memory bounded.
    """

    def __init__(self, max_entries: int = 60_000):
        self.max_entries = int(max_entries)
        self._data: "OrderedDict[Tuple[Any, str], _TTEntry]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.age = 0

    def new_search(self) -> None:
        self.age += 1

    def probe(self, state_key: Any, depth: int, mode: str) -> Optional[_TTEntry]:
        k = (state_key, mode)
        e = self._data.get(k)
        if e is None:
            self.misses += 1
            return None

        # Only usable if stored search was at least as deep.
        if e.depth < depth:
            self.misses += 1
            return None

        self.hits += 1
        # LRU bump
        self._data.move_to_end(k)
        return e

    def store(self, state_key: Any, depth: int, value: float, mode: str, best_action_sig: Optional[str] = None) -> None:
        k = (state_key, mode)
        e = self._data.get(k)
        if e is not None:
            # Don't overwrite a deeper entry with a shallower one unless it's stale.
            if e.depth > depth and (self.age - e.age) <= 2:
                return
            self._data.move_to_end(k)

        self._data[k] = _TTEntry(value=float(value), depth=int(depth), best_action_sig=best_action_sig, age=self.age)
        self.stores += 1

        # Evict LRU.
        if len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "stores": self.stores,
            "age": self.age,
        }


class GameTheoryEngineV3(GameTheoryEngine):
    """
    GTv3 Hybrid (drop-in upgrade over v1.3.0):

    Added:
    - Persistent transposition table keyed by a stable state key.
    - Iterative deepening within-turn expectimax (still "same turn only").
    - Move ordering: TT-best first + lightweight history scores.
    - Optional opponent reply approximation (one opponent move) blended into
      scoring for actions that pass control to an opponent.

    Philosophy:
    - Don't become "paranoid minimax" in a 4-player stochastic game.
      But do stop making moves that look great until the *very next player*
      cashes in on the opportunity you just created.
    """

    VERSION = "3.0.0"

    def __init__(self, *args: Any, **kwargs: Any):
        # Support CLI-style "key=value" tokens, mirroring v1.3.0.
        color = None
        extra_tokens: List[Any] = []
        if args:
            color = args[0]
            extra_tokens = list(args[1:])
        else:
            color = kwargs.get("color", None)

        for tok in extra_tokens:
            if isinstance(tok, str) and "=" in tok:
                k, v = tok.split("=", 1)
                k = k.strip()
                if k and k not in kwargs:
                    kwargs[k] = v.strip()

        profile = str(kwargs.pop("profile", "balanced")).strip().lower()
        if profile in {"ab_killer", "abkiller", "anti_ab", "vs_ab"}:
            # Goal: beat the stock AB bot by leaning into what AB is best at (short-horizon cleanup),
            # while keeping GTv3's stronger special-phase heuristics (opening, robber, discard).
            #
            # Key ideas:
            # - Delegate *most* in-turn decisions to an AB core (depth=2, prunning=true) nearly every prompt.
            # - Keep robber as a leader-focused heuristic (empirically stronger vs AB fields).
            # - Use AB's own base_fn weights in the delegate so we're not fighting the baseline evaluator.
            kwargs.setdefault("turn_search_depth", 2)
            kwargs.setdefault("turn_search_width", 6)
            kwargs.setdefault("turn_search_time_budget_ms", 90)
            kwargs.setdefault("score_jitter", 0.0)
            kwargs.setdefault("use_prompt_guard", True)
            kwargs.setdefault("use_turn_search", True)

            kwargs.setdefault("robber_use_ev", False)

            kwargs.setdefault("use_ab_tactical_core", True)
            kwargs.setdefault("ab_tactical_depth", 2)
            kwargs.setdefault("ab_tactical_prunning", True)
            kwargs.setdefault("ab_tactical_value_fn", "base_fn")
            kwargs.setdefault("ab_tactical_use_none_params", False)
            kwargs.setdefault("ab_tactical_max_calls_per_turn", 100)
            kwargs.setdefault("ab_tactical_min_branching", 2)
            kwargs.setdefault("ab_tactical_windows", "branching,endgame,conversion,pass_risk")

        elif profile == "aggressive_tactical":
            kwargs.setdefault("turn_search_depth", 5)
            kwargs.setdefault("turn_search_width", 10)
            kwargs.setdefault("turn_search_time_budget_ms", 240)
            kwargs.setdefault("score_jitter", 0.0)
            kwargs.setdefault("use_prompt_guard", True)
            kwargs.setdefault("use_turn_search", True)
        elif profile == "fast":
            kwargs.setdefault("turn_search_depth", 2)
            kwargs.setdefault("turn_search_width", 6)
            kwargs.setdefault("turn_search_time_budget_ms", 90)
            kwargs.setdefault("score_jitter", 0.0)
            kwargs.setdefault("use_prompt_guard", True)
            kwargs.setdefault("use_turn_search", True)
        else:  # balanced default
            kwargs.setdefault("turn_search_depth", 4)
            kwargs.setdefault("turn_search_width", 8)
            kwargs.setdefault("turn_search_time_budget_ms", 160)
            kwargs.setdefault("score_jitter", 0.0)
            kwargs.setdefault("use_prompt_guard", True)
            kwargs.setdefault("use_turn_search", True)
        self.profile = profile

        # --- New feature flags / knobs (popped before super) ---
        self.use_tt = str(kwargs.pop("use_tt", True)).lower() != "false"
        self.tt_max_entries = int(kwargs.pop("tt_max_entries", 60_000))

        # Turn-search upgrades
        self.turn_search_iterative_deepening = str(kwargs.pop("turn_search_id", True)).lower() != "false"
        self.turn_search_min_improve = float(kwargs.pop("turn_search_min_improve", 1e-6))

        # Ordering / history
        self.use_history_ordering = str(kwargs.pop("use_history_ordering", True)).lower() != "false"
        self._history_scores: Dict[str, float] = {}

        # Opponent reply approximation (cheap 2-ply "reality check")
        self.use_opp_reply = str(kwargs.pop("use_opp_reply", True)).lower() != "false"
        self.opp_reply_weight = float(kwargs.pop("opp_reply_weight", 0.55))  # 0=no reply, 1=full reply EV
        self.opp_reply_width = int(kwargs.pop("opp_reply_width", 8))
        self.opp_reply_model = str(kwargs.pop("opp_reply_model", "selfish"))  # {"selfish","adversarial"}
        self.opp_reply_budget_ms = int(kwargs.pop("opp_reply_budget_ms", 30))

        # Safe pass-risk filter.
        self.use_safe_pass_filter = str(kwargs.pop("use_safe_pass_filter", True)).lower() != "false"
        self.safe_pass_lambda = float(kwargs.pop("safe_pass_lambda", 0.35))

        # Selective AB tactical delegate (borrows AB's strongest short-horizon cleanup).
        self.use_ab_tactical_core = str(kwargs.pop("use_ab_tactical_core", True)).lower() != "false"
        self.ab_tactical_depth = int(kwargs.pop("ab_tactical_depth", 2))
        self.ab_tactical_prunning = str(kwargs.pop("ab_tactical_prunning", True)).lower() != "false"
        self.ab_tactical_value_fn = str(kwargs.pop("ab_tactical_value_fn", "contender_fn"))
        if self.ab_tactical_value_fn in {"C", "contender", "contender_fn"}:
            self.ab_tactical_value_fn = "contender_fn"
        else:
            self.ab_tactical_value_fn = "base_fn"
        self.ab_tactical_use_none_params = str(kwargs.pop("ab_tactical_use_none_params", True)).lower() != "false"
        self.ab_tactical_max_calls_per_turn = int(kwargs.pop("ab_tactical_max_calls_per_turn", 1))
        self.ab_tactical_min_branching = int(kwargs.pop("ab_tactical_min_branching", 9))
        raw_windows = str(kwargs.pop("ab_tactical_windows", "endgame,conversion,pass_risk"))
        raw_windows = raw_windows.replace("|", ",").replace(";", ",").replace("+", ",")
        self.ab_tactical_windows = {w.strip().lower() for w in raw_windows.split(",") if w.strip()}

        # Optional "strategic" Monte Carlo rollouts (expensive; off by default)
        self.use_mc_rollouts = str(kwargs.pop("use_mc_rollouts", False)).lower() != "false"
        self.mc_time_budget_ms = int(kwargs.pop("mc_time_budget_ms", 140))
        self.mc_rollout_depth = int(kwargs.pop("mc_rollout_depth", 10))
        self.mc_prune_to = int(kwargs.pop("mc_prune_to", 8))

        # Special-phase upgrades
        self.discard_use_value_fn = str(kwargs.pop("discard_use_value_fn", True)).lower() != "false"
        self.robber_heuristic_blocking = str(kwargs.pop("robber_heuristic_blocking", True)).lower() != "false"

        # Initialize base engine.
        super().__init__(color, **kwargs)

        # New components.
        self._tt = _LRUTranspositionTable(max_entries=self.tt_max_entries) if self.use_tt else None
        self._ab_tactical_core: Optional[AlphaBetaPlayer] = None
        self._ab_turn_marker: Optional[Tuple[int, int]] = None
        self._ab_calls_this_turn = 0

        # Per-move diagnostics.
        self.last_decision_info: Dict[str, Any] = {}
        self._mode_counts: Dict[str, int] = {}
        self._reply_budget_spent_ms = 0.0
        self._cache_reset_count = 0
        self._ab_tactical_calls_total = 0
        self._ab_tactical_hits = 0

        self._last_game_id: Optional[str] = None
        self._last_board_key: Any = None
        self._board_key_for_game: Any = None

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            base.replace(f"(v={GT_ENGINE_VERSION}", f"(v={self.VERSION}")
            + f",tt={self.use_tt},id={self.turn_search_iterative_deepening},opp_reply={self.use_opp_reply})"
        )

    # -------------------------
    # Stable hashing / keys
    # -------------------------

    def _record_mode(self, mode: str) -> None:
        self._mode_counts[mode] = int(self._mode_counts.get(mode, 0) + 1)

    def _clear_search_caches(self) -> None:
        if self._tt is not None:
            self._tt._data.clear()
            self._tt.hits = 0
            self._tt.misses = 0
            self._tt.stores = 0
            self._tt.age = 0
        self._history_scores.clear()
        self._ab_turn_marker = None
        self._ab_calls_this_turn = 0

    def _extract_board_key(self, state: Any) -> Any:
        board = _board_from_state(state)
        if board is None:
            return None
        m = getattr(board, "map", None) or board

        tiles: List[Tuple[str, Optional[str], Optional[int]]] = []
        try:
            land_tiles = getattr(m, "land_tiles", None)
            if isinstance(land_tiles, dict):
                for coord, tile in land_tiles.items():
                    tiles.append((repr(coord), _tile_resource(tile), _tile_number(tile)))
            elif isinstance(getattr(m, "tiles", None), dict):
                for coord, tile in getattr(m, "tiles").items():
                    tiles.append((repr(coord), _tile_resource(tile), _tile_number(tile)))
        except Exception:
            pass
        tiles.sort(key=lambda x: x[0])

        ports: List[Tuple[str, Tuple[str, ...]]] = []
        for attr in ("port_nodes", "ports_by_id", "ports"):
            pv = getattr(m, attr, None)
            if pv is None:
                continue
            try:
                if isinstance(pv, dict):
                    items = tuple(sorted((repr(k), repr(v)) for k, v in pv.items()))
                    ports.append((attr, tuple(f"{k}:{v}" for k, v in items)))
                else:
                    ports.append((attr, tuple(sorted(repr(x) for x in pv))))
            except Exception:
                ports.append((attr, (repr(pv),)))
        ports.sort(key=lambda x: x[0])

        return (tuple(tiles), tuple(ports))

    def _prompt_key(self, state: Any, playable_actions: Optional[Sequence[Any]] = None) -> str:
        for attr in ("phase", "current_prompt", "prompt", "prompt_type", "current_action", "pending_action"):
            v = getattr(state, attr, None)
            if v is not None:
                return str(getattr(v, "value", v))
        if playable_actions:
            try:
                return _action_type(playable_actions[0])
            except Exception:
                pass
        return "NONE"

    def _maybe_reset_caches_for_new_game(self, game: Any) -> None:
        state = getattr(game, "state", None)
        if state is None:
            return
        gid = getattr(game, "id", None)
        board_key = self._extract_board_key(state)

        game_changed = False
        if gid is not None:
            if gid != self._last_game_id:
                game_changed = True
        else:
            # Fallback when game ID is unavailable.
            if self._last_board_key != board_key:
                game_changed = True

        if game_changed:
            self._clear_search_caches()
            self._cache_reset_count += 1

        self._last_game_id = gid
        self._last_board_key = board_key
        self._board_key_for_game = board_key

    def _state_key(self, state: Any, playable_actions: Optional[Sequence[Any]] = None) -> Tuple[Any, ...]:
        """
        Build a stable, hashable state key.

        The goal is "correct enough" caching that survives Game.copy() but
        doesn't require intimate knowledge of the internal State layout.

        If any part fails, we fall back to repr(state) (slow but safe).
        """
        try:
            colors = list(getattr(state, "colors", []))
            colors_key = tuple(str(c) for c in colors)

            # Turn / phase flags.
            cur_idx = int(getattr(state, "current_player_index", -1))
            turn_idx = int(getattr(state, "current_turn_index", getattr(state, "turn", -1) or -1))
            is_initial = bool(getattr(state, "is_initial_build_phase", False))
            prompt_key = self._prompt_key(state, playable_actions)
            is_discarding = bool(getattr(state, "is_discarding", False))
            is_moving_knight = bool(getattr(state, "is_moving_knight", False))
            is_road_building = bool(getattr(state, "is_road_building", False))
            free_roads = int(getattr(state, "free_roads_available", 0) or 0)
            is_resolving_trade = bool(getattr(state, "is_resolving_trade", False))
            try:
                current_trade = tuple(getattr(state, "current_trade", ()))
            except Exception:
                current_trade = ()
            try:
                acceptees = tuple(bool(x) for x in (getattr(state, "acceptees", ()) or ()))
            except Exception:
                acceptees = ()

            # Robber coordinate (best effort).
            robber = None
            try:
                b = _board_from_state(state)
                robber = getattr(b, "robber_coordinate", None) or getattr(b, "robber", None)
            except Exception:
                robber = None

            # Buildings + hands + public VPs per color.
            per_player = []
            for c in colors:
                s = tuple(sorted((_safe_sort_key(x) for x in _try_get_player_buildings(state, c, "SETTLEMENT"))))
                ci = tuple(sorted((_safe_sort_key(x) for x in _try_get_player_buildings(state, c, "CITY"))))
                r = tuple(sorted((_safe_sort_key(_canonical_edge(x)) for x in _try_get_player_buildings(state, c, "ROAD"))))

                try:
                    hand = tuple(int(x) for x in (get_player_freqdeck(state, c) or (0, 0, 0, 0, 0)))
                except Exception:
                    hand = (0, 0, 0, 0, 0)

                try:
                    vp = int(get_actual_victory_points(state, c))
                except Exception:
                    vp = 0

                devs = _try_get_dev_cards_in_hand(state, c)

                per_player.append((str(c), s, ci, r, hand, devs, vp))

            per_player_key = tuple(per_player)
            try:
                resource_bank = tuple(int(x) for x in (getattr(state, "resource_freqdeck", ()) or ()))
            except Exception:
                resource_bank = ()
            try:
                dev_deck_len = int(len(getattr(state, "development_listdeck", ()) or ()))
            except Exception:
                dev_deck_len = -1
            board_key = self._board_key_for_game if self._board_key_for_game is not None else self._extract_board_key(state)

            return (
                board_key,
                colors_key,
                cur_idx,
                turn_idx,
                is_initial,
                prompt_key,
                is_discarding,
                is_moving_knight,
                is_road_building,
                free_roads,
                is_resolving_trade,
                current_trade,
                acceptees,
                _safe_sort_key(robber) if robber is not None else None,
                resource_bank,
                dev_deck_len,
                per_player_key,
            )
        except Exception:
            return ("FALLBACK", repr(state))

    # -------------------------
    # Ordering helpers
    # -------------------------

    def _history_score(self, action: Any) -> float:
        if not self.use_history_ordering:
            return 0.0
        return float(self._history_scores.get(_action_sig(action), 0.0))

    def _bump_history(self, action: Any, amount: float = 1.0) -> None:
        if not self.use_history_ordering:
            return
        k = _action_sig(action)
        self._history_scores[k] = float(self._history_scores.get(k, 0.0) + amount)

    def _order_actions_for_search(self, actions: Sequence[Any], threat_mode: bool, tt_best_sig: Optional[str]) -> List[Any]:
        scored: List[Tuple[float, Any]] = []
        for a in actions:
            s = 0.0
            if tt_best_sig is not None and _action_sig(a) == tt_best_sig:
                s += 1e6
            s += 50.0 * self._history_score(a)
            s += 10.0 * self._base_bias(_action_type(a), threat_mode)
            # tiny jitter to avoid pathological ties
            s += self._rand_jitter(0.0001)
            scored.append((s, a))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [a for _s, a in scored]

    def _is_conversion_action_type(self, at: str) -> bool:
        return at in {
            "BUILD_CITY",
            "BUILD_SETTLEMENT",
            "PLAY_YEAR_OF_PLENTY",
            "PLAY_MONOPOLY",
            "PLAY_ROAD_BUILDING",
            "PLAY_KNIGHT_CARD",
            "BUY_DEVELOPMENT_CARD",
            "MARITIME_TRADE",
            "CONFIRM_TRADE",
        }

    def _order_root_candidates(
        self,
        candidates: Sequence[Any],
        threat_mode: bool,
        tt_best_sig: Optional[str],
    ) -> List[Any]:
        scored: List[Tuple[Tuple[int, float], Any]] = []
        for a in candidates:
            at = _action_type(a)
            if at == "END_TURN":
                tier = 3
            elif self._is_conversion_action_type(at):
                tier = 0
            elif at.startswith("PLAY_"):
                tier = 1
            else:
                tier = 2

            s = 50.0 * self._history_score(a) + 10.0 * self._base_bias(at, threat_mode)
            if tt_best_sig is not None and _action_sig(a) == tt_best_sig:
                s += 1e5
            s += self._rand_jitter(0.0001)
            scored.append(((tier, -s), a))
        scored.sort(key=lambda t: t[0])
        return [a for _k, a in scored]

    def _refresh_ab_turn_counter(self, state: Any) -> None:
        marker = (int(getattr(state, "num_turns", -1)), int(getattr(state, "current_turn_index", -1)))
        if self._ab_turn_marker != marker:
            self._ab_turn_marker = marker
            self._ab_calls_this_turn = 0

    def _get_ab_tactical_core(self) -> AlphaBetaPlayer:
        desired_depth = max(1, int(self.ab_tactical_depth))
        needs_new = self._ab_tactical_core is None
        if self._ab_tactical_core is not None:
            if getattr(self._ab_tactical_core, "color", None) != self.color:
                needs_new = True
            elif int(getattr(self._ab_tactical_core, "depth", -1)) != desired_depth:
                needs_new = True
            elif bool(getattr(self._ab_tactical_core, "prunning", False)) != bool(self.ab_tactical_prunning):
                needs_new = True

        if needs_new:
            builder = "C" if self.ab_tactical_value_fn == "contender_fn" else "base_fn"
            params = None if (builder == "C" and self.ab_tactical_use_none_params) else DEFAULT_WEIGHTS
            self._ab_tactical_core = AlphaBetaPlayer(
                self.color,
                depth=desired_depth,
                prunning=self.ab_tactical_prunning,
                value_fn_builder_name=builder,
                params=params,
            )
        return self._ab_tactical_core

    def _should_use_ab_tactical_window(
        self,
        state: Any,
        playable_actions: Sequence[Any],
        leader_vp: int,
    ) -> bool:
        if not self.use_ab_tactical_core:
            return False
        self._refresh_ab_turn_counter(state)
        if self._ab_calls_this_turn >= max(0, self.ab_tactical_max_calls_per_turn):
            return False

        my_vp = 0
        try:
            my_vp = int(get_actual_victory_points(state, self.color))
        except Exception:
            my_vp = 0

        types = {_action_type(a) for a in playable_actions}

        if "endgame" in self.ab_tactical_windows and (my_vp >= 8 or leader_vp >= 8):
            return True
        if "conversion" in self.ab_tactical_windows and any(self._is_conversion_action_type(t) for t in types):
            return True
        if "branching" in self.ab_tactical_windows and len(playable_actions) >= max(2, self.ab_tactical_min_branching):
            return True
        if "pass_risk" in self.ab_tactical_windows and ("END_TURN" in types) and leader_vp >= 7:
            return True
        return False

    def _decide_with_ab_tactical(self, game: Any, playable_actions: Sequence[Any]) -> Optional[Any]:
        try:
            ab = self._get_ab_tactical_core()
            action = ab.decide(game, playable_actions)
            if action in playable_actions:
                self._ab_calls_this_turn += 1
                self._ab_tactical_calls_total += 1
                return action
            return None
        except Exception:
            return None

    # -------------------------
    # Opponent reply approximation
    # -------------------------

    def _approx_opponent_reply_value(
        self,
        game: Any,
        eval_for: Callable[[Any, Any], float],
        objective: Callable[[Any], float],
        deadline: float,
    ) -> float:
        """
        Approximate the *next* player's best immediate reply and return OUR objective after that reply.

        Two models:
        - selfish: opponent chooses actions to maximize their own eval_for(color).
        - adversarial: opponent chooses actions to minimize our objective (paranoid).
        """
        if time.time() >= deadline:
            return float(objective(game))

        opp = _current_color(game.state)
        if opp is None or opp == self.color:
            return float(objective(game))

        actions = _get_playable_actions(game)
        if not actions:
            return float(objective(game))

        # Optional prune (shared helper).
        if self.use_pruning:
            try:
                pruned = list_prunned_actions(game)
                ps = set(pruned)
                tmp = [a for a in actions if a in ps]
                if tmp:
                    actions = tmp
            except Exception:
                pass

        # Keep only a small shortlist to stay cheap.
        if len(actions) > self.opp_reply_width:
            actions = self._order_actions_for_search(actions, threat_mode=False, tt_best_sig=None)[: self.opp_reply_width]

        best_reply_score = float("-inf")
        best_reply_obj = float(objective(game))

        for a in actions:
            if time.time() >= deadline:
                break

            at = _action_type(a)

            # For chance-heavy actions, compute expected reply score.
            outcomes: Optional[List[Tuple[Any, float]]] = None
            if at == "MOVE_ROBBER" and self.robber_weighted_steal:
                outcomes = self._expand_weighted_robber_spectrum(game, a)
            if outcomes is None and at in {"ROLL", "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER"}:
                try:
                    outcomes = expand_spectrum(game, [a]).get(a)
                except Exception:
                    outcomes = None

            if outcomes:
                total = 0.0
                opp_score_ev = 0.0
                obj_ev = 0.0
                for og, p in outcomes:
                    pp = float(p)
                    if pp <= 0:
                        continue
                    total += pp
                    opp_score_ev += pp * float(eval_for(og, opp))
                    obj_ev += pp * float(objective(og))
                if total > 1e-9:
                    opp_score_ev /= total
                    obj_ev /= total
                else:
                    opp_score_ev = float(eval_for(game, opp))
                    obj_ev = float(objective(game))
            else:
                g2 = _try_execute_game(game, a)
                if g2 is None:
                    continue
                opp_score_ev = float(eval_for(g2, opp))
                obj_ev = float(objective(g2))

            if self.opp_reply_model == "adversarial":
                reply_score = -obj_ev
            else:
                # selfish default
                reply_score = opp_score_ev

            if reply_score > best_reply_score:
                best_reply_score = reply_score
                best_reply_obj = obj_ev

        return best_reply_obj

    # -------------------------
    # Strategic Monte Carlo (optional)
    # -------------------------

    def _mc_pick_action(
        self,
        game: Any,
        candidates: List[Any],
        eval_for: Callable[[Any, Any], float],
        objective: Callable[[Any], float],
        base_obj: float,
        deadline: float,
    ) -> Any:
        """
        Cheap multi-armed-bandit rollouts over candidates.
        Not full MCTS, but captures delayed consequences better than 1-step eval.

        Mechanics:
        - Maintain running mean objective per action.
        - Cycle through actions until time expires.
        - Progressively prune to top-K by mean.

        This is deliberately simple and bounded in CPU.
        """
        if time.time() >= deadline or not candidates:
            return candidates[0] if candidates else _get_playable_actions(game)[0]

        # Start with small set (candidate reduction already ran).
        cands = list(candidates)
        if len(cands) > max(2, self.mc_prune_to * 2):
            cands = cands[: max(2, self.mc_prune_to * 2)]

        stats: Dict[str, Tuple[int, float]] = {}  # sig -> (n, mean)
        best_action = cands[0]
        best_mean = float("-inf")

        def sample_outcome(g: Any, action: Any) -> Optional[Any]:
            at = _action_type(action)
            outs: Optional[List[Tuple[Any, float]]] = None
            if at == "MOVE_ROBBER" and self.robber_weighted_steal:
                outs = self._expand_weighted_robber_spectrum(g, action)
            if outs is None and at in {"ROLL", "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER"}:
                try:
                    outs = expand_spectrum(g, [action]).get(action)
                except Exception:
                    outs = None
            if outs:
                # Sample one outcome.
                r = self._rng.random()
                acc = 0.0
                total = sum(float(p) for _og, p in outs)
                if total <= 1e-9:
                    return outs[0][0]
                for og, p in outs:
                    acc += float(p) / total
                    if r <= acc:
                        return og
                return outs[-1][0]
            return _try_execute_game(g, action)

        def rollout_policy(g: Any) -> Optional[Any]:
            actions = _get_playable_actions(g)
            if not actions:
                return None
            cur = _current_color(g.state)
            # Greedy-ish for everyone, but shallow: pick best by quick simulated eval.
            if cur is None:
                return self._rng.choice(actions)

            # Reduce candidates for speed.
            cand = actions
            if len(cand) > 12:
                cand = self._order_actions_for_search(cand, threat_mode=False, tt_best_sig=None)[:12]

            best_a = cand[0]
            best_s = float("-inf")

            for a in cand:
                g2 = _try_execute_game(g, a)
                if g2 is None:
                    continue
                if cur == self.color:
                    s = float(objective(g2))
                else:
                    # Opponent greed: maximize their own eval.
                    s = float(eval_for(g2, cur))
                if s > best_s:
                    best_s = s
                    best_a = a
            return best_a

        # Bandit loop.
        while time.time() < deadline:
            # Prune to top-K by mean after some samples.
            if len(cands) > self.mc_prune_to:
                # Only prune if each has at least 1 sample.
                if all(stats.get(_action_sig(a), (0, 0.0))[0] >= 1 for a in cands):
                    cands.sort(key=lambda a: stats.get(_action_sig(a), (0, float("-inf")))[1], reverse=True)
                    cands = cands[: self.mc_prune_to]

            for a in list(cands):
                if time.time() >= deadline:
                    break

                og = sample_outcome(game, a)
                if og is None:
                    continue

                sim = og
                for _ in range(max(1, self.mc_rollout_depth)):
                    if time.time() >= deadline:
                        break
                    if _winning_color(sim) is not None:
                        break
                    ra = rollout_policy(sim)
                    if ra is None:
                        break
                    sim2 = _try_execute_game(sim, ra)
                    if sim2 is None:
                        break
                    sim = sim2

                val = float(objective(sim))
                sig = _action_sig(a)
                n, mean = stats.get(sig, (0, 0.0))
                n2 = n + 1
                mean2 = mean + (val - mean) / float(n2)
                stats[sig] = (n2, mean2)

                if mean2 > best_mean:
                    best_mean = mean2
                    best_action = a

        # If rollouts didn't beat baseline at all, fall back to the best mean anyway.
        return best_action

    # -------------------------
    # Decision override (core integration point)
    # -------------------------

    def decide(self, game: Any, playable_actions: Sequence[Any]) -> Any:
        t0 = time.time()
        self._maybe_reset_caches_for_new_game(game)
        state = game.state
        self._reply_budget_spent_ms = 0.0

        def _finish(mode: str, action: Any, **extra: Any) -> Any:
            self._record_mode(mode)
            info = {"mode": mode, "ms": int((time.time() - t0) * 1000)}
            info.update(extra)
            self.last_decision_info = info
            return action

        # 0) Immediate tactical win check.
        winning = _can_win_this_turn(state, self.color, playable_actions)
        if winning is not None:
            return _finish("win_now", winning)

        # 1) No choice.
        if len(playable_actions) == 1:
            return _finish("forced", playable_actions[0])

        first_type = _action_type(playable_actions[0])

        # 2) Initial placement.
        if getattr(state, "is_initial_build_phase", False) or ("INITIAL" in first_type):
            a = self._pick_initial(game, playable_actions)
            return _finish("opening", a)

        # 3) Special phases.
        if first_type == "DISCARD":
            if self.discard_use_value_fn:
                a = self._pick_discard_action(game, playable_actions) or playable_actions[0]
            else:
                a = self._pick_best_by_simulated_utility(state, playable_actions)
            return _finish("discard", a)

        if first_type == "MOVE_ROBBER":
            a = self._pick_robber_move(game, playable_actions)
            return _finish("robber", a)

        # 4) Build evaluator + per-decision caches.
        value_fn = self._build_value_fn()
        state_key_cache: Dict[int, Any] = {}

        eval_cache: Dict[Tuple[Any, Any], float] = {}
        opp_cache: Dict[Any, float] = {}

        def state_key_of(st: Any) -> Any:
            sid = id(st)
            if sid in state_key_cache:
                return state_key_cache[sid]
            sk = self._state_key(st)
            state_key_cache[sid] = sk
            return sk

        def eval_for(g: Any, color: Any) -> float:
            sk = state_key_of(g.state)
            key = (sk, str(color))
            if key in eval_cache:
                return eval_cache[key]
            try:
                if value_fn is None:
                    raise RuntimeError("no value_fn")
                v = float(value_fn(g, color))
            except Exception:
                v = float(self._utility(g.state, color))
            eval_cache[key] = v
            return v

        def opp_eval(g: Any) -> float:
            sk = state_key_of(g.state)
            if sk in opp_cache:
                return opp_cache[sk]
            enemies = [c for c in getattr(g.state, "colors", []) if c != self.color]
            if not enemies:
                opp_cache[sk] = 0.0
                return 0.0
            v = max(eval_for(g, c) for c in enemies)
            opp_cache[sk] = v
            return v

        def threat_lambda(s: Any) -> float:
            try:
                colors = list(getattr(s, "colors", []))
                max_vp = 0
                for c in colors:
                    try:
                        max_vp = max(max_vp, int(get_actual_victory_points(s, c)))
                    except Exception:
                        pass
                t = (max_vp - 6) / 4.0
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                return self.threat_lambda_base + t * (self.threat_lambda_late - self.threat_lambda_base)
            except Exception:
                return self.deny_weight

        def objective(g: Any) -> float:
            me = eval_for(g, self.color)
            if not self.use_multi_opp_threat:
                leader, _leader_vp = _leader_color_and_vp(g.state, self.color)
                if leader is None:
                    return me
                return me - threat_lambda(g.state) * eval_for(g, leader)
            return me - threat_lambda(g.state) * opp_eval(g)

        leader, leader_vp = _leader_color_and_vp(state, self.color)
        threat_mode = leader_vp >= 8
        base_obj = float(objective(game))
        root_state_key = self._state_key(state, playable_actions)

        # TT "age bump" for this decision.
        if self._tt is not None:
            self._tt.new_search()

        # 4.5) Selective AB tactical delegate.
        if self._is_my_turn_to_play(state) and self._should_use_ab_tactical_window(state, playable_actions, leader_vp):
            ab_action = self._decide_with_ab_tactical(game, playable_actions)
            if ab_action is not None:
                self._ab_tactical_hits += 1
                if self._tt is not None:
                    self._tt.store(root_state_key, depth=1, value=base_obj, mode="ROOT", best_action_sig=_action_sig(ab_action))
                return _finish(
                    "ab_tactical",
                    ab_action,
                    tt=(self._tt.stats() if self._tt is not None else None),
                    ab_calls_total=self._ab_tactical_calls_total,
                )

        # 5) Within-turn search (iterative deepening + TT).
        if self.use_turn_search and self.turn_search_depth > 1 and self._is_my_turn_to_play(state):
            deadline = time.time() + (self.turn_search_time_budget_ms / 1000.0)
            searched = self._turn_search_best_action(
                game=game,
                root_actions=list(playable_actions),
                leader=leader,
                leader_vp=leader_vp,
                threat_mode=threat_mode,
                objective_fn=objective,
                base_obj=base_obj,
                deadline=deadline,
            )
            if searched is not None:
                return _finish("turn_search", searched, tt=(self._tt.stats() if self._tt is not None else None))

        # 6) Candidate reduction.
        candidates = self._candidate_actions(
            game=game,
            playable_actions=playable_actions,
            leader=leader,
            leader_vp=leader_vp,
            threat_mode=threat_mode,
            objective_fn=objective,
            base_obj=base_obj,
        )

        # Optional strategic MC rollouts when branching is ugly.
        if self.use_mc_rollouts and len(candidates) >= 8:
            mc_deadline = time.time() + (self.mc_time_budget_ms / 1000.0)
            a = self._mc_pick_action(
                game=game,
                candidates=candidates,
                eval_for=eval_for,
                objective=objective,
                base_obj=base_obj,
                deadline=mc_deadline,
            )
            return _finish(
                "mc_rollouts",
                a,
                candidates=len(candidates),
                tt=(self._tt.stats() if self._tt is not None else None),
            )

        # 7) One-step expected-value scoring (+ optional opponent reply).
        try:
            outcomes_by_action = expand_spectrum(game, candidates)
        except Exception:
            outcomes_by_action = {}

        best_action = candidates[0]
        best_score = float("-inf")

        # Ordering hint: TT best move first at the root, if available.
        tt_best_sig: Optional[str] = None
        if self._tt is not None:
            tt_entry = self._tt.probe(root_state_key, depth=1, mode="ROOT")
            if tt_entry is not None and tt_entry.best_action_sig:
                tt_best_sig = tt_entry.best_action_sig
        candidates = self._order_root_candidates(candidates, threat_mode=threat_mode, tt_best_sig=tt_best_sig)

        reply_total_budget = max(0.0, float(self.opp_reply_budget_ms) / 1000.0)
        per_action_reply_budget = (reply_total_budget / max(1, len(candidates))) if self.use_opp_reply else 0.0

        for action in candidates:
            at = _action_type(action)
            score = self._base_bias(at, threat_mode)

            if at in {"OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "MARITIME_TRADE", "CONFIRM_TRADE"}:
                score += self._trade_stance_score(state, action, leader, leader_vp)

            # Prefer corrected robber expectation when possible.
            outcomes = None
            if at == "MOVE_ROBBER" and self.robber_weighted_steal:
                outcomes = self._expand_weighted_robber_spectrum(game, action)
            if outcomes is None:
                outcomes = outcomes_by_action.get(action)

            if outcomes:
                expected_no_reply = 0.0
                total_prob = 0.0
                pass_prob = 0.0
                for outcome_game, prob in outcomes:
                    p = float(prob)
                    if p <= 0:
                        continue
                    total_prob += p
                    expected_no_reply += p * float(objective(outcome_game))
                    if not self._is_my_turn_to_play(outcome_game.state):
                        pass_prob += p
                expected_no_reply = (expected_no_reply / total_prob) if total_prob > 1e-9 else float(base_obj)
                expected = expected_no_reply

                # Optional 2-ply "reply" adjustment: only if the turn passed away.
                if self.use_opp_reply and self.opp_reply_weight > 1e-6 and pass_prob > 1e-9:
                    reply_ev = 0.0
                    rtot = 0.0
                    action_reply_deadline = time.time() + per_action_reply_budget
                    t_reply_start = time.time()
                    for outcome_game, prob in outcomes:
                        p = float(prob)
                        if p <= 0:
                            continue
                        rtot += p
                        if self._is_my_turn_to_play(outcome_game.state):
                            reply_ev += p * float(objective(outcome_game))
                        else:
                            reply_ev += p * float(
                                self._approx_opponent_reply_value(
                                    outcome_game,
                                    eval_for,
                                    objective,
                                    action_reply_deadline,
                                )
                            )
                        if time.time() >= action_reply_deadline:
                            break
                    self._reply_budget_spent_ms += max(0.0, (time.time() - t_reply_start) * 1000.0)
                    reply_ev = (reply_ev / rtot) if rtot > 1e-9 else expected
                    if self.use_safe_pass_filter:
                        danger = max(0.0, expected_no_reply - reply_ev)
                        expected_no_reply -= self.safe_pass_lambda * danger * min(1.0, pass_prob)
                    expected = (1.0 - self.opp_reply_weight) * expected + self.opp_reply_weight * reply_ev
                    expected = min(expected, expected_no_reply) if self.use_safe_pass_filter else expected

                score += self.lookahead_weight * (expected - base_obj)

            else:
                # Deterministic fallback.
                game_copy = _try_execute_game(game, action)
                if game_copy is not None:
                    expected_no_reply = float(objective(game_copy))
                    expected = expected_no_reply

                    if self.use_opp_reply and self.opp_reply_weight > 1e-6 and (not self._is_my_turn_to_play(game_copy.state)):
                        t_reply_start = time.time()
                        reply_deadline = time.time() + per_action_reply_budget
                        reply_obj = float(self._approx_opponent_reply_value(game_copy, eval_for, objective, reply_deadline))
                        self._reply_budget_spent_ms += max(0.0, (time.time() - t_reply_start) * 1000.0)
                        if self.use_safe_pass_filter:
                            danger = max(0.0, expected_no_reply - reply_obj)
                            expected_no_reply -= self.safe_pass_lambda * danger
                        expected = (1.0 - self.opp_reply_weight) * expected + self.opp_reply_weight * reply_obj
                        expected = min(expected, expected_no_reply) if self.use_safe_pass_filter else expected

                    score += self.lookahead_weight * (expected - base_obj)
                else:
                    score += self._fallback_type_score(state, action, threat_mode)

            # Tie-break jitter.
            score += self._rand_jitter(0.5)

            if score > best_score:
                best_score = score
                best_action = action

        # Record TT and history at the root (for next time).
        if self._tt is not None:
            self._tt.store(root_state_key, depth=1, value=float(best_score), mode="ROOT", best_action_sig=_action_sig(best_action))
        self._bump_history(best_action, amount=0.25)

        return _finish(
            "one_step_ev",
            best_action,
            candidates=len(candidates),
            tt=(self._tt.stats() if self._tt is not None else None),
            reply_budget_spent_ms=round(self._reply_budget_spent_ms, 3),
            ab_calls_total=self._ab_tactical_calls_total,
        )

    # -------------------------
    # Turn search override (iterative deepening + TT + ordering)
    # -------------------------

    def _turn_search_best_action(
        self,
        game: Any,
        root_actions: List[Any],
        leader: Any,
        leader_vp: int,
        threat_mode: bool,
        objective_fn: Callable[[Any], float],
        base_obj: float,
        deadline: float,
    ) -> Optional[Any]:
        if not self.turn_search_iterative_deepening or self._tt is None:
            # Fall back to v1.3.0 implementation (fixed depth, no persistent TT).
            return super()._turn_search_best_action(
                game=game,
                root_actions=root_actions,
                leader=leader,
                leader_vp=leader_vp,
                threat_mode=threat_mode,
                objective_fn=objective_fn,
                base_obj=base_obj,
                deadline=deadline,
            )

        if time.time() >= deadline:
            return None

        # Root branching: small, good set.
        root_candidates = self._candidate_actions(
            game=game,
            playable_actions=root_actions,
            leader=leader,
            leader_vp=leader_vp,
            threat_mode=threat_mode,
            objective_fn=objective_fn,
            base_obj=base_obj,
        )[: max(1, self.turn_search_width)]

        # Order root candidates using TT + history.
        root_key = self._state_key(game.state, root_actions)
        root_tt = self._tt.probe(root_key, depth=1, mode="TURN")
        root_best_sig = root_tt.best_action_sig if root_tt is not None else None
        root_candidates = self._order_actions_for_search(root_candidates, threat_mode=threat_mode, tt_best_sig=root_best_sig)

        def is_terminal(g: Any, depth_left: int) -> bool:
            if depth_left <= 0:
                return True
            if time.time() >= deadline:
                return True
            if _winning_color(g) is not None:
                return True
            return not self._is_my_turn_to_play(g.state)

        def action_outcomes(g: Any, action: Any):
            at = _action_type(action)

            # Custom robber spectrum with weighted steals.
            if at == "MOVE_ROBBER" and self.robber_weighted_steal:
                out = self._expand_weighted_robber_spectrum(g, action)
                if out is not None:
                    return out

            # Chance actions: shared spectrum expander.
            if at in {"ROLL", "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER"}:
                try:
                    d = expand_spectrum(g, [action])
                    out = d.get(action)
                    if out:
                        return out
                except Exception:
                    pass

            # Deterministic fallback.
            g2 = _try_execute_game(g, action)
            if g2 is None:
                return []
            return [(g2, 1.0)]

        def search_value(g: Any, depth_left: int) -> float:
            if is_terminal(g, depth_left):
                try:
                    return float(objective_fn(g))
                except Exception:
                    return float("-inf")

            sk = self._state_key(g.state)
            hit = self._tt.probe(sk, depth_left, mode="TURN")
            if hit is not None:
                return float(hit.value)

            actions = _get_playable_actions(g)
            if not actions:
                return float(objective_fn(g))

            local_base = float(objective_fn(g))
            candidates = self._candidate_actions(
                game=g,
                playable_actions=actions,
                leader=leader,
                leader_vp=leader_vp,
                threat_mode=threat_mode,
                objective_fn=objective_fn,
                base_obj=local_base,
            )[: max(1, self.turn_search_width)]

            # TT best for ordering.
            tt_entry = self._tt.probe(sk, depth_left, mode="TURN")
            tt_best_sig = tt_entry.best_action_sig if tt_entry is not None else None
            candidates = self._order_actions_for_search(candidates, threat_mode=threat_mode, tt_best_sig=tt_best_sig)

            best = float("-inf")
            best_sig: Optional[str] = None

            for a in candidates:
                if time.time() >= deadline:
                    break
                outs = action_outcomes(g, a)
                if not outs:
                    continue
                ev = 0.0
                total = 0.0
                for og, p in outs:
                    pp = float(p)
                    if pp <= 0:
                        continue
                    total += pp
                    ev += pp * search_value(og, depth_left - 1)
                if total > 1e-9:
                    ev /= total
                    if ev > best:
                        best = ev
                        best_sig = _action_sig(a)

            if best == float("-inf"):
                best = float(objective_fn(g))

            # Store (depth_left means: remaining depth).
            self._tt.store(sk, depth_left, best, mode="TURN", best_action_sig=best_sig)
            return best

        best_action: Optional[Any] = None
        best_val = float("-inf")
        depth_reached = 0

        # Iterative deepening from 1..turn_search_depth.
        for depth in range(1, max(1, self.turn_search_depth) + 1):
            if time.time() >= deadline:
                break

            depth_reached = depth

            local_best_action: Optional[Any] = None
            local_best_val = float("-inf")

            for a in root_candidates:
                if time.time() >= deadline:
                    break
                outs = action_outcomes(game, a)
                if not outs:
                    continue
                ev = 0.0
                total = 0.0
                for og, p in outs:
                    pp = float(p)
                    if pp <= 0:
                        continue
                    total += pp
                    ev += pp * search_value(og, depth - 1)
                if total > 1e-9:
                    ev /= total
                    if ev > local_best_val:
                        local_best_val = ev
                        local_best_action = a

            # Only accept the iteration if it actually completed.
            if time.time() < deadline and local_best_action is not None:
                best_action = local_best_action
                best_val = local_best_val
                # Encourage this move in future ordering.
                self._bump_history(best_action, amount=0.15 * depth)

            # Shallow re-ordering: put current PV move first.
            if best_action is not None:
                sig = _action_sig(best_action)
                root_candidates = sorted(root_candidates, key=lambda a: 0 if _action_sig(a) == sig else 1)

        # Only accept if it improves over baseline by a margin.
        if best_action is not None and best_val > float(base_obj) + float(self.turn_search_min_improve):
            # Store root TT.
            self._tt.store(root_key, depth=depth_reached, value=best_val, mode="TURN", best_action_sig=_action_sig(best_action))
            return best_action

        return None

    # -------------------------
    # Diagnostics
    # -------------------------

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "profile": self.profile,
            "tt": (self._tt.stats() if self._tt is not None else None),
            "history_actions": (len(self._history_scores) if self.use_history_ordering else 0),
            "cache_resets": self._cache_reset_count,
            "reply_budget_spent_ms": round(self._reply_budget_spent_ms, 3),
            "modes": dict(self._mode_counts),
            "ab_tactical": {
                "enabled": self.use_ab_tactical_core,
                "calls_total": self._ab_tactical_calls_total,
                "hits": self._ab_tactical_hits,
                "calls_this_turn": self._ab_calls_this_turn,
                "depth": self.ab_tactical_depth,
            },
            "last_decision": self.last_decision_info,
        }
