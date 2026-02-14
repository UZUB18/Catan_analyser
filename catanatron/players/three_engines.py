"""
three_engines.py

Two Catanatron engines:
- STAT : StatsEngine             (dice probabilities, expected value, risk)
- WILD : WildSheepCultEngine     (coherent eccentricity + safety floor)

Run examples:
  catanatron-play --code=three_engines.py --players=STAT,W,VP,R --num=50 --quiet
  catanatron-play --code=three_engines.py --players=WILD,WILD,STAT,W --num=100 --parallel --quiet
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from catanatron import Player
from catanatron.state_functions import (
    get_actual_victory_points,
    get_player_freqdeck,
    get_enemy_colors,
)

# Optional state helpers (exist in many Catanatron installs, but we guard anyway).
try:
    from catanatron.state_functions import get_largest_army  # type: ignore
except Exception:  # pragma: no cover
    def get_largest_army(state: Any) -> Any:
        return None

try:
    from catanatron.state_functions import get_longest_road_color  # type: ignore
except Exception:  # pragma: no cover
    def get_longest_road_color(state: Any) -> Any:
        return None

try:
    from catanatron.state_functions import get_player_buildings  # type: ignore
except Exception:  # pragma: no cover
    get_player_buildings = None  # type: ignore

try:
    from catanatron.state_functions import get_dev_cards_in_hand  # type: ignore
except Exception:  # pragma: no cover
    get_dev_cards_in_hand = None  # type: ignore

# Optional simulation utilities.
try:
    from catanatron.apply_action import apply_action  # type: ignore
    from catanatron.state import State  # type: ignore
except Exception:  # pragma: no cover
    apply_action = None  # type: ignore
    State = None  # type: ignore

# Resource constants: prefer catanatron enums if present, otherwise strings.
try:
    from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE  # type: ignore
except Exception:  # pragma: no cover
    WOOD, BRICK, SHEEP, WHEAT, ORE = "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"

RESOURCES: List[str] = [WOOD, BRICK, SHEEP, WHEAT, ORE]
RES_IDX: Dict[str, int] = {r: i for i, r in enumerate(RESOURCES)}

# Standard build costs in freqdeck format: [WOOD, BRICK, SHEEP, WHEAT, ORE]
COST_ROAD = [1, 1, 0, 0, 0]
COST_SETTLEMENT = [1, 1, 1, 1, 0]
COST_CITY = [0, 0, 0, 2, 3]
COST_DEV = [0, 0, 1, 1, 1]

# Exact two-d6 probabilities.
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
    """If x is an Enum-like with .value, return .value, else return x."""
    return getattr(x, "value", x)


def _as_resource_name(x: Any) -> str:
    """Normalize a resource representation to our string constants when possible."""
    v = _safe_enum_value(x)
    if isinstance(v, str):
        # Some enums stringify like 'Resource.WOOD' — salvage the tail if needed.
        if "." in v and v.split(".")[-1] in {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"}:
            return v.split(".")[-1]
        return v
    return str(v)


def _freqdeck_total(fd: Sequence[int]) -> int:
    return int(sum(fd))


def _freqdeck_add(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [int(a[i] + b[i]) for i in range(5)]


def _freqdeck_sub(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [int(a[i] - b[i]) for i in range(5)]


def _freqdeck_deficit(have: Sequence[int], cost: Sequence[int]) -> List[int]:
    return [max(0, int(cost[i] - have[i])) for i in range(5)]


def _expected_turns_to_afford(hand: Sequence[int], inflow: Sequence[float], cost: Sequence[int]) -> float:
    """
    Heuristic: turns needed to cover each resource deficit at its expected inflow rate.
    Uses max across resources because you need *all* components.
    """
    deficits = _freqdeck_deficit(hand, cost)
    turns = 0.0
    for i, d in enumerate(deficits):
        if d <= 0:
            continue
        rate = float(inflow[i])
        if rate <= 1e-9:
            return 99.0  # effectively unreachable without trades
        turns = max(turns, d / rate)
    return turns


def _try_get_player_buildings(state: Any, color: Any, kind: str) -> List[Any]:
    """
    Best-effort building retrieval. Returns list of node_ids (settlements/cities) or edge_ids (roads).
    """
    if get_player_buildings is not None:
        try:
            return list(get_player_buildings(state, color, kind))  # type: ignore
        except Exception:
            pass

    # Fallback: try common state attributes.
    for attr in ["player_buildings", "buildings", "structures"]:
        obj = getattr(state, attr, None)
        if obj is None:
            continue
        try:
            # Try dict-like: obj[color][kind]
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
    """
    Best-effort extraction of resource type from a tile object.
    Returns one of RESOURCES or None if unknown/desert.
    """
    for attr in ["resource", "resource_type", "terrain", "type"]:
        if hasattr(tile, attr):
            r = _as_resource_name(getattr(tile, attr))
            if r in RES_IDX:
                return r
    # Sometimes tiles are dict-like
    if isinstance(tile, dict):
        for k in ["resource", "resource_type", "terrain", "type"]:
            if k in tile:
                r = _as_resource_name(tile[k])
                if r in RES_IDX:
                    return r
    return None


def _tile_number(tile: Any) -> Optional[int]:
    """
    Best-effort extraction of dice number/token from a tile object.
    """
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
    """
    Try hard to obtain tiles adjacent to a vertex/node.
    Different Catanatron versions expose different APIs, so this is intentionally defensive.
    """
    if board is None:
        return []

    # Common method names.
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

    # Common mapping attributes.
    for attr in ["node_to_tiles", "node_tiles", "tiles_by_node", "adjacent_tiles_by_node"]:
        m = getattr(board, attr, None)
        if isinstance(m, dict) and node_id in m:
            try:
                return list(m[node_id])
            except Exception:
                return []

    return []


def _node_production_profile(board: Any, node_id: Any) -> Tuple[List[float], float, int]:
    """
    Returns (expected_inflow_per_resource[5], variance_proxy, distinct_resource_count)
    using dice probabilities and adjacent tiles.
    """
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
        variance += p * (1.0 - p)  # per-tile Bernoulli-ish proxy
        seen.add(r)

    return inflow, variance, len(seen)


def _estimate_player_inflow(state: Any, color: Any) -> Tuple[List[float], float]:
    """
    Estimate expected resource inflow per turn (per dice roll) and a variance proxy.
    Uses settlements as weight 1, cities as weight 2.
    """
    board = _board_from_state(state)
    settlements = _try_get_player_buildings(state, color, "SETTLEMENT")
    cities = _try_get_player_buildings(state, color, "CITY")

    inflow = [0.0] * 5
    variance = 0.0

    for node in settlements:
        node_in, node_var, _ = _node_production_profile(board, node)
        inflow = [inflow[i] + node_in[i] * 1.0 for i in range(5)]
        variance += node_var * 1.0

    for node in cities:
        node_in, node_var, _ = _node_production_profile(board, node)
        inflow = [inflow[i] + node_in[i] * 2.0 for i in range(5)]
        variance += node_var * 4.0  # (2x payout) => variance scales ~square

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
        # Older catanatron variants expect (colors, player_color)
        enemies = list(get_enemy_colors(getattr(state, "colors", ()), me))
    best_c = None
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
    """
    Check if any playable action would bring us to 10+ VP after application.
    If simulation isn't available, return None.
    """
    if apply_action is None or State is None:
        return None
    for a in playable_actions:
        ns = _simulate_if_possible(state, a)
        if ns is None:
            continue
        try:
            if int(get_actual_victory_points(ns, me)) >= 10:
                return a
        except Exception:
            continue
    return None


def _action_type(a: Any) -> str:
    return str(_safe_enum_value(a.action_type.value))


def _parse_trade_value(value: Any) -> Tuple[List[int], List[int], Optional[Any]]:
    """
    Best-effort parsing of trade-like action.value.
    Returns: (give_freqdeck[5], get_freqdeck[5], maybe_partner)
    """
    give = [0, 0, 0, 0, 0]
    get = [0, 0, 0, 0, 0]
    partner = None

    if value is None:
        return give, get, partner

    # OFFER_TRADE often: 10-tuple of ints: give5 + get5
    if isinstance(value, tuple) and len(value) == 10 and all(isinstance(x, (int, float)) for x in value):
        g = list(value[:5])
        r = list(value[5:])
        return [int(x) for x in g], [int(x) for x in r], None

    # ACCEPT_TRADE sometimes includes (partner, give5, get5) or similar.
    if isinstance(value, tuple) and len(value) >= 11:
        # Try to find two consecutive 5-length int blocks.
        ints = [isinstance(x, (int, float)) for x in value]
        for start in range(len(value) - 10):
            block1 = value[start : start + 5]
            block2 = value[start + 5 : start + 10]
            if all(isinstance(x, (int, float)) for x in block1) and all(isinstance(x, (int, float)) for x in block2):
                partner = value[0] if start != 0 else None
                return [int(x) for x in block1], [int(x) for x in block2], partner

    # MARITIME_TRADE guide-style: (give1,give2,give3,give4,receive) as resources
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


# -----------------------------------------------------------------------------
# STAT Engine
# -----------------------------------------------------------------------------

class StatsEngine(Player):
    """
    Math/statistics engine:
    - Computes dice distribution exactly.
    - Estimates expected inflow & variance from current production.
    - Converts actions into changes in "expected turns to next VP-ish purchase".
    - Risk-adjusted scoring: variance penalty + robber-card penalty.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._rng = random.Random(1337)  # deterministic-ish tie-breaking

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
            return self._pick_best_by_metric(state, playable_actions)
        if first_type == "MOVE_ROBBER":
            return self._pick_robber_move(state, playable_actions)

        base_metrics = self._metrics(state)

        best_a = playable_actions[0]
        best_s = -1e18

        for a in playable_actions:
            at = _action_type(a)

            # Fast-path: rolling isn't a strategic choice.
            if at == "ROLL":
                return a

            ns = _simulate_if_possible(state, a)
            if ns is None:
                s = self._fallback_type_score(state, a)
            else:
                after = self._metrics(ns)
                s = self._score_action(at, base_metrics, after)

                # Tiny tie-break noise, but far smaller than real signal.
                s += self._rng.random() * 0.0005

            if s > best_s:
                best_s = s
                best_a = a

        return best_a

    # ---- Metrics & scoring ----

    def _metrics(self, state: Any) -> Dict[str, Any]:
        vp = int(get_actual_victory_points(state, self.color))
        hand = get_player_freqdeck(state, self.color)
        cards = _freqdeck_total(hand)

        inflow, var = _estimate_player_inflow(state, self.color)

        # Expected turns to next "VP-ish improvement".
        t_settle = _expected_turns_to_afford(hand, inflow, COST_SETTLEMENT)
        t_city = _expected_turns_to_afford(hand, inflow, COST_CITY)
        t_dev = _expected_turns_to_afford(hand, inflow, COST_DEV)

        # Dev cards have ~20% chance of VP in standard deck; also army pressure exists.
        # We treat dev as fractional VP progress.
        t_next = min(t_settle, t_city, t_dev / 0.85)  # dev slightly discounted

        return {
            "vp": vp,
            "hand": hand,
            "cards": cards,
            "inflow": inflow,
            "var": var,
            "t_next": float(t_next),
            "t_settle": float(t_settle),
            "t_city": float(t_city),
            "t_dev": float(t_dev),
        }

    def _score_action(self, action_type: str, before: Dict[str, Any], after: Dict[str, Any]) -> float:
        vp_delta = after["vp"] - before["vp"]

        # Core objective: win fast (minimize time-to-next-VP), but not by courting disaster.
        t_gain = before["t_next"] - after["t_next"]

        # Risk model:
        # - variance: high variance is volatile; penalize it unless we're desperate.
        # - robber exposure: convex penalty when >=7 cards.
        var_pen = 8.0 * float(after["var"])
        robber_pen = 3.2 * max(0.0, float(after["cards"]) - 6.0) ** 2

        # If we're behind in VP, allow more risk-taking (variance penalty softens).
        leader, leader_vp = _leader_color_and_vp_from_metrics_guess(before, self.color)
        behind = max(0, leader_vp - before["vp"])
        var_pen *= 1.0 / (1.0 + 0.25 * behind)

        # Action-type priors: small, because metrics should dominate.
        pri = {
            "BUILD_CITY": 10.0,
            "BUILD_SETTLEMENT": 9.0,
            "BUY_DEVELOPMENT_CARD": 5.0,
            "PLAY_KNIGHT_CARD": 3.0,
            "PLAY_MONOPOLY": 4.0,
            "PLAY_YEAR_OF_PLENTY": 4.0,
            "PLAY_ROAD_BUILDING": 2.0,
            "BUILD_ROAD": 2.0,
            "MARITIME_TRADE": 1.0,
            "ACCEPT_TRADE": 0.6,
            "OFFER_TRADE": 0.2,
            "END_TURN": 0.0,
        }.get(action_type, 0.0)

        # Weighting knobs (easy to tweak):
        return (
            220.0 * vp_delta
            + 60.0 * t_gain
            + pri
            - var_pen
            - robber_pen
        )

    def _fallback_type_score(self, state: Any, action: Any) -> float:
        at = _action_type(action)
        score = {
            "BUILD_CITY": 100.0,
            "BUILD_SETTLEMENT": 85.0,
            "BUY_DEVELOPMENT_CARD": 55.0,
            "PLAY_YEAR_OF_PLENTY": 50.0,
            "PLAY_MONOPOLY": 48.0,
            "PLAY_KNIGHT_CARD": 40.0,
            "BUILD_ROAD": 30.0,
            "MARITIME_TRADE": 12.0,
            "ACCEPT_TRADE": 10.0,
            "OFFER_TRADE": 5.0,
            "END_TURN": 0.0,
            "ROLL": 1.0,
        }.get(at, 0.0)

        # Spend-down bias at robber risk.
        cards = _freqdeck_total(get_player_freqdeck(state, self.color))
        if cards >= 7 and at in {"BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD", "BUILD_ROAD"}:
            score += 25.0

        return score

    # ---- Special phases ----

    def _pick_best_by_metric(self, state: Any, actions: Sequence[Any]) -> Any:
        base = self._metrics(state)
        best_a = actions[0]
        best_s = -1e18
        for a in actions:
            ns = _simulate_if_possible(state, a)
            if ns is None:
                continue
            after = self._metrics(ns)
            s = self._score_action("DISCARD", base, after)
            if s > best_s:
                best_s = s
                best_a = a
        return best_a

    def _pick_robber_move(self, state: Any, actions: Sequence[Any]) -> Any:
        leader, leader_vp = _leader_color_and_vp(state, self.color)

        best_a = actions[0]
        best_s = -1e18
        for a in actions:
            v = getattr(a, "value", None)
            target = None
            if isinstance(v, tuple) and len(v) >= 2:
                target = v[1]

            # Score: prefer targeting leader; then prefer people with more cards.
            s = 0.0
            if leader is not None and target == leader:
                s += 50.0 + 5.0 * leader_vp
            if target is not None:
                try:
                    s += 1.2 * _freqdeck_total(get_player_freqdeck(state, target))
                except Exception:
                    pass
            if target == self.color:
                s -= 100.0

            if s > best_s:
                best_s = s
                best_a = a
        return best_a

    def _pick_initial(self, game: Any, actions: Sequence[Any]) -> Any:
        """
        Stats-y initial placement:
        maximize expected inflow; prefer low variance + diversity.
        """
        state = game.state
        at = _action_type(actions[0])
        board = _board_from_state(state)

        if at == "BUILD_INITIAL_SETTLEMENT":
            best_a = actions[0]
            best_s = -1e18
            for a in actions:
                node = getattr(a, "value", None)
                inflow, var, div = _node_production_profile(board, node)
                s = 120.0 * sum(inflow) + 6.0 * div - 8.0 * var
                s += self._rng.random() * 0.001
                if s > best_s:
                    best_s = s
                    best_a = a
            return best_a

        if at == "BUILD_INITIAL_ROAD":
            # Push toward the endpoint with higher expected inflow.
            best_a = actions[0]
            best_s = -1e18
            for a in actions:
                edge = getattr(a, "value", None)
                s = 0.0
                if isinstance(edge, tuple) and len(edge) == 2:
                    n1, n2 = edge
                    s = 50.0 * max(sum(_node_production_profile(board, n1)[0]), sum(_node_production_profile(board, n2)[0]))
                s += self._rng.random() * 0.001
                if s > best_s:
                    best_s = s
                    best_a = a
            return best_a

        return actions[0]


def _leader_color_and_vp_from_metrics_guess(before_metrics: Dict[str, Any], me: Any) -> Tuple[Optional[Any], int]:
    """
    Metrics don't include enemy VP; this helper is used only to modulate variance penalty.
    We keep it simple: assume leader is at least us (so behind=0) if we can't compute.
    """
    # This function exists to avoid dragging enemy lookups into every metric call.
    # It's intentionally conservative.
    return None, int(before_metrics.get("vp", 0))


# -----------------------------------------------------------------------------
# WILD Engine (Sheep Hoarder Cult)
# -----------------------------------------------------------------------------

class WildSheepCultEngine(Player):
    """
    Slightly crazy, still competent:
    - Coherent theme: sheep is sacred; actions that increase sheep get big bonuses.
    - Safety floor: take winning move; spend down when robber-risk; discard non-sheep first.
    - Otherwise: score actions then choose via weighted randomness among top candidates.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._rng = random.Random()

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
            return self._discard_like_a_cultist(state, playable_actions)
        if first_type == "MOVE_ROBBER":
            return self._robber_like_a_cultist(state, playable_actions)

        # Robber-risk spend-down: if we can spend, do it.
        my_cards = _freqdeck_total(get_player_freqdeck(state, self.color))
        if my_cards >= 7:
            spender = self._find_spend_down(playable_actions)
            if spender is not None:
                return spender

        # Score actions, then pick with controlled chaos.
        scored: List[Tuple[float, Any]] = []
        for a in playable_actions:
            s = self._score_action(state, a)
            scored.append((s, a))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: min(6, len(scored))]

        # Weighted randomness among top actions (more weight to higher score).
        # This makes it "personality" rather than "coin flip".
        weights = [max(0.01, t[0] - top[-1][0] + 0.05) for t in top]
        choice = self._rng.choices([t[1] for t in top], weights=weights, k=1)[0]
        return choice

    # ---- Theme scoring ----

    def _score_action(self, state: Any, action: Any) -> float:
        at = _action_type(action)
        hand = get_player_freqdeck(state, self.color)
        sheep_count = hand[RES_IDX[SHEEP]]

        # Base competence (so it still plays Catan).
        base = {
            "ROLL": 1.0,
            "BUILD_CITY": 10.0,
            "BUILD_SETTLEMENT": 9.0,
            "BUY_DEVELOPMENT_CARD": 6.0,
            "BUILD_ROAD": 7.0,  # roads are "ritual circles"
            "PLAY_KNIGHT_CARD": 4.0,
            "PLAY_ROAD_BUILDING": 6.0,
            "PLAY_YEAR_OF_PLENTY": 7.0,
            "PLAY_MONOPOLY": 7.0,
            "MARITIME_TRADE": 2.0,
            "ACCEPT_TRADE": 2.5,
            "OFFER_TRADE": 1.5,
            "END_TURN": 0.0,
        }.get(at, 0.5)

        s = base

        # Sheep obsession: reward actions that explicitly mention sheep.
        v = getattr(action, "value", None)

        if at == "PLAY_YEAR_OF_PLENTY" and isinstance(v, tuple):
            r1 = _as_resource_name(v[0]) if len(v) > 0 else ""
            r2 = _as_resource_name(v[1]) if len(v) > 1 else ""
            s += (18.0 if r1 == SHEEP else 0.0) + (18.0 if r2 == SHEEP else 0.0)

        if at == "PLAY_MONOPOLY":
            r = _as_resource_name(v)
            if r == SHEEP:
                s += 30.0
            else:
                s += 6.0

        if at in {"MARITIME_TRADE", "OFFER_TRADE", "ACCEPT_TRADE"}:
            give, get, _ = _parse_trade_value(v)
            net_sheep = get[RES_IDX[SHEEP]] - give[RES_IDX[SHEEP]]
            s += 12.0 * net_sheep  # trades that grow the flock are holy
            # Slight penalty for giving away sheep.
            if net_sheep < 0:
                s -= 8.0 * (-net_sheep)

        # Settlement placement: if we can estimate sheep production at that node, reward it.
        if at in {"BUILD_SETTLEMENT", "BUILD_INITIAL_SETTLEMENT"}:
            node = v
            board = _board_from_state(state)
            inflow, _, div = _node_production_profile(board, node)
            s += 120.0 * inflow[RES_IDX[SHEEP]]
            s += 2.0 * div  # diversity still matters: cults need calories too

        # Dev cards are liked more when sheep is abundant (because sheep feels "safe").
        if at == "BUY_DEVELOPMENT_CARD":
            s += 1.5 * sheep_count

        # Mood noise: keeps it from being predictable.
        s += self._rng.random() * 1.2

        return s

    # ---- Safety & special phases ----

    def _find_spend_down(self, actions: Sequence[Any]) -> Optional[Any]:
        for preferred in ["BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD", "BUILD_ROAD"]:
            for a in actions:
                if _action_type(a) == preferred:
                    return a
        return None

    def _discard_like_a_cultist(self, state: Any, actions: Sequence[Any]) -> Any:
        """
        Keep sheep if at all possible.
        If simulation exists, choose discard action that preserves sheep and keeps utility.
        """
        # If we can simulate, choose discard leaving us with maximum sheep.
        best_a = actions[0]
        best_s = -1e18
        for a in actions:
            ns = _simulate_if_possible(state, a)
            if ns is None:
                continue
            hand = get_player_freqdeck(ns, self.color)
            sheep_left = hand[RES_IDX[SHEEP]]
            cards_left = _freqdeck_total(hand)
            # Prefer more sheep; also prefer fewer cards (robber safety).
            s = 100.0 * sheep_left - 1.5 * max(0, cards_left - 6)
            s += self._rng.random() * 0.01
            if s > best_s:
                best_s = s
                best_a = a
        return best_a

    def _robber_like_a_cultist(self, state: Any, actions: Sequence[Any]) -> Any:
        """
        Target the player with the most sheep (or the leader if sheep is unknown),
        but avoid self-targeting.
        """
        leader, _ = _leader_color_and_vp(state, self.color)

        best_a = actions[0]
        best_s = -1e18
        for a in actions:
            v = getattr(a, "value", None)
            target = None
            if isinstance(v, tuple) and len(v) >= 2:
                target = v[1]

            if target == self.color:
                continue

            s = 0.0
            if target is not None:
                try:
                    t_hand = get_player_freqdeck(state, target)
                    s += 25.0 * t_hand[RES_IDX[SHEEP]]
                    s += 0.4 * _freqdeck_total(t_hand)
                except Exception:
                    pass

            if leader is not None and target == leader:
                s += 8.0  # leader is always at least somewhat tasty

            s += self._rng.random() * 0.01
            if s > best_s:
                best_s = s
                best_a = a

        return best_a

    def _pick_initial(self, game: Any, actions: Sequence[Any]) -> Any:
        """
        Initial placement, cult edition:
        - Prefer sheep production, then overall inflow and diversity.
        """
        state = game.state
        at = _action_type(actions[0])
        board = _board_from_state(state)

        if at == "BUILD_INITIAL_SETTLEMENT":
            best_a = actions[0]
            best_s = -1e18
            for a in actions:
                node = getattr(a, "value", None)
                inflow, var, div = _node_production_profile(board, node)
                s = 220.0 * inflow[RES_IDX[SHEEP]] + 90.0 * sum(inflow) + 6.0 * div - 4.0 * var
                s += self._rng.random() * 0.001
                if s > best_s:
                    best_s = s
                    best_a = a
            return best_a

        if at == "BUILD_INITIAL_ROAD":
            # Roads are art: choose the one leading to the sheepiest endpoint if possible.
            best_a = actions[0]
            best_s = -1e18
            for a in actions:
                edge = getattr(a, "value", None)
                s = 0.0
                if isinstance(edge, tuple) and len(edge) == 2:
                    n1, n2 = edge
                    p1 = _node_production_profile(board, n1)[0]
                    p2 = _node_production_profile(board, n2)[0]
                    s = 120.0 * max(p1[RES_IDX[SHEEP]], p2[RES_IDX[SHEEP]]) + 30.0 * max(sum(p1), sum(p2))
                s += self._rng.random() * 0.001
                if s > best_s:
                    best_s = s
                    best_a = a
            return best_a

        return actions[0]

