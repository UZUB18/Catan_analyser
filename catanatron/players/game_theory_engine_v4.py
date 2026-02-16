"""
game_theory_engine_v4.py

GTv4 ("AB killer") is a small, targeted evolution of GTv3 focused on beating the
stock `AlphaBetaPlayer` baseline in benchmark ladders.

Implemented upgrades (from the GTv3 AB-killer roadmap):
  3) Dynamic AB-delegate depth (2 -> 3 when it's cheap or high-leverage)
  4) AB-action sanity veto layer (trade feed / kingmaking guardrails)
  5) Instrumentation counters for systematic tuning

Notes:
- GTv4 intentionally reuses the mature GTv3 core (opening, discard, robber,
  evaluator, TT, iterative same-turn search) and only overrides the parts that
  matter for AB matchups.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Sequence, Tuple

from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import DEFAULT_WEIGHTS
from catanatron.state_functions import get_actual_victory_points, get_player_freqdeck

from catanatron.players.game_theory_engine_v3 import (  # re-exported helpers
    BRICK,
    ORE,
    SHEEP,
    WHEAT,
    WOOD,
    GameTheoryEngineV3,
    RES_IDX,
    _action_sig,
    _action_type,
    _leader_color_and_vp,
    _parse_trade_value,
    _try_execute_game,
    _try_get_player_buildings,
)


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


class GameTheoryEngineV4(GameTheoryEngineV3):
    """
    GTv4: GTv3 + AB-killer enhancements (dynamic AB depth, sanity veto, instrumentation).

    Primary usage:
      - `GTv4:profile=ab_killer`
    """

    VERSION = "4.0.0"

    def __init__(self, *args: Any, **kwargs: Any):
        # Support CLI-style extra positional args as "key=value" tokens.
        color = None
        extra_tokens = []
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

        profile = str(kwargs.get("profile", "balanced")).strip().lower()
        is_ab_killer_profile = profile in {"ab_killer", "abkiller", "anti_ab", "vs_ab"}

        # --- Upgrade 3: Dynamic AB depth ---
        self.ab_tactical_dynamic_depth = _boolish(
            kwargs.pop("ab_tactical_dynamic_depth", True if is_ab_killer_profile else False),
            default=True if is_ab_killer_profile else False,
        )
        self.ab_tactical_depth_hi = int(kwargs.pop("ab_tactical_depth_hi", 3))
        self.ab_tactical_depth_hi_endgame_vp = int(kwargs.pop("ab_tactical_depth_hi_endgame_vp", 8))
        self.ab_tactical_depth_hi_small_branching = int(kwargs.pop("ab_tactical_depth_hi_small_branching", 6))
        self.ab_tactical_depth_hi_max_branching = int(kwargs.pop("ab_tactical_depth_hi_max_branching", 12))
        self.ab_tactical_depth_hi_conversion_max_branching = int(
            kwargs.pop("ab_tactical_depth_hi_conversion_max_branching", 10)
        )

        # --- Upgrade 4: Sanity veto ---
        self.sanity_veto_enabled = _boolish(
            kwargs.pop("sanity_veto_enabled", True if is_ab_killer_profile else False),
            default=True if is_ab_killer_profile else False,
        )
        self.sanity_veto_trade_feed = _boolish(kwargs.pop("sanity_veto_trade_feed", True), default=True)
        self.sanity_veto_leader_vp = int(kwargs.pop("sanity_veto_leader_vp", 9))

        # --- Upgrade 5: Instrumentation ---
        self.instrumentation_enabled = _boolish(kwargs.pop("instrumentation_enabled", True), default=True)

        # Initialize GTv3 base.
        super().__init__(color, **kwargs)

        # AB delegate cache keyed by (depth, prunning, builder, use_none_params).
        self._ab_core_cache: Dict[Tuple[int, bool, str, bool], AlphaBetaPlayer] = {}

        # Counters (per-game, cleared in reset_state).
        self._action_type_counts: Counter[str] = Counter()
        self._ab_depth_counts: Counter[int] = Counter()
        self._ab_veto_counts: Counter[str] = Counter()
        self._sanity_veto_counts: Counter[str] = Counter()

        # Per-decision scratch (not persisted across moves).
        self._last_ab_depth_used: Optional[int] = None
        self._last_ab_veto_reason: Optional[str] = None
        self._last_sanity_veto_reason: Optional[str] = None
        self._last_sanity_veto_from: Optional[str] = None

    # -------------------------
    # Upgrade 3: Dynamic AB-delegate depth
    # -------------------------

    def _choose_ab_tactical_depth(self, state: Any, playable_actions: Sequence[Any], leader_vp: int) -> int:
        base_depth = max(1, int(self.ab_tactical_depth))
        if not self.ab_tactical_dynamic_depth:
            return base_depth

        hi_depth = max(base_depth, int(self.ab_tactical_depth_hi))
        if hi_depth <= base_depth:
            return base_depth

        branching = len(playable_actions)
        if branching <= max(1, int(self.ab_tactical_depth_hi_small_branching)):
            return hi_depth
        if branching > max(1, int(self.ab_tactical_depth_hi_max_branching)):
            return base_depth

        my_vp = 0
        try:
            my_vp = int(get_actual_victory_points(state, self.color))
        except Exception:
            my_vp = 0

        if my_vp >= int(self.ab_tactical_depth_hi_endgame_vp) or leader_vp >= int(self.ab_tactical_depth_hi_endgame_vp):
            return hi_depth

        types = {_action_type(a) for a in playable_actions}
        hard_conversion = bool(
            types.intersection(
                {
                    "BUILD_CITY",
                    "BUILD_SETTLEMENT",
                    "BUY_DEVELOPMENT_CARD",
                    "PLAY_YEAR_OF_PLENTY",
                    "PLAY_MONOPOLY",
                    "PLAY_ROAD_BUILDING",
                    "PLAY_KNIGHT_CARD",
                }
            )
        )
        if hard_conversion and branching <= max(1, int(self.ab_tactical_depth_hi_conversion_max_branching)):
            return hi_depth

        return base_depth

    def _get_ab_tactical_core_for_depth(self, depth: int) -> AlphaBetaPlayer:
        depth = max(1, int(depth))
        builder = "C" if self.ab_tactical_value_fn == "contender_fn" else "base_fn"
        params = None if (builder == "C" and self.ab_tactical_use_none_params) else DEFAULT_WEIGHTS
        key = (depth, bool(self.ab_tactical_prunning), builder, bool(self.ab_tactical_use_none_params))

        core = self._ab_core_cache.get(key)
        if core is None or getattr(core, "color", None) != self.color:
            core = AlphaBetaPlayer(
                self.color,
                depth=depth,
                prunning=self.ab_tactical_prunning,
                value_fn_builder_name=builder,
                params=params,
            )
            self._ab_core_cache[key] = core
        return core

    # -------------------------
    # Upgrade 4: "Sanity veto" layer
    # -------------------------

    def _hand_tuple(self, state: Any, color: Any) -> Tuple[int, int, int, int, int]:
        try:
            return tuple(int(x) for x in (get_player_freqdeck(state, color) or (0, 0, 0, 0, 0)))  # type: ignore[return-value]
        except Exception:
            return (0, 0, 0, 0, 0)

    def _can_afford_city(self, hand: Tuple[int, int, int, int, int], state: Any, color: Any) -> bool:
        try:
            wheat_i = int(RES_IDX[_as_key(WHEAT)])
            ore_i = int(RES_IDX[_as_key(ORE)])
        except Exception:
            wheat_i, ore_i = 3, 4

        if hand[wheat_i] < 2 or hand[ore_i] < 3:
            return False

        # Must have a settlement to upgrade (best-effort check).
        try:
            return len(_try_get_player_buildings(state, color, "SETTLEMENT")) > 0
        except Exception:
            return True

    def _can_afford_settlement(self, hand: Tuple[int, int, int, int, int]) -> bool:
        try:
            wood_i = int(RES_IDX[_as_key(WOOD)])
            brick_i = int(RES_IDX[_as_key(BRICK)])
            sheep_i = int(RES_IDX[_as_key(SHEEP)])
            wheat_i = int(RES_IDX[_as_key(WHEAT)])
        except Exception:
            wood_i, brick_i, sheep_i, wheat_i = 0, 1, 2, 3
        return hand[wood_i] >= 1 and hand[brick_i] >= 1 and hand[sheep_i] >= 1 and hand[wheat_i] >= 1

    def _trade_would_enable_leader_vp_build(
        self,
        game: Any,
        offer: Sequence[int],
        ask: Sequence[int],
        leader: Any,
    ) -> Optional[str]:
        """
        Returns a reason string if the *resulting* leader hand enables a city/settlement
        when it previously could not. Otherwise returns None.

        Note:
        - For CONFIRM_TRADE, the leader is the partner (enemy) and receives `offer` and gives `ask`.
        - For ACCEPT_TRADE, the leader is the offerer and receives `ask` and gives `offer`.
        """
        leader_hand_before = self._hand_tuple(game.state, leader)
        leader_hand_after = tuple(int(leader_hand_before[i] - int(ask[i]) + int(offer[i])) for i in range(5))

        city_before = self._can_afford_city(leader_hand_before, game.state, leader)
        city_after = self._can_afford_city(leader_hand_after, game.state, leader)
        if (not city_before) and city_after:
            return "trade_enables_leader_city"

        sett_before = self._can_afford_settlement(leader_hand_before)
        sett_after = self._can_afford_settlement(leader_hand_after)
        if (not sett_before) and sett_after:
            return "trade_enables_leader_settlement"

        return None

    def _maybe_apply_sanity_veto(self, game: Any, action: Any, playable_actions: Sequence[Any]) -> Any:
        if not self.sanity_veto_enabled or not self.sanity_veto_trade_feed:
            return action

        at = _action_type(action)
        if at not in {"CONFIRM_TRADE", "ACCEPT_TRADE"}:
            return action

        leader, leader_vp = _leader_color_and_vp(game.state, self.color)
        if leader is None or int(leader_vp) < int(self.sanity_veto_leader_vp):
            return action

        give, get, partner = _parse_trade_value(getattr(action, "value", None))

        # Map partner when stored as offering turn index (DECIDE_TRADE uses current_trade).
        if isinstance(partner, int):
            try:
                colors = list(getattr(game.state, "colors", []))
                if 0 <= int(partner) < len(colors):
                    partner = colors[int(partner)]
            except Exception:
                pass

        if partner != leader:
            return action

        # Determine (offer, ask) from the *leader hand* perspective.
        # - CONFIRM_TRADE: enemy (partner) gets "give" and gives "get".
        # - ACCEPT_TRADE (partner is offerer): offerer gives "give" and gets "get",
        #   so leader receives "get" and gives "give".
        if at == "CONFIRM_TRADE":
            offer_to_leader = list(int(x) for x in give)
            ask_from_leader = list(int(x) for x in get)
        else:  # ACCEPT_TRADE
            offer_to_leader = list(int(x) for x in get)
            ask_from_leader = list(int(x) for x in give)

        reason = self._trade_would_enable_leader_vp_build(
            game=game,
            offer=offer_to_leader,
            ask=ask_from_leader,
            leader=leader,
        )
        if reason is None:
            return action

        # Veto: choose the safe fallback action.
        self._last_sanity_veto_reason = reason
        self._last_sanity_veto_from = at
        self._sanity_veto_counts[reason] += 1

        # Prefer confirming with a non-leader if possible; otherwise cancel/reject.
        if at == "CONFIRM_TRADE":
            for a in playable_actions:
                if _action_type(a) != "CONFIRM_TRADE":
                    continue
                _g, _k, p = _parse_trade_value(getattr(a, "value", None))
                if p != leader:
                    return a
            for a in playable_actions:
                if _action_type(a) == "CANCEL_TRADE":
                    return a
            return action

        # ACCEPT_TRADE
        for a in playable_actions:
            if _action_type(a) == "REJECT_TRADE":
                return a
        return action

    # -------------------------
    # Upgrade 3+5: AB delegate override with depth selection + counters
    # -------------------------

    def _decide_with_ab_tactical(self, game: Any, playable_actions: Sequence[Any]) -> Optional[Any]:
        self._last_ab_depth_used = None
        self._last_ab_veto_reason = None

        try:
            leader, leader_vp = _leader_color_and_vp(game.state, self.color)
            depth = self._choose_ab_tactical_depth(game.state, playable_actions, leader_vp=int(leader_vp))
            ab = self._get_ab_tactical_core_for_depth(depth)

            self._ab_calls_this_turn += 1
            self._ab_tactical_calls_total += 1
            self._last_ab_depth_used = int(depth)
            self._ab_depth_counts[int(depth)] += 1

            action = ab.decide(game, playable_actions)
            if action not in playable_actions:
                return None

            # Optional AB-specific veto hook (kept conservative; most trade prompts are out-of-turn).
            # If we veto, return None and let GTv3 scoring handle it.
            if self.sanity_veto_enabled and self.sanity_veto_trade_feed:
                # This only triggers if AB is used in a trade-resolving prompt (prompt guard disabled).
                vetoed = self._maybe_apply_sanity_veto(game, action, playable_actions)
                if vetoed is not action:
                    self._last_ab_veto_reason = str(self._last_sanity_veto_reason or "sanity_veto")
                    self._ab_veto_counts[self._last_ab_veto_reason] += 1
                    return None

            return action
        except Exception:
            return None

    # -------------------------
    # Upgrade 5: instrumentation + sanity veto integration
    # -------------------------

    def decide(self, game: Any, playable_actions: Sequence[Any]) -> Any:
        # Reset per-decision veto markers.
        self._last_sanity_veto_reason = None
        self._last_sanity_veto_from = None
        self._last_ab_veto_reason = None

        action = super().decide(game, playable_actions)

        # Sanity veto pass (covers trade prompts where AB delegate is intentionally disabled).
        action2 = self._maybe_apply_sanity_veto(game, action, playable_actions)
        if action2 is not action:
            # Make sure diagnostics reflect that we overrode the final action.
            try:
                self.last_decision_info = dict(self.last_decision_info)
                self.last_decision_info["sanity_veto"] = self._last_sanity_veto_reason
                self.last_decision_info["sanity_veto_from"] = self._last_sanity_veto_from
                self.last_decision_info["sanity_veto_to"] = _action_type(action2)
                self.last_decision_info["sanity_veto_from_sig"] = _action_sig(action)
                self.last_decision_info["sanity_veto_to_sig"] = _action_sig(action2)
            except Exception:
                pass
            action = action2

        # Lightweight counters (per-game).
        if self.instrumentation_enabled:
            at = _action_type(action)
            self._action_type_counts[at] += 1
            try:
                self.last_decision_info = dict(self.last_decision_info)
                self.last_decision_info["action_type"] = at
                self.last_decision_info["action_sig"] = _action_sig(action)
                if self.last_decision_info.get("mode") == "ab_tactical" and self._last_ab_depth_used is not None:
                    self.last_decision_info["ab_depth_used"] = int(self._last_ab_depth_used)
                if self._last_ab_veto_reason is not None:
                    self.last_decision_info["ab_veto"] = self._last_ab_veto_reason
            except Exception:
                pass

        return action

    def reset_state(self) -> None:
        # Called by CLI batch-play loops between games.
        try:
            self._action_type_counts.clear()
            self._ab_depth_counts.clear()
            self._ab_veto_counts.clear()
            self._sanity_veto_counts.clear()
        except Exception:
            pass
        self._last_ab_depth_used = None
        self._last_ab_veto_reason = None
        self._last_sanity_veto_reason = None
        self._last_sanity_veto_from = None
        super().reset_state()

    def diagnostics(self) -> Dict[str, Any]:
        d = super().diagnostics()
        d["gtv4"] = {
            "ab_dynamic_depth": self.ab_tactical_dynamic_depth,
            "ab_depth_hi": self.ab_tactical_depth_hi,
            "ab_depth_counts": dict(self._ab_depth_counts),
            "ab_veto_counts": dict(self._ab_veto_counts),
            "sanity_veto": {
                "enabled": self.sanity_veto_enabled,
                "trade_feed": self.sanity_veto_trade_feed,
                "leader_vp_threshold": self.sanity_veto_leader_vp,
                "counts": dict(self._sanity_veto_counts),
            },
            "action_types": dict(self._action_type_counts),
        }
        return d


def _as_key(x: Any) -> Any:
    # Helper for RES_IDX access: accept enum/str.
    return getattr(x, "value", x)

