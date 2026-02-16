# GTv3 Hybrid — Logic Upgrade Plan + Integration Notes (Catanatron)

This document fuses:
- The comparative analysis / roadmap PDF (GTv2 vs AlphaBeta, what actually matters) fileciteturn0file0
- The current engine implementation (`GameTheoryEngine` v1.3.0) fileciteturn0file1
- The GTv2 upgrade plan notes (TT/ID/ordering/MCTS ideas) fileciteturn0file2
- The “priority upgrades” code sketch (TT + iterative deepening + history heuristic) fileciteturn0file3

The goal is *not* “search deeper in the abstract”. The goal is **to beat engines that are deep-but-wrong (AlphaBeta in multiplayer)** and **variance-blind (deterministic evaluators)** by combining:
1) better compute efficiency,
2) better algorithmic fit for a stochastic 4-player game,
3) and better tactical sequencing in the only place where Catan is “single-agent”: *your own combo turn*.

---

## 1) The “So what?” diagnosis

### Why AlphaBeta wins games it shouldn’t
AlphaBeta’s advantage is mostly **tactical cleanliness**: it punishes moves that “look good now” but collapse after a short interaction sequence. That’s real.

But in 4-player Catan it’s also systematically wrong because it treats opponents as a single minimizer (paranoid minimax). That produces *anti-human* (and often anti-optimal) behavior:
- over-denial in situations where opponents won’t coordinate,
- under-investment in tempo/resource-flow plays that are “risky” but high EV,
- brittle play under variance.

So the counter-strategy is **not** “be more paranoid”. It’s:
- Go deeper *only where depth is structurally valid* (within-turn combo search),
- Add a light “reality check” about the **very next player’s** reply (not the whole table acting as one demon),
- And keep the opponent-aware objective that handles multiplayer gradients.

---

## 2) What to upgrade (ordered by leverage)

### Tier 0: fix correctness leaks (free strength)
Two bugs in v1.3.0 meaningfully reduce playing strength:
1) `_utility()` incorrectly calls `_try_get_player_buildings` (wrong signature), which can crash or degrade fallbacks.
2) `_trade_stance_score()` calls `_parse_trade_value(action)` instead of `action.value`, weakening trade evaluation.

These are fixed in the integrated `game_theory_engine_v3_0_0.py`.

---

### Tier 1: compute infrastructure (enables *everything*)
**Add a transposition table** with a **stable state key**.

Why it matters:
- It’s the cheapest way to buy effective depth.
- It stabilizes time budgets (less variance in “how long a move takes”).
- It enables iterative deepening because partial results keep value.

**Implementation notes:**
- Use a stable key that survives `Game.copy()` (not `id(state)`).
- Include enough state to avoid false hits: buildings, robber, hands, dev count, VP, turn indices.

---

### Tier 2: tactical sequencing where minimax is valid
**Iterative deepening within-turn expectimax**, with:
- TT lookups,
- PV move ordering (TT-best first),
- lightweight “history” scores to keep trying productive move types early.

This is where you directly outclass AlphaBeta, because AB’s depth is wasted modeling a non-existent unified minimizer, while you spend depth on real tactical sequencing: `trade → build → dev → build → end`.

---

### Tier 3: opponent reply modeling (cheap 2-ply sanity)
Add a small penalty/adjustment for “what happens right after I pass the turn”.

This is *not* minimax. Two practical models:
- **selfish** (default): opponent maximizes their own eval
- **adversarial**: opponent minimizes our objective (useful near endgame or vs denial-heavy bots)

Blend it (don’t fully replace EV): `expected = (1-w)*EV + w*EV_after_reply`.

This corrects a classic GT failure mode: “great EV” moves that hand the next player an immediate conversion.

---

### Tier 4: strategic sampling (optional, expensive)
When branching is ugly (trade explosions), a 1-step EV delta is brittle.
A simple bandit-style rollout layer can help:
- sample outcomes for stochastic actions,
- simulate a short greedy rollout for a few plies,
- choose the best mean.

This isn’t full MCTS, but it approximates the same benefit at a fraction of complexity.

---

## 3) Integration architecture (how the pieces fit)

Decision modes (ordered):
1) **Immediate win check** (already there)
2) **Opening**: existing sim+eval opening is fine
3) **Special prompts**: discard/robber
4) **Our-turn tactical**: iterative-deepening same-turn search (ID + TT)
5) **Strategic**:
   - Default: one-step EV (fast)
   - Optional: MC rollouts (bounded time) when branching is high

This hybrid is why it works: each mode spends compute where it returns real information.

---

## 4) What the integrated code delivers

`game_theory_engine_v3_0_0.py` provides:

- **Bug fixes** (utility + trade parsing)
- **GameTheoryEngineV3**: a subclass of the original engine with
  - persistent LRU transposition table
  - iterative deepening turn search (same interface, time-budgeted)
  - move ordering via TT + history
  - optional opponent reply approximation (blended)
  - optional Monte Carlo rollouts (off by default)

---

## 5) Calibration targets and “done” criteria

You don’t “ship” an engine; you ship a *measured advantage*.

Suggested minimal benchmark protocol:
- AB-depth2 vs GTv3 (n=300 games): target **≥ 65% win rate**
- AB-depth3 vs GTv3 (n=300 games): target **≥ 55% win rate**
- Mean decision time: keep below configured budgets with low variance
- TT hit rate during turn search: aim for **> 20%** (it’ll grow as games progress)

---

## 6) Practical usage

```python
from game_theory_engine_v3_0_0 import GameTheoryEngineV3

engine = GameTheoryEngineV3(
    "RED",
    # TT + ordering
    use_tt=True,
    tt_max_entries=60000,

    # Turn search (same-turn combo)
    use_turn_search=True,
    turn_search_depth=5,              # now interpreted as max depth for ID
    turn_search_width=8,
    turn_search_time_budget_ms=160,
    turn_search_id=True,

    # Opponent reply approximation
    use_opp_reply=True,
    opp_reply_weight=0.55,
    opp_reply_width=8,
    opp_reply_model="selfish",        # or "adversarial"

    # Optional rollouts (off by default)
    use_mc_rollouts=False,
)

# Later, introspect:
print(engine.diagnostics())
```

---

## 7) Where to go next (if you want dominance, not just parity)

If you want the “unfair” advantage suggested in the roadmap, the next steps are:
- Progressive widening MCTS for strategic phases
- learned value function (self-play) to replace hand-tuned weights
- explicit opponent archetype modeling (trade behavior + robber behavior)

But the upgrades in this file are the highest ROI steps that are *immediately integrable*.

