# GTv3/GTv4 "ab_killer" — Upgrade Roadmap (vs AlphaBeta)

This file tracks the high-ROI upgrades for **`GTv4:profile=ab_killer`** (and historically `GTv3:profile=ab_killer`).

Goal: increase win-rate vs `AB:2:true` while keeping runtime reasonable.

---

## Upgrade candidates (ordered)

### 1) Discard decisions via value function (NOT coarse `_utility`)
- **Problem:** DISCARD scoring used a fallback utility, not the engine’s real evaluator.
- **Plan:** for each DISCARD action, simulate via `game.copy().execute(action)` and score with `value_fn(game2, my_color)`.
- **Status:** implemented (2026-02-16).

### 2) Robber heuristic: coordinate-aware leader blocking
- **Problem:** `robber_use_ev=false` was strong vs AB fields, but the heuristic mostly ignored *which tile* gets blocked.
- **Plan:** when choosing MOVE_ROBBER, add expected leader production blocked on that coordinate (dice probability × adjacent leader settlements/cities), and penalize blocking our own best tiles.
- **Status:** implemented (2026-02-16).

### 3) Dynamic AB-delegate depth
- **Problem:** depth=2 is great as a default, but endgame / low-branching spots can justify depth=3.
- **Plan:** keep `ab_tactical_depth=2` normally, but temporarily switch to `3` when:
  - `leader_vp >= 8` or `my_vp >= 8`, or
  - branching is small (cheap), or
  - hard conversion sequences are present.
- **Status:** implemented in GTv4 (2026-02-17).

### 4) AB-action "sanity veto" layer (multiplayer blunder filter)
- **Problem:** AB can be tactically clean but strategically wrong in 4P (e.g., feeding the leader with a trade).
- **Plan:** veto a small set of high-confidence blunders (trade feed / kingmaking), then fall back to GT scoring / next-best shortlist.
- **Status:** implemented in GTv4 (2026-02-17).

### 5) Instrumentation-driven tuning loop
- **Problem:** we need to know *which* phases/action-types cause losses vs AB to target improvements.
- **Plan:** add lightweight counters/logging per game:
  - how often AB delegate fires (and at which depth),
  - robber/discard decisions,
  - action-type distribution,
  then run benchmark ladders and tune systematically.
- **Status:** implemented (engine-level counters + diagnostics) in GTv4 (2026-02-17).

