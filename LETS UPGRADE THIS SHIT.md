# LETS UPGRADE THIS SHIT — GameTheoryEngine (GT) Upgrade Roadmap

> Scope: This document is based on the current codebase in `catanatron/` and the GT implementation in `catanatron/players/game_theory_engine.py` (v1.0.0).
>
> Goal: **Improve GT strength vs AB/SAB without breaking the simulator**. Every proposal below is written to be incremental, guarded, and testable.

---

## Snapshot: where we are right now

- **Engine:** `GameTheoryEngine` (`GT`)
- **File:** `catanatron/players/game_theory_engine.py`
- **Version:** `1.0.0`
- **Core dependencies used:**
  - `catanatron/players/value.py` (`get_value_fn`, `DEFAULT_WEIGHTS`)
  - `catanatron/players/tree_search_utils.py` (`expand_spectrum`, `list_prunned_actions`)

### Current benchmark result (example you posted)
GT vs baseline `AB:2:true` (1000 games total, seat-rotated):
- **GT win rate:** 18.0%
- **Baseline per-player win rate:** 27.33%
- **Advantage:** -9.33%

Interpretation: GT is losing clearly; we need to improve **openings**, **turn tactics**, and **evaluation consistency**.

---

## How GT works today (v1.0.0)

### Decision flow (high level)
In `GameTheoryEngine.decide(game, playable_actions)`:

1. **Immediate win check** via `_can_win_this_turn` (simulate each action, pick any that hits 10 VP).
2. **Shortcut** if only 1 legal action.
3. **Phase handlers**
   - Initial placement: `_pick_initial`
   - Discard: `_pick_best_by_simulated_utility`
   - Robber: `_pick_robber_move`
4. **Leader detection** using `_leader_color_and_vp` and `threat_mode = leader_vp >= 8`.
5. **Build `value_fn`** (`base_fn` / `contender_fn`).
6. **Candidate reduction**
   - optional: `list_prunned_actions(game)`
   - cap: `max_candidates` (default 14) using `_base_bias` and trade stance as a cheap pre-score
7. **1-step expected-value scoring** per candidate action
   - attempts: `expand_spectrum(game, candidates)`
   - computes an expected post-action value and utilities
8. **Fallback** if spectrum fails: `_simulate_if_possible` with `State.copy + apply_action`.
9. Adds a tiny random noise for tie-breaking.

### Scoring terms used today
For each action:

- **Priors**: `_base_bias(action_type, threat_mode)`
- **Trade stance**: `_trade_stance_score` for OFFER/ACCEPT/MARITIME
- **Lookahead**:
  - `eval_gain = E[value_fn(after)] - value_fn(before)`
  - `my_gain = E[_utility(after)] - _utility(before)`
  - `leader_gain = E[_utility(leader_after)] - _utility(leader_before)`
  - `score += lookahead_weight * eval_gain + my_gain - lam * leader_gain`

This design is sensible, but it has two big issues:

1. **Mixed evaluation systems** (`value_fn` vs `_utility`). These can disagree, causing GT to pick actions that “look good” in one metric but are bad in the other.
2. **GT is mostly 1-step**, while AB is multi-ply (depth=2+) and will exploit tactical sequences.

---

## Key weaknesses to address (from code + gameplay realities)

### A) `_utility()` is too coarse (and mismatched)
Current `_utility` only uses:
- VP, number of settlements/cities/roads/dev, hand size, robber penalty

It ignores:
- production quality (dice probabilities)
- ports and conversion
- expansion reachability
- blocking value (taking a node that kills opponent plans)
- hand synergy (city/settle readiness)

But `value_fn` *does* include several of these. Mixing them is a recipe for inconsistent choices.

### B) Opening logic is myopic
`_pick_initial` scores a settlement by:
- `sum(inflow) + diversity - variance`

This is a good baseline, but AB/SAB gains a lot from:
- 2-settlement synergy
- ports, ore/wheat priority, road reachability

In Catan, opening quality is a huge portion of win-rate.

### C) Denial model is incomplete
- “Leader” is defined by current VP only.
- Denial uses `_utility(leader)` which doesn’t reflect “threat” very well.

In 4-player, you want:
- threat = VP + near-term conversion + production + dev threats
- denial pressure that ramps up smoothly

### D) Candidate pruning can hide tactics
- `max_candidates=14` is fine, but the ranking is based on `_base_bias` and trade stance.
- This can drop a high-value tactical play that `value_fn` would love.

### E) Performance wastes
- repeated calls to `_utility` and `value_fn` on many near-identical states
- no per-decision memoization

---

## Upgrade principles (non-breaking)

Every upgrade below follows these rules:

1. **Feature flags + defaults**: new behavior behind `kwargs` (ex: `use_opening_search=True`).
2. **Fallbacks stay**: if `expand_spectrum` or simulation fails, fall back to existing logic.
3. **Small deltas**: implement one upgrade at a time, bump version, benchmark, then proceed.
4. **No protocol breaks**: `decide(game, playable_actions)` must always return a valid action.

Recommended sanity tests after every change:
- `pytest -q`
- `catanatron-play --players GT,R,R,R --num 5 --quiet`
- quick benchmark: `catanatron-benchmark --candidates GT --baseline AB:2:true --games-per-seat 25 --parallel --workers 6 --label quick`.

---

# Stage 1 — MOST IMPORTANT (highest ROI, lowest risk)

## 1) Unify evaluation: stop mixing `value_fn` and `_utility` for gains

### Problem
Right now GT uses:
- `eval_gain` from `value_fn`
- `my_gain` + `leader_gain` from `_utility`

This means GT can “win” in one scoring space while losing in the other.

### Upgrade
Make gains consistently use **one** evaluation model:

- Use `value_fn` for **my gain**.
- Use a **threat model** for opponents that’s compatible with 4-player.

Concretely:

1) Define:
- `my_eval(game) = value_fn(game, self.color)`

2) Define opponent threat as **max opponent eval** (simple and effective):
- `opp_eval(game) = max(value_fn(game, c) for c in enemies)`

3) Score an action using expected outcomes:
- `Δmy = E[my_eval(after)] - my_eval(before)`
- `Δopp = E[opp_eval(after)] - opp_eval(before)`
- `score += Δmy - λ * Δopp`

Where `λ` ramps with threat:
- early: λ ~ 0.2–0.4
- late / leader near win: λ ~ 0.8–1.2

### Why this is safe
- You already have `value_fn` and fallbacks.
- If `value_fn` errors, you can fall back to `_utility` just like today.

### Implementation notes
- Add config flags:
  - `use_value_fn_gains=True`
  - `use_multi_opp_threat=True`
  - `threat_lambda_base=0.35`
  - `threat_lambda_late=0.95`

- Only evaluate opponents for the **top-K candidates** (after candidate filtering), not for every playable action.

### Validation
Benchmark before/after:
- GT vs AB:2:true, 500 games/seat
- Ensure no runtime explosion (watch avg ticks/sec)

---

## 2) Opening upgrade: evaluate initial actions by simulation + `value_fn`

### Problem
`_pick_initial` uses a hand-rolled production heuristic that is not aligned with `value_fn` or AB’s priorities.

### Upgrade
During initial placement, choose actions by **simulating them and scoring with `value_fn`**, not by a separate inflow heuristic.

Recommended minimal safe version:

- If action is `BUILD_INITIAL_SETTLEMENT`:
  1) For each settlement action `a`:
     - simulate `game_copy.execute(a)`
     - score `value_fn(game_copy, self.color)`
  2) pick argmax

- If action is `BUILD_INITIAL_ROAD`:
  1) For each road action `a`:
     - simulate
     - score value
  2) pick argmax

This is a strict improvement because it uses the same evaluator AB uses.

### Why this is safe
- Initial placement is deterministic.
- Uses existing, well-tested `game.copy().execute()` and `value_fn`.
- If simulation fails in some edge version, fall back to current inflow-based heuristic.

### Implementation notes
- Add flag: `use_value_opening=True` (default True)
- Keep the old heuristic as fallback and as a debug comparison.

### Extra (still safe): port awareness
If board exposes port info, add a small bonus for ports that match your opening resources.

---

## 3) Robber upgrade: choose robber move by expected value, not target VP

### Problem
Current `_pick_robber_move`:
- mostly targets leader
- uses VP + hand size
- does not measure tile/production impact

### Upgrade
For each MOVE_ROBBER action, compute expected outcome using `expand_spectrum`:

- `E[my_eval(after)] - λ * E[opp_eval(after)]`

Pick the best. This automatically:
- blocks high-value tiles
- favors stealing outcomes that help you
- targets the real threat

### Why this is safe
- MOVE_ROBBER is already supported by `expand_spectrum`.
- If `expand_spectrum` fails, keep the old VP/hand-based logic.

---

## 4) Candidate selection upgrade: rank candidates using quick simulated value, not `_base_bias`

### Problem
When there are many playable actions, GT uses `_base_bias` to select top actions.
That can drop tactics like:
- dev card plays
- weird trades
- road placements that open a settlement

### Upgrade
Replace the “quick score” with a *real* quick value estimate:

- For each action `a`:
  - if deterministic: simulate once, compute `Δmy_eval`
  - if stochastic (ROLL, BUY_DEV, MOVE_ROBBER): use a **cheap expectation**
    - either `expand_spectrum` limited to that action
    - or keep current behavior but never exclude these types

Then keep top `max_candidates` by `Δmy_eval` (with guardrails to always include required-phase actions).

### Why this is safe
- It only affects the shortlist; final selection still uses full scoring.
- If quick sim fails, fall back to `_base_bias`.

---

## 5) Add per-decision memoization (speed win, strength win)

### Problem
GT recomputes evaluation many times for the same `game_copy` states.

### Upgrade
Within one `decide(...)` call, add caches:
- `eval_cache[(color, state_fingerprint)] -> float`
- `utility_cache` if you still use `_utility`

Fingerprint options (choose the safest):
- **minimal**: `id(game.state)` won’t work for copies
- **better**: `len(state.action_records), state.current_player_index, tuple(state.player_state.items())` (cheap-ish)

Keep cache local to `decide` so it cannot go stale.

### Why this is safe
- Read-only; doesn’t change logic.
- Easy to disable if it misbehaves.

---

# Stage 2 — GOOD TO HAVE (medium effort, big strength upside)

## 1) Add within-turn search (GT becomes “SAB + denial + value_fn”)

### Problem
AB/SAB wins because it sees tactical sequences (especially within a single turn).
GT is 1-step.

### Upgrade
Implement a **within-turn beam search**:
- Terminal when action is `END_TURN` OR depth limit reached.
- Only simulate **our turn** (like SAB), which avoids multiplayer minimax weirdness.
- Use `expand_spectrum` for chance actions.

Suggested params:
- `turn_search_depth=3`
- `beam_width=8`
- `time_budget_ms=200` (per decision)

Use `value_fn` at leaves.

### Non-breaking strategy
- feature flag: `use_turn_search=False` by default at first
- then enable once stable + faster with caching

### Why it should help
- Catan strength is in “combo turns”: trade → build → buy dev → play dev, etc.
- SAB shows this advantage clearly.

---

## 2) Multi-opponent threat model (not just “leader by VP”)

### Problem
“Leader by VP” misses:
- a player with strong production but fewer current VP
- dev-card threats (army/VP cards)

### Upgrade
Define `threat_score(color)` using a weighted feature set:
- actual VP
- estimated production (use `_estimate_player_inflow` or `build_production_features`)
- dev cards in hand (if visible)
- longest road potential (if accessible)

Then:
- `opp_eval = max(threat_score(enemy))`
- choose denial actions that reduce the top threat, not necessarily current VP leader

### Safety
- threat_score is additive and doesn’t affect legality.
- keep old VP leader logic as fallback.

---

## 3) Trade policy upgrade (acceptance threshold + “don’t feed the table”)

### Problem
Trade actions are high-leverage and hard. Current logic is heuristic-only.

### Upgrade
For `ACCEPT_TRADE` and `OFFER_TRADE`:
- simulate trade action
- compute `Δmy_eval` and `Δopp_eval`
- accept if `Δmy_eval - λ*Δopp_eval > threshold`

Threshold examples:
- early: `> 0`
- late: `> +small_margin`

This turns trade decisions into the same consistent evaluation framework.

---

# Stage 3 — MAYBE GOOD (higher risk / longer-term)

## 1) Auto-tune GT parameters with your benchmark harness

Tune:
- `deny_weight`
- `lookahead_weight`
- base biases
- candidate caps

Method:
- grid search or random search
- optimize “advantage vs baseline per-player”

Why “maybe”:
- easy to overfit to AB opponents
- needs disciplined benchmark splits (train/test seeds)

---

## 2) Add opponent-style adaptation

If opponents are AB/SAB (tactical), you may:
- increase within-turn search depth
- prioritize hand safety and tempo

If opponents are random/greedy:
- prioritize pure growth (lower denial)

Why “maybe”:
- needs reliable opponent identification
- adds code complexity

---

## 3) Hybrid evaluation: value_fn + domain-specific “Catan truths”

Sometimes pure value_fn misses:
- blocking a key intersection
- racing longest road
- bank/port conversion value

Add small, well-tested adjustments to value_fn:
- explicit blocking bonus
- port conversion bonus
- “tempo” bonus (getting to the next build faster)

Why “maybe”:
- easy to add bias that hurts generality

---

# Suggested version plan

- **v1.0.1**: Stage 1.1 (unify eval gains) + caching
- **v1.1.0**: Stage 1.2 (opening via simulation + value_fn) + robber EV scoring
- **v1.2.0**: Stage 1.3 (candidate selection via quick sim)
- **v1.3.0**: Stage 2.1 (within-turn beam search, off by default)

After each version:
- run quick benchmark (25 games/seat)
- then full benchmark (500+ games/seat)
- record results in `GT_ENGINE_UPGRADE_LOG.md`

---

# Minimal “don’t break anything” checklist

For every upgrade PR:

1. `pytest -q` passes.
2. `catanatron-play --players GT,R,R,R --num 5 --quiet` completes.
3. `catanatron-benchmark --candidates GT --baseline AB:2:true --games-per-seat 25 --parallel --workers 6 --label smoke` completes.
4. No new crashes in:
   - initial placement
   - discard phase
   - robber move
   - trade resolve phases

---

## Next step (recommended)
Start with Stage 1.1 (unify evaluation gains) because it’s the highest leverage and lowest risk.

If you want, I can implement Stage 1.1 as a backwards-compatible patch (feature-flagged) and run:
- quick_100 benchmark
- full 500/seat benchmark

