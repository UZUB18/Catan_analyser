# three_engines Future Upgrade Guide: Using AB Supporting Files

## Goal
Use existing AB support modules to make `catanatron/players/three_engines.py` stronger without turning it into a copy of `AlphaBetaPlayer`.

## Supporting files to reuse

### 1) `catanatron/players/value.py`
Use this for shared evaluation logic:
- `get_value_fn(name, params, value_function=None)`
- `base_fn(...)` / `contender_fn(...)`
- `DEFAULT_WEIGHTS` and related feature composition

Why reuse:
- Keeps one evaluation baseline across engines
- Makes GT/STAT/WILD comparisons more consistent
- Reduces duplicate scoring code

### 2) `catanatron/players/tree_search_utils.py`
Use this for search expansion/pruning primitives:
- `expand_spectrum(game, actions)`
- `execute_spectrum(game, action)`
- `list_prunned_actions(game)`

Why reuse:
- Gives probability-aware outcome expansion for non-deterministic actions
- Reuses existing pruning heuristics
- Avoids re-implementing action-outcome logic

## Current status (important)
`three_engines.py` currently defines independent heuristic players:
- `GameTheoryEngine`
- `StatsEngine`
- `WildSheepCultEngine`

It does **not** currently depend on `value.py` or `tree_search_utils.py`.

## Recommended integration pattern (cleanest)

### Hybrid policy pattern
Keep each engine's identity, but add optional shared lookahead:

1. **Engine-specific candidate generation** (GT/STAT/WILD logic)
2. **Optional prune** with `list_prunned_actions(game)`
3. **One-step expectation** with `expand_spectrum(...)`
4. **Shared evaluation** with `get_value_fn(...)` (or custom wrapper)
5. **Final tie-breaker** remains engine-specific

This preserves style differences while gaining AB-grade infrastructure.

## Suggested architecture changes

### A) Add internal helpers in `three_engines.py`
- `_candidate_actions(game, playable_actions, mode)`
- `_expected_value(game, action, eval_fn)` using `expand_spectrum`
- `_shared_eval(game, color, profile)` that wraps `get_value_fn`

### B) Keep search depth shallow by default
For runtime safety, start with:
- depth 0: current heuristic behavior
- depth 1: expected-value over stochastic outcomes (recommended default for upgrades)
- depth 2+: opt-in only for benchmarked configurations

### C) Add feature flags per engine
Example knobs:
- `use_shared_eval: bool`
- `use_pruning: bool`
- `lookahead_depth: int`
- `value_profile: "base_fn" | "contender_fn" | "custom"`

## Practical upgrade roadmap

### Phase 1 (low risk)
- Import `get_value_fn` and `expand_spectrum`
- Evaluate top-K engine candidates with one-step expected value
- Keep original action as fallback

### Phase 2 (medium risk)
- Add optional `list_prunned_actions(game)` pre-filter
- Add cached eval (`state_key -> value`) to reduce repeated scoring

### Phase 3 (higher impact)
- Add iterative deepening for late-game only
- Add transposition cache for repeated states
- Keep engine personality via custom tie-break rules

## Guardrails
- Do not replace GT/STAT/WILD heuristics wholesale with AB.
- Do not force deep search on every turn (runtime spikes).
- Keep deterministic fallback when helper calls fail.
- Add tests for parity: old mode vs upgraded mode should both run.

## Minimal test checklist
- Existing `three_engines` players still initialize from CLI
- `--players=GT,STAT,WILD,R` runs without crashes
- Upgraded mode beats baseline AB in controlled self-play sample before defaulting on
- Runtime budget remains acceptable under `--num` batch runs

## Bottom line
For future upgrades, prefer **reuse + extension**:
- Reuse `value.py` for consistent evaluation
- Reuse `tree_search_utils.py` for probability/pruning mechanics
- Keep `three_engines.py` as a distinct hybrid policy layer
