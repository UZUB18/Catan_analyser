# GT Engine Upgrade Log

## Engine
- **Name:** `GameTheoryEngine` (`GT` CLI code)
- **Current module:** `catanatron/players/game_theory_engine.py`
- **Version:** **v1.0.0**
- **Date:** 2026-02-14

---

## Why this upgrade
You asked to:
1. Extract GT from `three_engines.py` into a standalone file.
2. Upgrade GT using supporting files:
   - `catanatron/players/value.py`
   - `catanatron/players/tree_search_utils.py`
3. Remove old GT copy from `three_engines.py`.
4. Validate and test vs AB.

---

## What changed

### 1) Extraction to standalone file
- **Added:** `catanatron/players/game_theory_engine.py`
- **Moved:** `GameTheoryEngine` class out of `catanatron/players/three_engines.py`
- **Kept GT name:** class remains `GameTheoryEngine` and CLI code remains `GT`

### 2) Integration with supporting files
The new GT now uses:

#### `value.py`
- `get_value_fn(...)`
- `DEFAULT_WEIGHTS`

Usage in GT:
- Shared evaluation baseline for action scoring (`base_fn`/`contender_fn` support)
- Expected value gain term is now part of decision score

#### `tree_search_utils.py`
- `list_prunned_actions(game)`
- `expand_spectrum(game, actions)`

Usage in GT:
- Optional pruning to reduce low-value actions
- Probability-weighted action outcome scoring for stochastic actions

### 3) Wiring updates
- **Updated imports:** `catanatron/cli/cli_players.py`
  - `GameTheoryEngine` now imported from `catanatron.players.game_theory_engine`
  - `StatsEngine` and `WildSheepCultEngine` still imported from `three_engines.py`

### 4) Old GT removed from three_engines
- `GameTheoryEngine` class deleted from `catanatron/players/three_engines.py`
- File header/docs updated to reflect only STAT + WILD inside that file

---

## Testing and validation run

### Compile/syntax checks
- `python -m py_compile catanatron/players/game_theory_engine.py catanatron/players/three_engines.py catanatron/cli/cli_players.py`
- Result: **PASS**

### Test suite
- `pytest -q`
- Result: **63 passed**

### GT registration check
- `parse_cli_string('GT,STAT,WILD,R')`
- Result: **PASS** (all players instantiate correctly)

### GT vs AB simulations
1. `GT` vs `AB:1` x3 (10 games total, 4 players: `GT,AB:1,AB:1,AB:1`)
   - Result snapshot: `{'Color.BLUE': 4, 'Color.ORANGE': 4, 'Color.RED': 1, 'Color.WHITE': 1}`
2. `GT` vs `AB:2` (1 game, 4 players: `GT,AB:2,AB:2,AB:2`)
   - Result snapshot: `{'Color.ORANGE': 1}`

> Note: AB-depth multiplayer runs can be slow; treat these as smoke/perf sanity checks, not statistically significant strength claims.

---

## Versioning policy for future GT upgrades
Use semantic-ish engine versions:
- **MAJOR** (`v2.0.0`): strategy architecture changes (e.g., multi-ply search redesign)
- **MINOR** (`v1.1.0`): meaningful strength/runtime improvements, new features
- **PATCH** (`v1.0.1`): bugfixes/refactors with no intended strategy shift

For every version, log:
1. Design goal
2. Code deltas (files + key methods)
3. Benchmark setup (opponents, map, VP target, number of games)
4. Win-rate and runtime deltas vs previous version
5. Rollback note (how to revert quickly)

---

## Suggested next versions
- **v1.1.0:** Add transposition cache for repeated state evals
- **v1.2.0:** Add iterative deepening for late-game only
- **v1.3.0:** Add move ordering tuned for denial mode
- **v2.0.0:** Hybrid GT + AB selective deep search
