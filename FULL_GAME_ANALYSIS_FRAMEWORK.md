# Full-Game Catan Analysis Framework (Implementation Context)

This document describes the **actual implemented system** for `full_game` mode in this repository, with file map, data formats, formulas, state machine behavior, major algorithms, and important code snippets.

---

## 1) Stack, runtime, and architectural style

## Languages / frameworks
- **Language**: Python
- **UI framework**: Tkinter (`ttk`)
- **Data style**: `dataclass` + `Enum` + typed dict/list/set
- **Testing**: `unittest`
- **Execution model**: background worker thread + queue for progress/results

## Architecture style
- The full-game core uses a **rules engine + simulator** split:
  - `catan_analyzer/game/*` = strict rules/state transition layer.
  - `catan_analyzer/analysis/fullgame_rollout.py` = policy-driven rollout/evaluation layer.
- State transitions are effectively **immutable-style** from caller perspective:
  - `apply_action(state, action)` clones state and returns a new state.

---

## 2) File map and responsibilities

## Core full-game engine files (`catan_analyzer/game/`)

### `state.py`
Defines:
- `GamePhase` enum
- `DevCardType` enum
- `PlayerState` dataclass
- `GameState` dataclass
- helpers:
  - `initialize_game_state(...)`
  - `player_visible_victory_points(...)`
  - `player_total_victory_points(...)`
  - setup order, deck generation, next-player helpers

### `actions.py`
Defines:
- `GameAction(kind: str, data: dict[str, Any])`
- string constants for all actions
- typed action constructors (`build_city(...)`, `play_knight(...)`, etc.)

### `rules.py`
Defines:
- Economic constants: `ROAD_COST`, `SETTLEMENT_COST`, `CITY_COST`, `DEV_COST`
- `list_legal_actions(state)` (phase-gated generator)
- `apply_action(state, action)` (strict rules reducer)
- helper utilities for robber, production, trade ratios, legality checks

### `awards.py`
Defines:
- `longest_road_length_for_player(...)` via DFS on edges
- `recompute_longest_road(state)`
- `recompute_largest_army(state)`
- `recompute_awards(state)`

### `__init__.py`
Exports engine API constants and functions for external consumers.

---

## Analysis integration files

### `catan_analyzer/analysis/types.py`
Adds:
- `AnalysisMode.FULL_GAME = "full_game"`
- full-game config fields:
  - `full_game_rollouts`
  - `full_game_max_turns`
  - `full_game_candidate_vertices`
  - `full_game_trade_offer_limit`
- `FullGameSummary` dataclass
- `AnalysisResult.full_game_summary`

### `catan_analyzer/analysis/fullgame_rollout.py`
Implements:
- `FullGameAnalyzer.analyze(...)`
- rollout initialization, simulation loop, policy decisions, scoring formulas
- explain lines for UI (`FG#...`)

### `catan_analyzer/analysis/simulation.py`
Integration:
- `create_analyzer(...)` returns `FullGameAnalyzer` for full-game mode
- `_validate_config(...)` validates full-game config bounds

### `catan_analyzer/analysis/seeding.py`
Deterministic seeding helper used for reproducibility.

---

## UI / app integration files

### `catan_analyzer/ui/mode_descriptions.py`
- Adds mode copy for `full_game`.

### `catan_analyzer/ui/panels.py`
- `build_config()` maps existing tuning controls into full-game config values.
- Results panel appends `FullGameSummary` details (win rates, ETAs, avg length).

### `catan_analyzer/app.py`
- Uses generic analyzer flow:
  - build config
  - create analyzer
  - run in thread
  - stream progress
  - render result

---

## Tests added

### `tests/test_game_rules.py`
Covers:
- setup second settlement starting resources
- roll-7 discard then robber move
- dev card same-turn-play restriction
- 2:1 port bank trade behavior
- longest-road path behavior with blocked vertices

### `tests/test_fullgame_analysis.py`
Covers:
- full-game returns ranking + summary
- full-game reproducible with fixed seed

---

## 3) Data model details (formats + semantics)

## Enums

```python
class GamePhase(str, Enum):
    SETUP_SETTLEMENT, SETUP_ROAD, TURN_START,
    ROBBER_DISCARD, ROBBER_MOVE, ROBBER_STEAL,
    TRADE, BUILD, DEV_PLAY, GAME_OVER
```

```python
class DevCardType(str, Enum):
    KNIGHT, VICTORY_POINT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY
```

## Development deck composition

```python
DEV_DECK_COMPOSITION = {
    KNIGHT: 14, VICTORY_POINT: 5,
    ROAD_BUILDING: 2, YEAR_OF_PLENTY: 2, MONOPOLY: 2
}
```

## Player state format

`PlayerState` fields:
- `hand: Dict[Resource, int]`
- `settlements: set[int]`
- `cities: set[int]`
- `roads: set[EdgeKey]`
- `ports: set[PortType]`
- `dev_cards: Dict[DevCardType, int]`
- `new_dev_cards: Dict[DevCardType, int]` (turn lock for dev play)
- `revealed_vp_cards: int`
- `played_knights: int`
- `played_non_vp_dev_this_turn: bool`

## Game state format

`GameState` fields include:
- board + players
- `phase`, `current_player_id`, `turn_number`
- `robber_tile_id`
- `bank` resource counts (finite bank)
- `dev_deck` stack
- setup cursor fields (`setup_order`, `setup_index`, `pending_setup_vertex_id`)
- robber workflow fields (`discard_queue`, `pending_steal_target_ids`, `dice_roll`)
- awards (`longest_road_owner/length`, `largest_army_owner/size`)
- terminal status (`winner_id`)
- `event_log`
- internal RNG object (`_rng`)

---

## 4) Action protocol

All actions are serialized as:

```python
GameAction(kind: str, data: dict[str, Any])
```

Canonical action kinds include:
- setup: `place_setup_settlement`, `place_setup_road`
- roll/robber: `roll_dice`, `discard_resources`, `move_robber`, `steal_resource`, `skip_steal`
- trade: `trade_bank`, `trade_player`, `end_trade_phase`
- build: `build_road`, `build_settlement`, `build_city`, `buy_dev_card`, `end_build_phase`
- dev/turn: `play_knight`, `play_road_building`, `play_year_of_plenty`, `play_monopoly`, `reveal_vp`, `end_turn`

---

## 5) Rules engine: full state machine behavior

The engine is phase-gated:
- `list_legal_actions(state)` emits only actions legal in current phase.
- `apply_action(state, action)` validates phase + legality + cost + resource/deck constraints.

## Setup
- Setup order: forward then reverse snake.
- In `SETUP_SETTLEMENT`: legal settlement vertices via board distance-rule legality.
- In `SETUP_ROAD`: road must exist, be free, and touch the just-placed settlement vertex.
- On each player’s **second** setup settlement:
  - gain 1 resource from each adjacent non-desert tile, subject to bank availability.

## Turn start and dice
- In `TURN_START`, only `roll_dice`.
- If roll = 7:
  - move to `ROBBER_DISCARD` if any player has >7 cards else `ROBBER_MOVE`.
- Else:
  - production executed, then `TRADE`.

## Production (implemented model)
- For each settlement: +1 on matching token.
- For each city: +2 on matching token.
- Robber tile blocks production on that tile.
- Finite bank enforced.
- If total demand for a resource exceeds bank supply, that resource payout is globally blocked for that roll.

## Robber flow
- `ROBBER_DISCARD`:
  - discard player queue ordered by current turn order.
  - each player discards exactly `floor(hand/2)`.
- `ROBBER_MOVE`:
  - robber must move to a different valid tile id.
- `ROBBER_STEAL`:
  - eligible targets = opponents with cards + building adjacent to robber tile.
  - steal 1 random card (or explicit resource if valid).
  - then transition to `TRADE`.

## Trade
- `trade_bank`:
  - ratio = 2 (specific port) else 3 (3:1 port) else 4.
  - exact ratio must be paid; bank must have requested output card.
- `trade_player`:
  - explicit `give` and `receive` maps must both be non-empty.
  - both players must actually have required resources.

## Build
- Road:
  - pay road cost,
  - edge must be legal and unoccupied,
  - connectivity check: touches own building OR own road network endpoint (with opponent-vertex blocking rules considered by path usage).
- Settlement:
  - pay settlement cost,
  - board settlement legality + distance rule,
  - must connect to at least one owned road.
- City:
  - pay city cost,
  - only upgrade own settlement.
- Dev card:
  - pay dev cost,
  - dev deck must not be empty,
  - drawn card placed in `dev_cards` + `new_dev_cards`.

## Dev play
- Non-VP devs:
  - blocked if played a non-VP dev already this turn.
  - blocked if only copies were bought this turn (`new_dev_cards`).
- Knight:
  - move robber + optional steal + `played_knights += 1`.
- Road Building:
  - places first edge (required legal), second edge optional if legal.
- Year of Plenty:
  - takes two bank resources if both available.
- Monopoly:
  - takes all chosen resource from all opponents.
- Reveal VP:
  - converts hidden VP visibility, checks winner.

## End turn
- Reset:
  - `played_non_vp_dev_this_turn = False`
  - zero out `new_dev_cards`
- Advance player, increment turn number, phase -> `TURN_START`.

## Winner check
- Called after major scoring-impacting actions.
- Uses:
  - settlements, cities, revealed/hidden VP cards, longest road bonus, largest army bonus.
- If `>=10`, phase set to `GAME_OVER`.

---

## 6) Awards subsystem details

## Largest Army
- tracked by `played_knights`.
- minimum threshold 3.
- transfer only on strict exceed; ties don’t transfer.

## Longest Road algorithm

Core function:
`longest_road_length_for_player(board, player_roads, blocked_vertices)`.

Method:
1. Build incident-edge adjacency from player road edges.
2. DFS from each vertex.
3. Track used edges (simple path in edge-space).
4. If traversal arrives via edge at a blocked vertex (opponent settlement/city), chain stops.
5. Return max edge count.

Awarding logic:
- must be at least 5.
- transfer only on strict exceed over current holder.
- ties do not steal.

---

## 7) FullGameAnalyzer exact rollout system

## Candidate selection
1. Build baseline ranking using static scorer `score_vertex`.
2. Take top `K = full_game_candidate_vertices`.
3. For each candidate vertex:
   - force P1 first setup settlement there,
   - run `R = full_game_rollouts` full simulations.

## Deterministic seeding
- Base seed: `analysis_seed(board, mc_seed, salt="full_game_rollout")`.
- Per simulation seed:
  `derive_seed(base_seed, "candidate", vertex_id, rollout_index)`.

This makes full-game mode reproducible under fixed inputs.

## Setup policies used in simulation
- Forced first P1 settlement at candidate vertex.
- Best setup road = road whose non-anchor endpoint has max `score_vertex`.
- Remaining setup settlement/road choices use greedy score-based heuristics.

## Main simulation loop
Per step, branch by `state.phase`:
- `TURN_START` -> roll dice
- `ROBBER_DISCARD` -> first legal discard action
- `ROBBER_MOVE` -> heuristic robber tile selection
- `ROBBER_STEAL` -> target with largest hand
- `TRADE` -> utility-filtered best trades up to offer limit
- `BUILD` -> prioritized build policy with action budget
- `DEV_PLAY` -> prioritized dev policy then end turn

Terminated by:
- winner found (`GAME_OVER`) OR
- turn cap `full_game_max_turns` (with internal floor 20 / external call floor 40)

## Per-outcome metrics tracked
- winner id
- winner turn
- turns_to_victory per player
- robber loss to focal player proxy

---

## 8) Policy formulas and heuristics

## 8.1 Candidate scoring formula
For simulated candidates:

\[
\text{full\_game\_boost} = 8.0 \cdot \text{win\_rate} - \frac{\text{avg\_win\_turn}}{60.0}
\]

\[
\text{new\_total} = 0.45 \cdot \text{baseline\_total} + \text{full\_game\_boost}
\]

Also writes:
- `tempo_score = 100 / avg_win_turn`
- `recipe_coverage_score = win_rate * 5`
- `robber_penalty = avg_robber_loss`

## 8.2 Robber placement heuristic
For each candidate robber tile:
- Add opponent settlement pressure: `+1.00 * pips`
- Add opponent city pressure: `+1.40 * pips`
- Penalize self settlement pressure: `-0.45 * pips`
- Penalize self city pressure: `-0.85 * pips`
- Bonus `+2.0` if steal targets exist.

Choose max score tile.

## 8.3 Trade utility model
Trade actions considered:
- bank trades from legal action generator
- 1-for-1 player trades from legal action generator

Utility basis: `_build_access_score(hand)`:

\[
3.0/(1+\text{missing(settlement)}) +
3.2/(1+\text{missing(city)}) +
2.2/(1+\text{missing(dev)}) +
1.0/(1+\text{missing(road)})
\]

For player trade:
- reject if target is far-ahead and gains,
- reject if own delta <= 0,
- reject if target hurt too much (< -0.35 proxy),
- choose max:
\[
 \text{utility} = \Delta_{\text{ours}} - 0.45 \cdot \max(0, \Delta_{\text{theirs}})
\]

## 8.4 Build policy priority
At each build step (budget 8):
1. best settlement action (highest `score_vertex`)
2. best city action (highest pip-yield vertex)
3. buy dev
4. best road expansion (endpoint yield proxy)
Then end build phase.

## 8.5 Dev policy priority
1. reveal VP if `visible + hidden >= 10`
2. Knight (best robber pressure)
3. Road Building
4. Year of Plenty toward best missing-cost target
5. Monopoly if expected collection >= 3
Else no dev play, then end turn.

---

## 9) Output objects produced by full-game mode

`AnalysisResult`:
- `global_ranking: list[VertexScore]`
- `predicted_sequence: list[DraftPick]`
- `top_recommendations: list[VertexScore]`
- `explain_lines: list[MctsLineExplanation]`
- `full_game_summary: FullGameSummary`

`FullGameSummary`:
- `rollout_count`
- `player_win_rates: dict[player_id, probability]`
- `expected_turns_to_victory: dict[player_id, avg_turns]`
- `average_game_length_turns`

Explain lines format example:
- `FG#1 V23 (sim)` etc.

---

## 10) UI and runtime integration path

1. User selects `mode = full_game`.
2. UI derives full-game config from tuning controls.
3. `app.py` launches worker thread.
4. Worker instantiates analyzer via `create_analyzer`.
5. Analyzer emits progress via `AnalysisRuntime`.
6. Main thread polls queue, updates progress bar + ETA.
7. On success, results panel displays ranking + full-game summary details.

---

## 11) Config validation constraints

In `analysis/simulation.py::_validate_config`:
- `full_game_rollouts >= 1`
- `full_game_max_turns >= 20`
- `full_game_candidate_vertices >= 1`
- `full_game_trade_offer_limit >= 0`

---

## 12) Important snippets

## Analyzer factory wiring

```python
def create_analyzer(mode):
    normalized = AnalysisMode(mode)
    if normalized is AnalysisMode.FULL_GAME:
        return FullGameAnalyzer()
    ...
```

## Deterministic per-rollout seed

```python
base_seed = analysis_seed(board, config.mc_seed, salt="full_game_rollout")
rollout_seed = derive_seed(base_seed, "candidate", vertex_id, rollout_index)
```

## Bank trade ratio logic

```python
if specific_port in player.ports: return 2
if PortType.ANY_3TO1 in player.ports: return 3
return 4
```

## Winner check

```python
if player_total_victory_points(state, current_player_id) >= 10:
    state.winner_id = current_player_id
    state.phase = GamePhase.GAME_OVER
```

## Longest-road DFS edge-simple path

```python
def dfs(v, used_edges, arrived_via_edge):
    if arrived_via_edge and v in blocked_vertices:
        return 0
    best = 0
    for edge in incident_edges[v]:
        if edge in used_edges: continue
        used_edges.add(edge)
        best = max(best, 1 + dfs(other(edge, v), used_edges, True))
        used_edges.remove(edge)
    return best
```

---

## 13) Current behavior notes (important for handoff)

1. This is a **policy-driven full-game simulator**, not exhaustive game-tree search.
2. Hidden information is not solved with ISMCTS/belief tracking yet.
3. Player trade generation is intentionally simple (primarily 1-for-1 offers).
4. Production under bank scarcity uses all-or-nothing per resource type per roll (no partial allocation).
5. Full-game scoring is blended with baseline static score for ranking stability.

---

## 14) Style conventions in this subsystem

- Prefer typed dataclasses and enums.
- Keep domain rules in `game/*`, keep analysis heuristics in `analysis/*`.
- `list_legal_actions` centralizes legality surface.
- `apply_action` is single source of state transition truth.
- Tests focus on rule fidelity + deterministic reproducibility.

---

## 15) Minimal execution trace (example)

Given `mode=full_game`:
1. App builds config.
2. Analyzer computes baseline ranking.
3. Picks top-K candidates.
4. For each candidate:
   - initialize strict setup state with forced first pick.
   - complete setup via heuristics.
   - simulate turn loop with rules engine until win/cap.
5. Aggregate win rates and turn metrics.
6. Re-rank vertices with full-game formula.
7. Build draft prediction.
8. Emit `AnalysisResult` + `FullGameSummary`.
9. UI renders ranking + explain lines + summary stats.

---

## 16) Test coverage summary for this feature

- Rule path tests:
  - setup resource grant
  - roll-7 discard path
  - dev timing lockout
  - port-based bank ratio
  - longest-road blocking behavior
- Analyzer tests:
  - full-game returns summary/ranking
  - deterministic reproducibility with fixed seed

---

If you hand this document + the referenced files to another engineer, they can follow the entire currently implemented full-game analysis pipeline end-to-end (state model, legal actions, reducer, rollout policy loop, scoring, and UI integration).
