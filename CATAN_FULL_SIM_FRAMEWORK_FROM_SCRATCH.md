# CATAN Full-Game Simulation & Initial-Placement Analyzer — Rebuild Spec

This spec replaces the current policy-driven rollout framework with a **rule-faithful, high‑performance, multi‑agent simulation stack** that can (a) run millions of full games in parallel and (b) rank the **top 8 opening settlement locations** on a given board configuration for an **8‑player** game.

It is written to be implementable even if your repo structure changes: the modules are cleanly separable, and every “smart” component is pluggable.

---

## 0) Why the current framework can’t be “game‑theoretic” yet

Your current `full_game` mode is explicitly *policy-driven* (not game-tree search), uses phase-gated action generation, and blends rollout results back into static scoring. It also deliberately simplifies player-to-player trading and does not track hidden information with ISMCTS/belief models. These choices are reasonable for a first pass, but they hard-limit accuracy. (See your own “Current behavior notes.”)

Key fidelity gap: your production model blocks an entire resource type globally when supply is insufficient. The official rules have an exception when the shortage affects only one player: in that case the player receives as many cards as remain.

Key performance gap: the simulator runs in a single Python process and clones state on each action, which is catastrophic for rollout throughput.

---

## 1) Design goals

1. **Rules fidelity**: match CATAN base rules exactly; support optional variants (combined trade/build; extensions).
2. **Information correctness**: correctly represent *what players know* vs *what the simulator knows*.
3. **Strong play ≠ hard-coded heuristics**: support multiple agent strengths:
   - fast heuristics (baseline)
   - ISMCTS / draft-MCTS (setup and key turns)
   - optional RL policy/value nets (GPU accelerated)
4. **Scale**: embarrassingly parallel game rollouts across CPU cores; optional batched GPU inference.
5. **Reproducibility**: deterministic seeding per board + candidate + rollout index.
6. **Explainability**: always output “why these top 8” with resource/pip/port/expansion diagnostics plus statistical confidence.

---

## 2) Canonical CATAN base constants & data (encode as config)

### 2.1 Resources and costs
Resources: `BRICK, LUMBER, WOOL, GRAIN, ORE`

Building costs (base game):
- Road = 1 BRICK + 1 LUMBER
- Settlement = 1 BRICK + 1 LUMBER + 1 WOOL + 1 GRAIN
- City = 3 ORE + 2 GRAIN
- Development card = 1 WOOL + 1 GRAIN + 1 ORE

### 2.2 Dice distribution (2d6)
Let `ways(s)` be number of outcomes for sum `s`:
`ways = {2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1}`
Probability: `P(s) = ways(s) / 36`

### 2.3 Number token “pip” weight
Define `pips(s) = ways(s)` for `s != 7`, and `pips(7)=0`.
This matches the rulebook explanation that 6 and 8 have 5 pips because there are 5 ways to roll them.

### 2.4 Victory points
- Settlement = 1 VP
- City = 2 VP
- Longest Road = 2 VP, requires ≥5 roads, transfers only if strictly longer.
- Largest Army = 2 VP, requires ≥3 played knights, transfers only if strictly larger.
- VP development cards = 1 VP (hidden until revealed).

Win condition: first player to reach ≥10 VP **on their turn** wins immediately.

### 2.5 Development deck (base game)
Total dev cards = 25:
- 14 Knight
- 6 Progress (2 Road Building, 2 Year of Plenty, 2 Monopoly)
- 5 Victory Point

Rules:
- You may play **1** dev card per turn (knight or progress).
- You may play it **any time during your turn**, even before rolling.
- You may not play a dev card on the same turn you bought it **except**: if you buy a VP card and it brings you to 10, you may reveal and win immediately.

### 2.6 Trading
Domestic trade:
- Only the current player may trade; others may not trade among themselves during that turn.
- Multiple cards are allowed; giving away cards or trading “like for like” is disallowed.

Maritime trade:
- 4:1 always available.
- With a harbor you can trade 3:1 (generic) or 2:1 (special for one resource). Special harbors apply only to that resource and do **not** grant 3:1.

### 2.7 Robber
On a 7:
- No resource production.
- Any player with >7 resource cards discards half (round down).
- Roller moves the robber to any other tile (or desert).
- Roller steals 1 random resource from a chosen adjacent opponent (if any have cards).

Robber blocks production on the tile it occupies.

### 2.8 Bank scarcity rule (critical)
If a resource stack cannot satisfy all production owed:
- **Normally** nobody receives that resource that turn.
- **Exception**: if the shortage affects only one player, that player receives as many cards as remain; the rest are lost.

This matters a lot in 8-player games.

---

## 3) Engine architecture (from scratch)

### 3.1 Layering
**Do not** mix “rules” with “strategy.” Split into:

1. `core/board.py` — immutable board topology + tile/port metadata.
2. `core/state.py` — compact mutable state container.
3. `core/rules.py` — legality + transitions only (pure logic).
4. `core/encoding.py` — fixed integer action encoding (for speed + ML).
5. `ai/*` — agents (heuristic, MCTS, RL, hybrid).
6. `sim/*` — parallel rollout harness, metrics, ranking.
7. `ui/*` — visualization / app glue (optional).

### 3.2 Data-oriented state (SoA)
For rollout speed, ban Python `set`/`dict` in the hot loop. Use:
- `resources[player, 5] : uint8/uint16`
- `dev_hand[player, 5] : uint8` (counts per type)
- `dev_new[player, 5] : uint8` (lockout tracking)
- `played_knights[player] : uint8`
- `vp_revealed[player] : uint8`
- `settlement_owner[vertex] : int8` (−1 empty else player id)
- `city_owner[vertex] : int8`
- `road_owner[edge] : int8`
- `port_mask[player] : uint8` (bitmask of ports controlled)
- `robber_tile : uint8`
- `current_player : uint8`
- `turn : uint16`
- `flags` bitfield: `HAS_ROLLED`, `PLAYED_DEV_THIS_TURN`, etc.
- `bank[5] : int16`
- `dev_deck[] : uint8` (stack) + `dev_top : uint8`

Use `numpy` arrays and (ideally) **Numba** or a Rust/C++ extension for the `step()` loop.

### 3.3 Action encoding (int32)
Represent every possible action as a compact integer:
- `action = (kind_id << 24) | payload`
- payload encodes vertex/edge/resource types and counts in fixed bit ranges.

This removes Python object churn and enables batched evaluation.

### 3.4 Turn model (rule-faithful without fragile phases)
Your current engine is phase-gated (`TURN_START -> TRADE -> BUILD -> DEV_PLAY`). That is easy, but it fights two realities:
- Dev cards can be played before roll.
- Experienced rules allow interleaving trade/build arbitrarily.

Instead: maintain a **turn flags** model.
Legal actions are computed from flags:
- If `!HAS_ROLLED`: legal = {play_dev_if_allowed, roll_dice, reveal_vp_if_winning}
- If `HAS_ROLLED`: legal = {play_dev_if_allowed, domestic_trade, maritime_trade, build_actions, end_turn}
- Robber subflow is handled via a small “interrupt” state (`robber_mode` enum) that temporarily overrides legal actions.

This retains correctness but allows the combined trade/build behaviour naturally.

### 3.5 Hidden information model
True CATAN has hidden info:
- dev deck order
- opponents’ hands
- unrevealed VP cards

The simulator knows everything; *players* do not. You need:
- `PublicState` view (board, roads/settlements/cities, played knights, known trades, revealed VP, bank sizes optional)
- `PrivateState[player]` (their hand + dev cards)
- `BeliefState[player]` (distributions over others’ hands and dev deck composition)

For strong agents: use ISMCTS (determinization) or learned belief networks.

---

## 4) Correct rules algorithms (with edge cases)

### 4.1 Resource production with bank scarcity exception
Given a dice sum `s != 7`:
1. For each tile t where `number(t)=s` and `t != robber_tile`:
2. For each adjacent vertex v:
   - if settlement at v: owner gets +1 of tile resource
   - if city at v: owner gets +2
3. Aggregate owed amounts per player per resource: `owed[player, res]`
4. For each resource `res`:
   - `total = sum_p owed[p,res]`
   - if `total == 0`: continue
   - if `bank[res] >= total`: pay all
   - else:
     - let `affected = {p | owed[p,res] > 0}`
     - if `len(affected) == 1`:
         pay `min(bank[res], owed[p,res])` to that player
       else:
         pay none
   - decrement bank by paid amount

### 4.2 Longest road exact computation (fast enough)
Exact longest road is NP-ish in the worst case because cycles. You can still do it fast with:
- adjacency lists per vertex restricted to edges owned by player
- blocked vertices = vertices occupied by opponents’ settlements/cities (stop-through rule)
- DFS with edge-visited set (your current algorithm is correct conceptually)

Performance upgrade:
- compute longest road only when a road is built (not every action)
- memoize per (player, component_id) using incremental updates
- store player road graph as bitset; use iterative stack DFS with small fixed arrays

### 4.3 Domestic trade protocol (simulate realism without infinite bargaining)
A full bargaining game is endless. For simulation you need a *mechanism*.

Recommended mechanism:
- Current player can issue up to `K` offers (K configurable).
- Each offer is evaluated by each opponent independently.
- Opponents either accept or reject; no counteroffers in “fast” mode.
- In “realistic” mode: allow 1 counteroffer per opponent with concession schedule.

Make this pluggable: trade realism is the #1 source of strategic bias.

### 4.4 Development card effects
- Road Building: attempt to place 2 legal roads; if only 1 legal exists, place 1; if 0, do nothing.
- Year of Plenty: choose any two resources available in bank. If one choice is unavailable, agent must choose a different available resource (don’t auto-fail the card).
- Monopoly: collect all cards of chosen resource from opponents; bank unaffected.

---

## 5) Agent stack (from “fast” to “game-theoretic-ish”)

### 5.1 Baseline heuristic agent (for throughput)
Keep a fast, deterministic bot but make it **feature-driven** and consistent:

Core decision principle:
> Choose the action that maximizes expected (discounted) probability of reaching 10 VP, approximated by a value function.

Implement value via a cheap surrogate:
- `V(state, player) = w1*VP + w2*E[prod] + w3*port_value + w4*expansion_value - w5*robber_risk - w6*hand_over7_risk`

Where:
- `E[prod]` is expected resources/turn from owned vertices (pips-weighted)
- `robber_risk` uses exposure to high-pip tiles (6/8) and hand size
- `expansion_value` estimates accessible future settlement sites along road frontiers

### 5.2 Setup-phase “draft MCTS”
For initial placements, heuristics are weakest because the game is effectively a draft. Use a small MCTS just for setup:
- nodes = partial placement states (snake order)
- actions = legal settlement/road placements
- rollout evaluation = fast heuristic full-game rollouts (or a trained value net)

This captures “if I take X, others will take Y and ruin my plan.”

### 5.3 ISMCTS for key turns (optional but the closest thing to “game theory”)
ISMCTS loop for the current player:
1. sample a determinization consistent with player beliefs (opponent hands + dev deck)
2. run UCT search for N iterations
3. use rollout policy for opponents
4. choose action with highest visit count or value

This is expensive, so use it only:
- during setup
- when deciding robber placement / dev play
- when choosing build order in midgame

### 5.4 RL policy/value network (GPU path)
If you actually want “as close as possible” at scale, you eventually train.
Pipeline:
- represent observation as fixed tensor (board + public + private)
- train a policy/value net via self-play (PPO/IMPALA)
- during simulation, replace opponent heuristics with the policy net
- batch inference across many simultaneous rollouts on the GPU

This is how you get both strength and speed.

---

## 6) Initial settlement ranking (top 8) — the actual analyzer

### 6.1 What you’re ranking (be explicit)
For a vertex `v`, define:
- `W(v) = P(win | you force first settlement at v, then everyone plays according to agent stack, under setup order, board config, RNG)`
- Optionally also compute: `E[VP_final]`, `E[turns_to_win | win]`, and `P(top2_finish)` for 8-player settings.

Rank vertices by `W(v)` (primary), break ties by `E[turns_to_win | win]` (faster is better).

### 6.2 Candidate generation (static filter before simulation)
Don’t simulate all vertices blindly.
Compute cheap static features per vertex:
- total_pips = sum pips(adjacent numbers)
- resource_vector = expected production per resource (pips-weighted)
- diversity = Shannon entropy of resource_vector (normalized)
- missing_key = penalty if brick or lumber absent (early tempo)
- port_synergy = value if vertex is on port and has surplus resource production
- expansion_potential = count/quality of reachable future settlement vertices within 2-3 roads

Take top `K` by a weighted static score (K ~ 20–40), then simulate.

### 6.3 Adaptive rollout allocation (bandit)
Your current system runs a fixed `R` rollouts per candidate. That wastes compute.

Use Thompson Sampling:
- For each candidate i, maintain `wins_i, games_i`
- Posterior `Beta(1+wins_i, 1+games_i-wins_i)`
- Repeatedly pick the candidate with highest sampled win rate and run another rollout
- Stop when you hit budget or top-8 ordering is stable with high probability

### 6.4 Parallel execution plan
Embarrassingly parallel across rollouts:
- Spawn `num_workers = os.cpu_count()` processes (not threads).
- Each worker runs a batch of rollouts and returns aggregated metrics.
- Board is shared read-only (shared memory).
- Each worker has its own RNG seeded deterministically from (board_hash, candidate_id, rollout_range).

GPU optional:
- If you use a policy net, run inference in a dedicated GPU process; workers send observation batches via shared queue.

### 6.5 Output payload
For each of the top 8:
- vertex id + coordinates
- adjacent tiles (resource, number, pips)
- on-port? which port?
- `win_rate_mean` + 95% credible interval
- expected turns to win (conditional and unconditional)
- explanation strings: “why this beats the next best”

Also output a *predicted first-round pick list* for 8-player snake draft from setup-MCTS.

---

## 7) Validation suite (non-negotiable)
- rule tests: dev timing, bank shortage exception, robber discard, trade legality, victory timing
- property tests: invariants (non-negative resources, piece counts, distance rule)
- deterministic tests: given seed, final winner/turn reproducible

---

## 8) Minimal pseudocode for the rollout harness

```python
def evaluate_vertices(board, candidates, cfg):
    shared_board = freeze_board(board)  # shared memory arrays
    stats = {v: BetaStats() for v in candidates}

    for _ in range(cfg.total_rollouts_budget):
        v = thompson_pick(stats)
        submit_rollout_job(v)

    collect_results_and_update(stats)
    return top8(stats)
```

Worker:
```python
def worker_job(board_shm, candidate_v, seed, batch_n):
    rng = PCG(seed)
    wins = 0
    turns_to_win_sum = 0
    for i in range(batch_n):
        s0 = initial_state(board_shm, rng)
        s0 = force_setup_first_settlement(s0, candidate_v)
        s0 = play_setup_with_draft_mcts_or_policy(s0, rng)
        winner, turn = play_full_game(s0, agents, rng, cfg.max_turns)
        if winner == FOCAL_PLAYER:
            wins += 1
            turns_to_win_sum += turn
    return wins, batch_n, turns_to_win_sum
```

---

## 9) Notes on 8-player “base board” realism
The official base components and rules assume 3–4 players. An 8-player game on the standard 19-hex board is extremely congested, increases bank shortages, and amplifies robber impact. This is fine for simulation—just treat it as a custom config—but it will not match the “intended” balancing. If you instead use an expanded board, put the expansion map into the board config and the entire framework still works.

---
