# Catan Analyzer (Tkinter)

Catan opening analyzer for 2-4 players with randomized base-game boards.

## Run
```bash
python main.py
```

## Catanatron integration
- The project now vendors a local `catanatron/` package (core + players + CLI).
- A dedicated **Open Catanatron CLI** button/menu item is available in the Tkinter app.
- You can also run the CLI directly:
```bash
python -m catanatron.cli.play --help-players
catanatron-play --players=R,R,R,R --num=20
catanatron --players=R,R,R,R --num=20
```

## Test
```bash
python -m unittest discover -s tests -v
```

## Prediction engines (analysis modes)

| Mode | What it predicts | When to use it | Reliability notes |
|---|---|---|---|
| `heuristic` | Static opening value from yield/diversity/ports/risk/topology/synergy | Fast scan, live UI iteration, quick what-if checks | Deterministic for a fixed board/config. Lowest compute cost, but least game-flow realism. |
| `monte_carlo` | Expected production via dice-roll simulation, then heuristic ranking | Smooth out raw pip variance; compare close vertices probabilistically | Stochastic unless `mc_seed` is fixed. More iterations and rolls/game reduce noise. |
| `phase_rollout_mc` | Opening strength via rollout features (tempo, recipe coverage, fragility, port conversion, robber exposure) | Best default for stronger opening recommendations without full tree search | Seeded runs are reproducible. Sensitive to rollout count/horizon and robber policy assumptions. |
| `mcts_lite_opening` | Settlement + road-intent search with opponent responses and explainable principal line | Highest-fidelity opening planning, blocking-aware choices, explainer usage | Strongest opening planner here but slowest. Stability improves with more iterations/candidate limits and fixed seed. |
| `hybrid_opening` | Consensus of heuristic + phase rollout + MCTS-lite (+ optional Monte Carlo) | Recommended “all-around” mode when you want robust opening picks | Most stable top-8 for practical play; penalizes disagreement across engines to avoid brittle picks. |

## Practical mode selection
- Need speed first: **`heuristic`**.
- Need better probabilistic yield estimates: **`monte_carlo`**.
- Need balanced quality/runtime for real opening picks: **`phase_rollout_mc`**.
- Need sequence-level planning + blocking + explanation: **`mcts_lite_opening`**.
- Want the most reliable blended recommendation: **`hybrid_opening`**.

## Reliability and scope
- For reproducible stochastic results, set `mc_seed`.
- If `mc_seed` is left empty, stochastic modes now derive a deterministic seed from board layout.
- Low simulation budgets can change rank order on close candidates.
- Outputs are **opening-focused** (not full hidden-information Catan play).
- Robber behavior is policy-driven (`target_strongest_opponent`, `random_legal_move`, `worst_case_us`), so use a policy matching your risk posture.
- Every mode returns draft-aware pick sequence and `2 * player_count` highlighted recommendations.
- The mode picker in the UI now shows reliability/speed guidance before you run analysis.

## Core capabilities
- Standard 19-hex board randomization with official tile/token counts.
- Red-number spacing rule enforced (6/8 not adjacent).
- Topology-aware frontier + best-path expansion scoring.
- Draft simulation with official snake order:
  - 2 players: `1-2-2-1`
  - 3 players: `1-2-3-3-2-1`
  - 4 players: `1-2-3-4-4-3-2-1`
- Ranking table overlays and click-to-highlight vertex inspection.
