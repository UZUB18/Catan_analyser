# Catan Analyzer — Implemented Changes & Upgrades

This file summarizes the upgrades implemented in this project during this build cycle.

## 1) Core Gameplay/Board Improvements

- Added realistic base-game randomization using official tile/token counts.
- Enforced red-token spacing validity rule:
  - `6` and `8` tokens are never adjacent.
- Kept standard 2–4 player snake draft order simulation:
  - 2p: `1-2-2-1`
  - 3p: `1-2-3-3-2-1`
  - 4p: `1-2-3-4-4-3-2-1`

## 2) Analysis Engines Added/Upgraded

Implemented and integrated 5 opening-prediction engines:

1. `heuristic`
   - Fast static scoring (yield/diversity/ports/risk/synergy/topology).

2. `monte_carlo`
   - Dice-roll simulation-based expected-yield override used in ranking.

3. `phase_rollout_mc`
   - Phase-only rollout evaluator (early game focus) with:
     - tempo-to-3rd-settlement scoring,
     - recipe coverage scoring,
     - fragility penalties,
     - port conversion viability,
     - robber exposure penalties.

4. `mcts_lite_opening`
   - Opening tree search over settlement + road-intent plies.
   - Opponent best-response heuristic with blocking externality impact.
   - UCT-style selection + rollout/evaluation integration.

5. `hybrid_opening`
   - New consensus analyzer that blends outputs from:
     - heuristic,
     - phase rollout,
     - MCTS-lite,
     - optional Monte Carlo.
   - Uses weighted normalized signals + stability/disagreement penalty.
   - Produces ranked recommendations and explanation lines.

## 3) Topology & Blocking Upgrades

- Added explicit topology helpers and usage in scoring:
  - expansion frontier value,
  - best path value (value-distance tradeoff).
- Added blocking/externality evaluation utilities:
  - best expansion snapshots,
  - blocking delta estimation against focal player.

## 4) MCTS-lite Reliability/Tuning Upgrades

- Added explanation output (`explain_lines`, `mcts_summary`) to results.
- Added “port-first discouragement” heuristics for opening settlement quality:
  - first-pick port penalties when production support is weak.
- Reduced overemphasis on port conversion in phase evaluator weighting.
- Improved action expansion behavior (no hard first-only expansion bias).

## 5) Determinism & Reproducibility Upgrades

- Added deterministic seed utilities:
  - board-derived fallback seeds when `mc_seed` is unset.
- Applied deterministic seeding to stochastic engines:
  - Monte Carlo expected-yield simulation,
  - Phase rollout evaluator,
  - MCTS-lite analyzer.
- Made phase rollouts order-independent by deriving per-evaluation/per-rollout RNG streams.

## 6) UI/UX Upgrades

- Added mode selector support for all implemented analyzers (including hybrid).
- Added mode-specific parameter panels:
  - Monte Carlo settings,
  - Phase rollout settings,
  - MCTS-lite settings,
  - Hybrid consensus settings/weights.
- Added **Mode details** panel (pre-analysis):
  - description,
  - reliability note,
  - speed note,
  - best-use guidance per mode.
- Added MCTS/Hybrid explanation panel in output area.
- Improved row-click interaction:
  - selecting a ranking row highlights the settlement vertex and adjacent resource hexes it controls.

## 7) Data Model/Config Upgrades

- Extended `AnalysisMode` enum with:
  - `mcts_lite_opening`,
  - `hybrid_opening`.
- Extended `AnalysisConfig` with MCTS and Hybrid tuning fields.
- Extended `AnalysisResult` with explanation outputs:
  - `explain_lines`,
  - `mcts_summary`.

## 8) Documentation Upgrades

- Updated `README.md` with:
  - engine descriptions,
  - reliability/speed usage guidance,
  - reproducibility notes.
- Updated `FUTURE_UPGRADES.md` with:
  - engine-focused roadmap,
  - reliability program priorities,
  - LLM-in-the-loop opportunities and safety guardrails.

## 9) Test Coverage Upgrades

- Expanded and updated tests for:
  - deterministic behavior under seeded configs,
  - reproducibility in stochastic modes (including no explicit seed),
  - MCTS-lite explanation population,
  - hybrid mode output consistency and explanation presence,
  - board/randomizer validity constraints.
- Current suite passes with expanded coverage.

## 10) Files Added During These Upgrades

- `catan_analyzer/analysis/blocking.py`
- `catan_analyzer/analysis/mcts_lite.py`
- `catan_analyzer/analysis/seeding.py`
- `catan_analyzer/ui/mode_descriptions.py`
- `FUTURE_UPGRADES.md` (roadmap content refreshed/expanded)
- `UPGRADES_IMPLEMENTED.md` (this file)

---

If desired, this can be split into a formal `CHANGELOG.md` format (versioned by date/release tag).
