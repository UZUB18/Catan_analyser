# Future Upgrades Roadmap

## Purpose
- Keep a practical plan for improving prediction quality, speed, and trustworthiness.
- Tie roadmap items directly to the four shipped prediction engines.

## Current engines: usage + reliability focus

| Engine | Best use today | Current reliability risks | Priority fixes |
|---|---|---|---|
| `heuristic` | Instant baseline ranking | Weight bias; limited temporal/game-flow realism | Weight calibration, redundancy/flexibility features |
| `monte_carlo` | Probabilistic yield smoothing | High variance at low iteration budgets | Confidence intervals, adaptive stopping, CRN |
| `phase_rollout_mc` | Best default opening evaluator | Rollout assumptions and policy sensitivity | Opponent-policy expansion, better trade/build realism |
| `mcts_lite_opening` | Highest-quality opening line search | Branching noise, compute limits | Progressive widening, transpositions, stronger priors |
| `hybrid_opening` | Most reliable blended recommendation for top-8 starts | Runtime and weight calibration complexity | Adaptive weighting, confidence reporting, auto-tuning |

## Roadmap (0-3 months)

### Heuristic engine
- Calibrate weights with seeded self-play tournaments and publish deltas.
- Add explicit resource-redundancy penalty unless conversion paths are strong.
- Add early flexibility signal (4:1/3:1/2:1 conversion accessibility).

### Monte Carlo engine
- Add per-vertex confidence intervals in the UI.
- Use common random numbers for fair mode/policy A/B comparisons.
- Add adaptive iteration stopping once rank confidence is high enough.

### Phase Rollout MC engine
- Expand opponent policy library (`balanced`, `greedy_ports`, `road_rush`).
- Improve build/trade realism before third settlement (especially road chaining).
- Expose robber-policy sensitivity report beside top picks.

### MCTS-lite engine
- Add progressive widening for high-branching settlement/road decisions.
- Add transposition table keyed by canonical opening state.
- Warm-start node priors from heuristic + phase rollout scores.

### Cross-mode reliability program
- Create fixed seed-board benchmark suite for regression tracking.
- Track rank-stability metrics (top-3 overlap, top-8 Kendall tau, runtime).
- Add performance budget tests to prevent silent speed regressions.
- Add hybrid disagreement diagnostics (which engine vetoed/reinforced each top pick).

## Roadmap (3-9 months)
- Hybrid engine: MCTS root search + phase rollout leaves + MC uncertainty estimates.
- Opponent modeling: learned pick tendencies by player archetype and seat order.
- Fast surrogate evaluator for near-real-time UI ranking refreshes.
- Policy bake-off harness with leaderboard (win rate, turn-20 VP, robber exposure, tempo-to-third).

## LLM-in-the-loop ideas (with guardrails)

### Candidate roles
- Explanation layer: convert engine features into plain-language "why this pick" summaries.
- Coaching layer: structured "what-if" comparison for user-proposed openings.
- Policy ideation: propose candidate robber/opponent policy variants for offline testing.

### Mandatory guardrails
1. **LLM never writes scores directly**; it can only suggest candidates for simulation.
2. **Structured IO only** (schema-validated metrics, no free-form engine control).
3. **Offline gatekeeping**: promote changes only if benchmark deltas exceed preset thresholds.
4. **Human approval required** before enabling any LLM-suggested default policy.
5. **Full audit trail**: prompt/version/seed/metric logging for every accepted change.
6. **Safety fallback**: automatic rollback to last validated policy pack on regression.

### Rollout plan
- Phase 1 (shadow): LLM explanations only, no policy impact.
- Phase 2 (assist): LLM proposes policies, simulator validates, humans review.
- Phase 3 (limited production): opt-in LLM coaching with hard guardrails and monitoring.

## Open questions
- What benchmark gain threshold is required for default-policy promotion?
- Which player count (2/3/4) should anchor first-round calibration?
- How should uncertainty be shown so users trust, but do not over-trust, rankings?
