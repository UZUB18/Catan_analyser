# How the king has fallen

## From AB dominance to GTv4 supremacy

This document explains the evolution of the Game Theory (GT) engines in this codebase:

- **GT classic** (`GameTheoryEngine`, v1.3.0 baseline logic)
- **GTv2**
- **GTv3**
- **GTv4** (`GameTheoryEngineV4`, current “ab_killer” peak)

It covers the engine philosophy, the math behind decisions, and why GTv4 now beats the former king (`AB:2:true`) in large benchmarks.

---

## 1) The old king: why AB was so hard to beat

The stock **AlphaBetaPlayer** is strong because it does one thing very well:

- Looks ahead several plies with pruning
- Uses expected values over stochastic outcomes (via outcome expansion)
- Is extremely tactically clean in short horizons

In Catan terms, AB is good at:

- “If I do this trade/build now, what tactical line opens immediately?”
- Not missing obvious forced tactical conversions in its search window

Historically, lightweight heuristic engines struggled because AB punished:

- shallow lookahead
- weak trade discipline
- tactical blunders near endgame

---

## 2) GT classic (v1.3.0): the foundation

GT classic already had a strong strategic architecture:

## 2.1 Core objective

GT does not only maximize self-score. It optimizes:

\[
\text{Objective}(s) \approx V_{\text{me}}(s) - \lambda(s)\cdot V_{\text{threat}}(s)
\]

Where:

- \(V_{\text{me}}\): your own value-function score
- \(V_{\text{threat}}\): strongest opponent pressure (leader / max opponent eval)
- \(\lambda(s)\): dynamic threat weight (higher late-game)

This is why GT “feels” political and table-aware rather than purely selfish.

## 2.2 Stochastic handling

GT evaluates actions using expected value over outcome branches:

\[
EV(a) = \sum_i p_i \cdot \text{Objective}(s_i)
\]

For robber interactions, weighted steal probabilities are modeled rather than naive uniform assumptions.

## 2.3 Catan-aware heuristics

- Opening placement logic with simulation + evaluator
- Resource scarcity weighting
- Trade stance scoring (gain vs feeding leader)
- Candidate filtering to control branching

**Limitation:** AB still had an edge in tactical short-horizon cleanup, especially in volatile or conversion-heavy turns.

---

## 3) GTv2: stronger tactical realism

GTv2 pushed tactical quality up by tightening decision-time structure:

- Better action candidate quality before full scoring
- Improved trade and opponent-reply-aware behavior
- Better blend of EV and tactical checks

The result: less “strategic but tactically loose” behavior.

But AB could still outperform in high-pressure tactical windows.

---

## 4) GTv3: hybrid architecture

GTv3 introduced the big structural jump: **hybrid search + caching + delegate tactics**.

## 4.1 Persistent transposition table (TT)

GTv3 hashes game states and stores searched values:

- avoids re-solving repeated states
- improves move ordering (TT-best first)
- increases effective depth under same compute

## 4.2 Iterative deepening + history ordering

Instead of brittle fixed-depth one-shot search, GTv3 deepens progressively and learns ordering from prior good moves:

- early tactical signal appears faster
- time budget gets used more reliably

## 4.3 Opponent reply approximation

A cheap one-reply model penalizes moves that look good now but hand immediate tactical punishment to the next player.

## 4.4 AB tactical delegate (the turning point)

GTv3 can selectively call an internal AB core in “tactical windows” (branching/endgame/conversion/pass-risk), preserving GT strategy while borrowing AB’s best local tactic.

This gave the first serious anti-AB profile: **`GTv3:profile=ab_killer`**.

## 4.5 Critical special-phase upgrades (already landed before v4)

- **Discard by real value function** (not coarse fallback utility)
- **Robber heuristic with coordinate-aware leader blocking**

These removed costly phase-specific leaks.

---

## 5) GTv4: the ultimate anti-AB iteration

GTv4 keeps GTv3’s full strength and adds roadmap upgrades 3/4/5.

---

## 5.1 Upgrade 3: Dynamic AB-delegate depth

Static depth=2 is cheap and strong, but some positions justify depth=3.

GTv4 uses:

- default tactical depth = 2
- temporary depth = 3 when:
  - endgame pressure (my VP or leader VP high),
  - branching is small (cheap),
  - hard conversion opportunities exist (city/settlement/dev conversion lines)

This is a compute-allocation strategy:

- spend deeper search only where marginal tactical value is highest
- keep runtime practical in high-branching states

---

## 5.2 Upgrade 4: Sanity veto layer (multiplayer blunder filter)

AB can still make multiplayer-strategic mistakes, especially in trade resolution.

GTv4 adds a conservative veto:

- If a proposed trade feeds the VP leader and newly enables immediate VP structures (city/settlement affordability), veto it.
- Replace with safer alternative (`REJECT_TRADE` / `CANCEL_TRADE` / non-leader confirm when available).

This preserves tactical cleanliness without allowing obvious kingmaking.

Also, trade parsing/partner identification was hardened so “leader-feeding” detection actually works across trade value shapes.

---

## 5.3 Upgrade 5: instrumentation-driven tuning

GTv4 exposes per-game diagnostics and counters:

- action type distribution
- AB delegate depth usage (d2 vs d3)
- veto counts and reasons
- mode usage

This turns tuning into an empirical loop:

1. Benchmark
2. Inspect failure patterns
3. Adjust thresholds/windows
4. Re-benchmark

No more blind parameter guessing.

---

## 6) Why GTv4 plays Catan so well now

GTv4’s strength is not one trick; it is a layered control system:

1. **Strategic objective**: maximize own growth while suppressing table leader.
2. **Probabilistic realism**: expected-value handling for stochastic Catan outcomes.
3. **Tactical precision**: AB delegation in key windows.
4. **Multiplayer sanity**: veto obvious kingmaking trades.
5. **Phase quality**: strong discard/robber handling (historically common leak points).
6. **Compute efficiency**: TT + iterative deepening + selective depth escalation.

In practical gameplay this means:

- Converts resources efficiently without blindly overextending
- Disrupts leaders without self-sabotage
- Avoids “good-looking but strategically losing” trade choices
- Maintains tactical sharpness when race-to-10 becomes concrete

---

## 7) Statistical proof that the king fell

From your 1000-game benchmark vs `AB:2:true`:

- **GTv4 wins:** 382 / 1000 = **38.2%**
- **AB per-player baseline:** **20.6%**
- **Advantage:** **+17.6 percentage points**

This is a very large and statistically decisive edge in this benchmark setup.

Seat breakdown also shows GTv4 remains strong across positions (with expected seat variance in multiplayer Catan).

---

## 8) Engine timeline summary

- **GT classic (v1.3.0):** strategic EV + threat-aware foundation
- **GTv2:** stronger tactical realism and candidate quality
- **GTv3:** hybrid architecture (TT, iterative deepening, opponent-reply modeling, AB tactical delegate)
- **GTv4:** dynamic depth + multiplayer sanity veto + instrumentation loop

In other words:

> GT started as a strong strategist, learned tactical brutality from AB, then learned when to overrule AB’s multiplayer blind spots.

That is why **GTv4:profile=ab_killer** now outperforms the former king.

