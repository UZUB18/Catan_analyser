from __future__ import annotations

from dataclasses import dataclass

from catan_analyzer.analysis.types import AnalysisMode


@dataclass(frozen=True)
class ModeDescription:
    summary: str
    reliability: str
    speed: str
    best_use: str

    def to_ui_text(self) -> str:
        return (
            f"{self.summary}\n"
            f"Reliability: {self.reliability}\n"
            f"Speed: {self.speed}\n"
            f"Best use: {self.best_use}"
        )


_DEFAULT_DESCRIPTION = ModeDescription(
    summary="Custom analysis mode.",
    reliability="Varies by implementation.",
    speed="Varies by implementation.",
    best_use="Use when this mode is explicitly recommended for your workflow.",
)


_HYBRID_DESCRIPTION = ModeDescription(
    summary="Hybrid strategy that combines fast scoring with deeper simulation.",
    reliability="High for balanced pick quality across many board states.",
    speed="Medium-slow (faster than full deep search, slower than heuristic).",
    best_use="Strong default when you want better confidence without max runtime.",
)


_MODE_DESCRIPTIONS: dict[str, ModeDescription] = {
    AnalysisMode.HEURISTIC.value: ModeDescription(
        summary="Rule-based static scoring of yield, diversity, ports, and risk.",
        reliability="Medium and fully deterministic.",
        speed="Very fast.",
        best_use="Quick board scans and rapid iteration while testing setups.",
    ),
    AnalysisMode.MONTE_CARLO.value: ModeDescription(
        summary="Dice-roll simulation estimates expected production before ranking.",
        reliability="Medium-high when iterations are sufficiently large.",
        speed="Medium (scales with iteration and roll count).",
        best_use="When heuristic ties need variance-aware separation.",
    ),
    AnalysisMode.PHASE_ROLLOUT_MC.value: ModeDescription(
        summary="Short-horizon rollout model adds tempo, recipes, and robber pressure.",
        reliability="High for early-game momentum and resilience comparisons.",
        speed="Slow.",
        best_use="Comparing openings where development path quality matters.",
    ),
    AnalysisMode.MCTS_LITE_OPENING.value: ModeDescription(
        summary="Tree search explores opening lines with opponent blocking impact.",
        reliability="High on contested top candidates; stochastic but informative.",
        speed="Slowest.",
        best_use="Final decision support for high-stakes opening picks.",
    ),
    AnalysisMode.HYBRID_OPENING.value: _HYBRID_DESCRIPTION,
    AnalysisMode.FULL_GAME.value: ModeDescription(
        summary="Full-rule game simulation with robber, dev cards, awards, and finite bank/deck.",
        reliability="High for strategic race dynamics when rollout count is sufficient.",
        speed="Slowest (multi-turn, full-game rollouts).",
        best_use="When you need win-odds style guidance instead of opening-only rankings.",
    ),
    "hybrid": _HYBRID_DESCRIPTION,
}


def get_mode_description(mode: AnalysisMode | str) -> ModeDescription:
    mode_value = mode.value if isinstance(mode, AnalysisMode) else str(mode)
    normalized = mode_value.strip().lower()
    if normalized in _MODE_DESCRIPTIONS:
        return _MODE_DESCRIPTIONS[normalized]
    if "hybrid" in normalized:
        return _HYBRID_DESCRIPTION
    return _DEFAULT_DESCRIPTION
