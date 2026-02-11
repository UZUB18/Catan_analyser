from __future__ import annotations

from dataclasses import dataclass

from catan_analyzer.analysis.types import VertexScore


@dataclass(frozen=True)
class SensitivityBadge:
    key: str
    label: str
    explanation: str


_STABLE_BADGE = SensitivityBadge(
    key="stable",
    label="Stable",
    explanation="Rank likely robust to small parameter tweaks.",
)
_WATCH_BADGE = SensitivityBadge(
    key="watch",
    label="Watch",
    explanation="Nearby scores are close; small tweaks may reorder neighbors.",
)
_VOLATILE_BADGE = SensitivityBadge(
    key="volatile",
    label="Volatile",
    explanation="Very tight score cluster; ranking is sensitive to small changes.",
)


def compute_sensitivity_badges(ranking: list[VertexScore]) -> dict[int, SensitivityBadge]:
    if not ranking:
        return {}

    if len(ranking) == 1:
        return {ranking[0].vertex_id: _STABLE_BADGE}

    top_score = ranking[0].total_score
    bottom_score = ranking[-1].total_score
    span = max(1e-6, top_score - bottom_score)

    badges: dict[int, SensitivityBadge] = {}
    for index, score in enumerate(ranking):
        prev_gap = (ranking[index - 1].total_score - score.total_score) if index > 0 else None
        next_gap = (score.total_score - ranking[index + 1].total_score) if index < len(ranking) - 1 else None
        candidate_gaps = [gap for gap in (prev_gap, next_gap) if gap is not None]
        closest_gap = min(candidate_gaps) if candidate_gaps else span
        normalized_gap = closest_gap / span

        if normalized_gap >= 0.08 or closest_gap >= 0.22:
            badges[score.vertex_id] = _STABLE_BADGE
        elif normalized_gap >= 0.035 or closest_gap >= 0.10:
            badges[score.vertex_id] = _WATCH_BADGE
        else:
            badges[score.vertex_id] = _VOLATILE_BADGE
    return badges


def explain_vertex_score(
    score: VertexScore,
    badge: SensitivityBadge | None = None,
) -> str:
    positives = [
        ("Yield", score.expected_yield),
        ("Diversity", score.diversity_score),
        ("Port", score.port_score),
        ("Synergy", score.synergy_score),
        ("Frontier", score.frontier_score),
        ("Best path", score.best_path_score),
        ("Tempo", score.tempo_score),
        ("Recipe", score.recipe_coverage_score),
        ("Port conv", score.port_conversion_score),
    ]
    penalties = [
        ("Risk", score.risk_penalty),
        ("Fragility", score.fragility_penalty),
        ("Robber", score.robber_penalty),
    ]

    top_positives = [entry for entry in positives if entry[1] > 0.0]
    top_positives.sort(key=lambda item: item[1], reverse=True)
    top_penalties = [entry for entry in penalties if entry[1] > 0.0]
    top_penalties.sort(key=lambda item: item[1], reverse=True)

    lines = [
        f"Vertex {score.vertex_id} | Total score: {score.total_score:.2f}",
    ]
    if badge is not None:
        lines.append(f"Sensitivity: {badge.label} — {badge.explanation}")

    lines.append("\nTop contributions:")
    if top_positives:
        for label, value in top_positives[:4]:
            lines.append(f"  + {label:<10} {value:>6.2f}")
    else:
        lines.append("  + None")

    lines.append("\nTop penalties:")
    if top_penalties:
        for label, value in top_penalties[:3]:
            lines.append(f"  - {label:<10} {value:>6.2f}")
    else:
        lines.append("  - None")

    lines.append("\nFull breakdown:")
    lines.append(
        "  Yield={:.2f} Diversity={:.2f} Port={:.2f} Risk={:.2f} Synergy={:.2f}".format(
            score.expected_yield,
            score.diversity_score,
            score.port_score,
            score.risk_penalty,
            score.synergy_score,
        )
    )
    lines.append(
        "  Frontier={:.2f} BestPath={:.2f} Tempo={:.2f} Recipe={:.2f}".format(
            score.frontier_score,
            score.best_path_score,
            score.tempo_score,
            score.recipe_coverage_score,
        )
    )
    lines.append(
        "  Fragility={:.2f} PortConv={:.2f} Robber={:.2f}".format(
            score.fragility_penalty,
            score.port_conversion_score,
            score.robber_penalty,
        )
    )
    return "\n".join(lines)

