from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AnalysisMode(str, Enum):
    HEURISTIC = "heuristic"
    MONTE_CARLO = "monte_carlo"
    PHASE_ROLLOUT_MC = "phase_rollout_mc"
    MCTS_LITE_OPENING = "mcts_lite_opening"
    HYBRID_OPENING = "hybrid_opening"
    FULL_GAME = "full_game"


class RobberPolicy(str, Enum):
    TARGET_STRONGEST_OPPONENT = "target_strongest_opponent"
    RANDOM_LEGAL_MOVE = "random_legal_move"
    WORST_CASE_US = "worst_case_us"


@dataclass(frozen=True)
class AnalysisConfig:
    player_count: int = 4
    mode: AnalysisMode = AnalysisMode.HEURISTIC
    include_ports: bool = True
    mc_iterations: int = 25_000
    mc_rolls_per_game: int = 60
    mc_seed: Optional[int] = None
    phase_rollout_count: int = 180
    phase_turn_horizon: int = 20
    robber_policy: RobberPolicy = RobberPolicy.TARGET_STRONGEST_OPPONENT
    allow_bank_trading: bool = False
    mcts_iterations: int = 140
    mcts_max_plies: int = 24
    mcts_exploration_c: float = 1.15
    mcts_candidate_settlements: int = 10
    mcts_candidate_road_directions: int = 3
    opponent_block_weight: float = 0.30
    hybrid_include_monte_carlo: bool = False
    hybrid_stability_penalty_weight: float = 0.24
    hybrid_weight_heuristic: float = 0.24
    hybrid_weight_phase_rollout: float = 0.32
    hybrid_weight_mcts_lite: float = 0.44
    hybrid_weight_monte_carlo: float = 0.18
    parallel_workers: int = 0  # 0 = auto
    full_game_rollouts: int = 24
    full_game_max_turns: int = 180
    full_game_candidate_vertices: int = 10
    full_game_trade_offer_limit: int = 2


@dataclass
class VertexScore:
    vertex_id: int
    total_score: float
    expected_yield: float
    diversity_score: float
    port_score: float
    risk_penalty: float
    synergy_score: float = 0.0
    frontier_score: float = 0.0
    best_path_score: float = 0.0
    tempo_score: float = 0.0
    recipe_coverage_score: float = 0.0
    fragility_penalty: float = 0.0
    port_conversion_score: float = 0.0
    robber_penalty: float = 0.0


@dataclass
class DraftPick:
    turn_index: int
    player_id: int
    vertex_id: int
    score_snapshot: VertexScore


@dataclass
class MctsLineExplanation:
    ply_index: int
    actor: int
    action: str
    self_value: float
    blocking_delta: float
    uct_value: float
    visits: int


@dataclass
class MctsSummary:
    root_visits: int
    best_line_score: float
    alt_line_score_gap: float
    runtime_ms: float


@dataclass
class FullGameCandidateReport:
    vertex_id: int
    simulations: int
    win_rate: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    avg_win_turn: float
    avg_game_length_turns: float
    avg_robber_loss_to_focal: float
    total_pips: int
    on_port: bool


@dataclass
class FullGameSummary:
    rollout_count: int
    player_win_rates: dict[int, float]
    expected_turns_to_victory: dict[int, float]
    average_game_length_turns: float
    top_candidates: list[FullGameCandidateReport] = field(default_factory=list)
    predicted_first_round: list[int] = field(default_factory=list)


@dataclass
class AnalysisResult:
    global_ranking: list[VertexScore]
    predicted_sequence: list[DraftPick]
    top_recommendations: list[VertexScore]
    explain_lines: list[MctsLineExplanation] = field(default_factory=list)
    mcts_summary: Optional[MctsSummary] = None
    full_game_summary: Optional[FullGameSummary] = None
