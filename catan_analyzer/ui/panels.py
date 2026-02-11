from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from catan_analyzer.analysis.types import AnalysisConfig, AnalysisMode, AnalysisResult, RobberPolicy
from catan_analyzer.ui.explainability import compute_sensitivity_badges, explain_vertex_score
from catan_analyzer.ui.mode_descriptions import get_mode_description
from catan_analyzer.ui.themes import UiTheme, available_themes, get_theme


class AnalyzerControls(ttk.LabelFrame):
    def __init__(
        self,
        master: tk.Widget,
        *,
        on_randomize,
        on_analyze,
        on_cancel=None,
        on_knowledge_test_toggle=None,
        on_theme_changed=None,
        on_ui_scale_changed=None,
        on_scroll_request=None,
    ) -> None:
        super().__init__(master, text="Controls", padding=10)
        self._on_randomize = on_randomize
        self._on_analyze = on_analyze
        self._on_cancel = on_cancel
        self._on_knowledge_test_toggle = on_knowledge_test_toggle
        self._on_theme_changed = on_theme_changed
        self._on_ui_scale_changed = on_ui_scale_changed
        self._on_scroll_request = on_scroll_request

        self.player_count_var = tk.IntVar(value=4)
        self.mode_var = tk.StringVar(value=AnalysisMode.HEURISTIC.value)
        self.include_ports_var = tk.BooleanVar(value=True)
        self.knowledge_test_enabled_var = tk.BooleanVar(value=False)
        self.theme_var = tk.StringVar(value="light")
        self.ui_scale_var = tk.StringVar(value="100%")
        self.parallel_workers_var = tk.IntVar(value=0)
        self.tuning_profile_var = tk.StringVar(value="balanced")
        self.tuning_effort_var = tk.IntVar(value=3)
        self.mc_iterations_var = tk.IntVar(value=25000)
        self.mc_rolls_var = tk.IntVar(value=60)
        self.phase_rollout_count_var = tk.IntVar(value=180)
        self.phase_horizon_var = tk.IntVar(value=20)
        self.robber_policy_var = tk.StringVar(value=RobberPolicy.TARGET_STRONGEST_OPPONENT.value)
        self.allow_bank_trading_var = tk.BooleanVar(value=False)
        self.mcts_iterations_var = tk.IntVar(value=140)
        self.mcts_max_plies_var = tk.IntVar(value=24)
        self.mcts_exploration_c_var = tk.DoubleVar(value=1.15)
        self.mcts_settlement_candidates_var = tk.IntVar(value=10)
        self.mcts_road_candidates_var = tk.IntVar(value=3)
        self.mcts_opponent_block_weight_var = tk.DoubleVar(value=0.30)
        self.hybrid_include_mc_var = tk.BooleanVar(value=False)
        self.hybrid_stability_weight_var = tk.DoubleVar(value=0.24)
        self.hybrid_weight_heuristic_var = tk.DoubleVar(value=0.24)
        self.hybrid_weight_phase_var = tk.DoubleVar(value=0.32)
        self.hybrid_weight_mcts_var = tk.DoubleVar(value=0.44)
        self.hybrid_weight_mc_var = tk.DoubleVar(value=0.18)
        self.mode_description_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_value_var = tk.DoubleVar(value=0.0)
        self.progress_detail_var = tk.StringVar(value="Idle • 0.0% • elapsed 00:00 • ETA --:--")
        self._card_fade_after_id: str | None = None
        self._card_text_widgets: list[tk.Label] = []
        light_theme = get_theme("light")
        self._card_fade_start = light_theme.card_text_start
        self._card_fade_end = light_theme.card_text_end
        self._card_title_lookup = {
            AnalysisMode.HEURISTIC.value: "Heuristic Study",
            AnalysisMode.MONTE_CARLO.value: "Monte Carlo Study",
            AnalysisMode.PHASE_ROLLOUT_MC.value: "Phase Rollout Study",
            AnalysisMode.MCTS_LITE_OPENING.value: "MCTS-lite Study",
            AnalysisMode.HYBRID_OPENING.value: "Hybrid Consensus Study",
            AnalysisMode.FULL_GAME.value: "Full-Game Study",
        }
        self._section_titles = {
            "tuning": "Parameter Tuning",
            "monte_carlo": "Monte Carlo Settings",
            "phase": "Phase Rollout Settings",
            "mcts": "MCTS-lite Settings",
            "hybrid": "Hybrid Consensus Settings",
        }
        self._collapsed_sections = {key: False for key in self._section_titles}
        self._section_toggle_buttons: dict[str, ttk.Button] = {}
        self._section_frames: dict[str, tk.Widget] = {}

        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Players").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            self,
            values=[2, 3, 4],
            textvariable=self.player_count_var,
            state="readonly",
            width=8,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(self, text="Mode").grid(row=1, column=0, sticky="w", pady=(8, 0))
        mode_combo = ttk.Combobox(
            self,
            values=[mode.value for mode in AnalysisMode],
            textvariable=self.mode_var,
            state="readonly",
            width=18,
        )
        mode_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        self.mode_card_frame = tk.Frame(
            self,
            bg="#F2E7D2",
            bd=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground="#BFA47A",
            padx=10,
            pady=8,
        )
        self.mode_card_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.mode_card_frame.grid_columnconfigure(0, weight=1)

        self.mode_card_heading = tk.Label(
            self.mode_card_frame,
            text="Gallery Placard",
            bg="#F2E7D2",
            font=("Georgia", 11, "bold"),
            anchor="w",
            justify="left",
        )
        self.mode_card_heading.grid(row=0, column=0, sticky="w")

        self.mode_card_summary = tk.Label(
            self.mode_card_frame,
            text="",
            bg="#F2E7D2",
            wraplength=336,
            anchor="w",
            justify="left",
            font=("Georgia", 10),
        )
        self.mode_card_summary.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.mode_card_reliability = tk.Label(
            self.mode_card_frame,
            text="",
            bg="#F2E7D2",
            wraplength=336,
            anchor="w",
            justify="left",
            font=("Georgia", 10),
        )
        self.mode_card_reliability.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        self.mode_card_speed = tk.Label(
            self.mode_card_frame,
            text="",
            bg="#F2E7D2",
            wraplength=336,
            anchor="w",
            justify="left",
            font=("Georgia", 10),
        )
        self.mode_card_speed.grid(row=3, column=0, sticky="ew", pady=(2, 0))

        self.mode_card_best_use = tk.Label(
            self.mode_card_frame,
            text="",
            bg="#F2E7D2",
            wraplength=336,
            anchor="w",
            justify="left",
            font=("Georgia", 10),
        )
        self.mode_card_best_use.grid(row=4, column=0, sticky="ew", pady=(2, 0))

        self._card_text_widgets = [
            self.mode_card_heading,
            self.mode_card_summary,
            self.mode_card_reliability,
            self.mode_card_speed,
            self.mode_card_best_use,
        ]

        self.theme_row = tk.Frame(self.mode_card_frame, bg=self.mode_card_frame.cget("bg"))
        self.theme_row.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.theme_row.grid_columnconfigure(1, weight=1)
        self.theme_label = tk.Label(
            self.theme_row,
            text="Theme",
            bg=self.mode_card_frame.cget("bg"),
            anchor="w",
            justify="left",
            font=("Segoe UI", 9, "bold"),
        )
        self.theme_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.theme_combo = ttk.Combobox(
            self.theme_row,
            values=available_themes(),
            textvariable=self.theme_var,
            state="readonly",
            width=16,
        )
        self.theme_combo.grid(row=0, column=1, sticky="ew")
        self.theme_combo.bind("<<ComboboxSelected>>", self._handle_theme_change)
        self._guard_combobox_wheel_change(self.theme_combo)

        self.ui_scale_label = tk.Label(
            self.theme_row,
            text="UI Zoom",
            bg=self.mode_card_frame.cget("bg"),
            anchor="w",
            justify="left",
            font=("Segoe UI", 9, "bold"),
        )
        self.ui_scale_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.ui_scale_combo = ttk.Combobox(
            self.theme_row,
            values=["90%", "100%", "110%", "125%", "140%", "160%"],
            textvariable=self.ui_scale_var,
            state="readonly",
            width=16,
        )
        self.ui_scale_combo.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self.ui_scale_combo.bind("<<ComboboxSelected>>", self._handle_ui_scale_change)
        self._guard_combobox_wheel_change(self.ui_scale_combo)

        ttk.Checkbutton(
            self,
            text="Include ports in scoring",
            variable=self.include_ports_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            self,
            text="Knowledge test: pick 4 before Analyze",
            variable=self.knowledge_test_enabled_var,
            command=self._handle_knowledge_toggle,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(self, text="Parallel workers (0=auto)").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self, textvariable=self.parallel_workers_var, width=12).grid(
            row=5, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )

        self.tuning_toggle_button = ttk.Button(
            self,
            text=self._make_section_toggle_label("tuning"),
            command=lambda: self._toggle_section("tuning"),
        )
        self.tuning_toggle_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.tuning_frame = ttk.LabelFrame(self, text="Parameter tuning")
        self.tuning_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(self.tuning_frame, text="Profile").grid(row=0, column=0, sticky="w")
        profile_combo = ttk.Combobox(
            self.tuning_frame,
            values=["speed", "balanced", "quality", "exhaustive"],
            textvariable=self.tuning_profile_var,
            state="readonly",
            width=14,
        )
        profile_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        profile_combo.bind("<<ComboboxSelected>>", self._on_tuning_changed)

        ttk.Label(self.tuning_frame, text="Effort").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.tuning_effort_scale = tk.Scale(
            self.tuning_frame,
            from_=1,
            to=5,
            orient="horizontal",
            variable=self.tuning_effort_var,
            showvalue=True,
            command=self._on_tuning_scale_changed,
            length=160,
        )
        self.tuning_effort_scale.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        self.apply_tuning_button = ttk.Button(
            self.tuning_frame,
            text="Apply suggested values",
            command=self._apply_recommended_tuning,
        )
        self.apply_tuning_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.tuning_frame.columnconfigure(1, weight=1)

        self.mc_toggle_button = ttk.Button(
            self,
            text=self._make_section_toggle_label("monte_carlo"),
            command=lambda: self._toggle_section("monte_carlo"),
        )
        self.mc_toggle_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.legacy_mc_frame = ttk.LabelFrame(self, text="Monte Carlo")
        self.legacy_mc_frame.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(self.legacy_mc_frame, text="Iterations").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.legacy_mc_frame, textvariable=self.mc_iterations_var, width=12).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        ttk.Label(self.legacy_mc_frame, text="Rolls/game").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.legacy_mc_frame, textvariable=self.mc_rolls_var, width=12).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        self.legacy_mc_frame.columnconfigure(1, weight=1)

        self.phase_toggle_button = ttk.Button(
            self,
            text=self._make_section_toggle_label("phase"),
            command=lambda: self._toggle_section("phase"),
        )
        self.phase_toggle_button.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.phase_frame = ttk.LabelFrame(self, text="Phase Rollout MC")
        self.phase_frame.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(self.phase_frame, text="Rollouts").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.phase_frame, textvariable=self.phase_rollout_count_var, width=12).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        ttk.Label(self.phase_frame, text="Turn horizon").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.phase_frame, textvariable=self.phase_horizon_var, width=12).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.phase_frame, text="Robber policy").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            self.phase_frame,
            values=[policy.value for policy in RobberPolicy],
            textvariable=self.robber_policy_var,
            state="readonly",
            width=22,
        ).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Checkbutton(
            self.phase_frame,
            text="Allow 4:1 bank trading",
            variable=self.allow_bank_trading_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.phase_frame.columnconfigure(1, weight=1)

        self.mcts_toggle_button = ttk.Button(
            self,
            text=self._make_section_toggle_label("mcts"),
            command=lambda: self._toggle_section("mcts"),
        )
        self.mcts_toggle_button.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.mcts_frame = ttk.LabelFrame(self, text="MCTS-lite Opening")
        self.mcts_frame.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(self.mcts_frame, text="Iterations").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.mcts_frame, textvariable=self.mcts_iterations_var, width=10).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        ttk.Label(self.mcts_frame, text="Max plies").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.mcts_frame, textvariable=self.mcts_max_plies_var, width=10).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.mcts_frame, text="Exploration C").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.mcts_frame, textvariable=self.mcts_exploration_c_var, width=10).grid(
            row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.mcts_frame, text="Settle candidates").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.mcts_frame, textvariable=self.mcts_settlement_candidates_var, width=10).grid(
            row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.mcts_frame, text="Road candidates").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.mcts_frame, textvariable=self.mcts_road_candidates_var, width=10).grid(
            row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.mcts_frame, text="Opp block weight").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.mcts_frame, textvariable=self.mcts_opponent_block_weight_var, width=10).grid(
            row=5, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        self.mcts_frame.columnconfigure(1, weight=1)

        self.hybrid_toggle_button = ttk.Button(
            self,
            text=self._make_section_toggle_label("hybrid"),
            command=lambda: self._toggle_section("hybrid"),
        )
        self.hybrid_toggle_button.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.hybrid_frame = ttk.LabelFrame(self, text="Hybrid Consensus")
        self.hybrid_frame.grid(row=15, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(
            self.hybrid_frame,
            text="Include Monte Carlo component",
            variable=self.hybrid_include_mc_var,
            command=self._refresh_mode_specific_controls,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self.hybrid_frame, text="Stability penalty").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.hybrid_frame, textvariable=self.hybrid_stability_weight_var, width=10).grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.hybrid_frame, text="Weight: Heuristic").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.hybrid_frame, textvariable=self.hybrid_weight_heuristic_var, width=10).grid(
            row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.hybrid_frame, text="Weight: Phase").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.hybrid_frame, textvariable=self.hybrid_weight_phase_var, width=10).grid(
            row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.hybrid_frame, text="Weight: MCTS").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.hybrid_frame, textvariable=self.hybrid_weight_mcts_var, width=10).grid(
            row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(self.hybrid_frame, text="Weight: Monte Carlo").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.hybrid_frame, textvariable=self.hybrid_weight_mc_var, width=10).grid(
            row=5, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        self.hybrid_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(self)
        button_frame.grid(row=16, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.randomize_button = ttk.Button(button_frame, text="Randomize", command=self._on_randomize)
        self.randomize_button.pack(side="left", fill="x", expand=True)
        self.analyze_button = ttk.Button(button_frame, text="Analyze", command=self._on_analyze)
        self.analyze_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._handle_cancel)
        self.cancel_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.cancel_button.configure(state="disabled")

        self.progress_bar = ttk.Progressbar(
            self,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.progress_value_var,
        )
        self.progress_bar.grid(row=17, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(self, textvariable=self.progress_detail_var, foreground="#666666").grid(
            row=18, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        ttk.Label(self, textvariable=self.status_var, foreground="#444444").grid(
            row=19, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        self._register_collapsible_section("tuning", self.tuning_toggle_button, self.tuning_frame)
        self._register_collapsible_section("monte_carlo", self.mc_toggle_button, self.legacy_mc_frame)
        self._register_collapsible_section("phase", self.phase_toggle_button, self.phase_frame)
        self._register_collapsible_section("mcts", self.mcts_toggle_button, self.mcts_frame)
        self._register_collapsible_section("hybrid", self.hybrid_toggle_button, self.hybrid_frame)

        self.columnconfigure(1, weight=1)
        self._refresh_mode_specific_controls()
        self._apply_wheel_guard_to_comboboxes(self)
        self.reset_progress()

    def build_config(self) -> AnalysisConfig:
        mode = AnalysisMode(self.mode_var.get())
        player_count = int(self.player_count_var.get())
        mc_iterations = max(1, int(self.mc_iterations_var.get()))
        mc_rolls = max(1, int(self.mc_rolls_var.get()))
        phase_rollout_count = max(1, int(self.phase_rollout_count_var.get()))
        phase_horizon = max(1, int(self.phase_horizon_var.get()))
        robber_policy = RobberPolicy(self.robber_policy_var.get())
        mcts_iterations = max(1, int(self.mcts_iterations_var.get()))
        mcts_max_plies = max(1, int(self.mcts_max_plies_var.get()))
        mcts_exploration_c = max(0.01, float(self.mcts_exploration_c_var.get()))
        mcts_settlement_candidates = max(1, int(self.mcts_settlement_candidates_var.get()))
        mcts_road_candidates = max(1, int(self.mcts_road_candidates_var.get()))
        opponent_block_weight = min(1.0, max(0.0, float(self.mcts_opponent_block_weight_var.get())))
        hybrid_stability = max(0.0, float(self.hybrid_stability_weight_var.get()))
        hybrid_weight_h = max(0.0, float(self.hybrid_weight_heuristic_var.get()))
        hybrid_weight_p = max(0.0, float(self.hybrid_weight_phase_var.get()))
        hybrid_weight_mcts = max(0.0, float(self.hybrid_weight_mcts_var.get()))
        hybrid_weight_mc = max(0.0, float(self.hybrid_weight_mc_var.get()))
        parallel_workers = max(0, int(self.parallel_workers_var.get()))
        full_game_rollouts = max(4, int(round(phase_rollout_count / 6)))
        full_game_max_turns = max(80, int(round(phase_horizon * 9)))
        full_game_candidate_vertices = max(4, int(self.mcts_settlement_candidates_var.get()))
        full_game_trade_offer_limit = max(0, min(4, int(self.mcts_road_candidates_var.get())))
        return AnalysisConfig(
            player_count=player_count,
            mode=mode,
            include_ports=bool(self.include_ports_var.get()),
            mc_iterations=mc_iterations,
            mc_rolls_per_game=mc_rolls,
            phase_rollout_count=phase_rollout_count,
            phase_turn_horizon=phase_horizon,
            robber_policy=robber_policy,
            allow_bank_trading=bool(self.allow_bank_trading_var.get()),
            mcts_iterations=mcts_iterations,
            mcts_max_plies=mcts_max_plies,
            mcts_exploration_c=mcts_exploration_c,
            mcts_candidate_settlements=mcts_settlement_candidates,
            mcts_candidate_road_directions=mcts_road_candidates,
            opponent_block_weight=opponent_block_weight,
            hybrid_include_monte_carlo=bool(self.hybrid_include_mc_var.get()),
            hybrid_stability_penalty_weight=hybrid_stability,
            hybrid_weight_heuristic=hybrid_weight_h,
            hybrid_weight_phase_rollout=hybrid_weight_p,
            hybrid_weight_mcts_lite=hybrid_weight_mcts,
            hybrid_weight_monte_carlo=hybrid_weight_mc,
            parallel_workers=parallel_workers,
            full_game_rollouts=full_game_rollouts,
            full_game_max_turns=full_game_max_turns,
            full_game_candidate_vertices=full_game_candidate_vertices,
            full_game_trade_offer_limit=full_game_trade_offer_limit,
        )

    def set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        readonly_state = "disabled" if busy else "readonly"
        self.randomize_button.configure(state=state)
        self.analyze_button.configure(state=state)
        self._set_widget_state(self, state, readonly_state)
        self.randomize_button.configure(state=state)
        self.analyze_button.configure(state=state)
        self.cancel_button.configure(state=("normal" if busy else "disabled"))

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_progress(
        self,
        percent: float,
        stage: str,
        *,
        elapsed_seconds: float,
        eta_seconds: float | None,
    ) -> None:
        clamped = max(0.0, min(100.0, float(percent)))
        elapsed_text = self._format_duration(elapsed_seconds)
        eta_text = "--:--" if eta_seconds is None else self._format_duration(eta_seconds)
        self.progress_value_var.set(clamped)
        self.progress_detail_var.set(
            f"{stage} • {clamped:5.1f}% • elapsed {elapsed_text} • ETA {eta_text}"
        )

    def reset_progress(self) -> None:
        self.set_progress(0.0, "Idle", elapsed_seconds=0.0, eta_seconds=None)

    def is_knowledge_test_enabled(self) -> bool:
        return bool(self.knowledge_test_enabled_var.get())

    def _handle_cancel(self) -> None:
        if self._on_cancel:
            self._on_cancel()

    def _on_mode_change(self, _event: tk.Event) -> None:
        self._refresh_mode_specific_controls()

    def _refresh_mode_specific_controls(self) -> None:
        description = get_mode_description(self.mode_var.get())
        self.mode_description_var.set(description.to_ui_text())
        self._update_mode_card(description)
        try:
            mode = AnalysisMode(self.mode_var.get())
        except ValueError:
            mode = None

        show_tuning = True
        show_mc = bool(
            mode is AnalysisMode.MONTE_CARLO
            or (mode is AnalysisMode.HYBRID_OPENING and self.hybrid_include_mc_var.get())
        )
        show_phase = bool(mode in {AnalysisMode.PHASE_ROLLOUT_MC, AnalysisMode.HYBRID_OPENING, AnalysisMode.FULL_GAME})
        show_mcts = bool(mode in {AnalysisMode.MCTS_LITE_OPENING, AnalysisMode.HYBRID_OPENING, AnalysisMode.FULL_GAME})
        show_hybrid = bool(mode is AnalysisMode.HYBRID_OPENING)

        self._set_section_visibility("tuning", show_tuning)
        self._set_section_visibility("monte_carlo", show_mc)
        self._set_section_visibility("phase", show_phase)
        self._set_section_visibility("mcts", show_mcts)
        self._set_section_visibility("hybrid", show_hybrid)

    def _set_widget_state(self, root: tk.Widget, state: str, readonly_state: str) -> None:
        for child in root.winfo_children():
            if isinstance(child, ttk.Combobox):
                child.configure(state=readonly_state)
            elif isinstance(child, ttk.Entry):
                child.configure(state=state)
            elif isinstance(child, ttk.Checkbutton):
                child.configure(state=state)
            elif isinstance(child, ttk.Button):
                child.configure(state=state)
            elif isinstance(child, tk.Scale):
                child.configure(state=state)
            self._set_widget_state(child, state, readonly_state)

    def _register_collapsible_section(self, key: str, button: ttk.Button, frame: tk.Widget) -> None:
        self._section_toggle_buttons[key] = button
        self._section_frames[key] = frame
        button.configure(text=self._make_section_toggle_label(key))

    def _make_section_toggle_label(self, key: str) -> str:
        title = self._section_titles.get(key, key.title())
        collapsed = self._collapsed_sections.get(key, False)
        return f"{'▶' if collapsed else '▼'} {title}"

    def _toggle_section(self, key: str) -> None:
        if key not in self._collapsed_sections:
            return
        self._collapsed_sections[key] = not self._collapsed_sections[key]
        self._refresh_mode_specific_controls()

    def _set_section_visibility(self, key: str, visible: bool) -> None:
        button = self._section_toggle_buttons.get(key)
        frame = self._section_frames.get(key)
        if button is None or frame is None:
            return

        button.configure(text=self._make_section_toggle_label(key))
        if not visible:
            button.grid_remove()
            frame.grid_remove()
            return

        button.grid()
        if self._collapsed_sections.get(key, False):
            frame.grid_remove()
        else:
            frame.grid()

    def _handle_knowledge_toggle(self) -> None:
        if self._on_knowledge_test_toggle:
            self._on_knowledge_test_toggle(self.is_knowledge_test_enabled())

    def get_selected_theme(self) -> str:
        return str(self.theme_var.get()).strip().lower()

    def set_selected_theme(self, theme_key: str) -> None:
        self.theme_var.set(theme_key)

    def _handle_theme_change(self, _event: tk.Event) -> None:
        if self._on_theme_changed:
            self._on_theme_changed(self.get_selected_theme())

    def get_selected_ui_scale(self) -> float:
        raw = str(self.ui_scale_var.get()).strip().replace("%", "")
        try:
            value = float(raw)
        except ValueError:
            value = 100.0
        return max(70.0, min(220.0, value)) / 100.0

    def set_selected_ui_scale(self, scale_factor: float) -> None:
        percent = int(round(max(0.7, min(2.2, float(scale_factor))) * 100))
        self.ui_scale_var.set(f"{percent}%")

    def _handle_ui_scale_change(self, _event: tk.Event) -> None:
        if self._on_ui_scale_changed:
            self._on_ui_scale_changed(self.get_selected_ui_scale())

    def _guard_combobox_wheel_change(self, combobox: ttk.Combobox) -> None:
        combobox.bind("<MouseWheel>", self._block_mousewheel_change)
        combobox.bind("<Button-4>", self._block_mousewheel_change)
        combobox.bind("<Button-5>", self._block_mousewheel_change)

    def _apply_wheel_guard_to_comboboxes(self, root: tk.Widget) -> None:
        for child in root.winfo_children():
            if isinstance(child, ttk.Combobox):
                self._guard_combobox_wheel_change(child)
            self._apply_wheel_guard_to_comboboxes(child)

    def _block_mousewheel_change(self, event: tk.Event) -> str:
        if self._on_scroll_request:
            self._on_scroll_request(event)
        return "break"

    def _on_tuning_changed(self, _event: tk.Event) -> None:
        self._apply_recommended_tuning()

    def _on_tuning_scale_changed(self, _value: str) -> None:
        self._apply_recommended_tuning()

    def _apply_recommended_tuning(self) -> None:
        profile = str(self.tuning_profile_var.get()).strip().lower()
        effort = max(1, min(5, int(self.tuning_effort_var.get())))
        player_count = max(2, min(4, int(self.player_count_var.get())))
        mode = AnalysisMode(self.mode_var.get())

        profile_multiplier = {
            "speed": 0.70,
            "balanced": 1.00,
            "quality": 1.40,
            "exhaustive": 2.00,
        }.get(profile, 1.00)
        effort_multiplier = 0.70 + 0.20 * effort
        player_multiplier = 1.0 + 0.10 * (player_count - 2)
        scale = profile_multiplier * effort_multiplier * player_multiplier

        self.mc_iterations_var.set(self._round_to_step(18_000 * scale, 1_000))
        self.mc_rolls_var.set(self._round_to_step(40 * (0.80 + 0.20 * scale), 2))
        self.phase_rollout_count_var.set(self._round_to_step(120 * scale, 6))
        self.phase_horizon_var.set(self._round_to_step(14 * (0.85 + 0.20 * scale), 1))
        self.mcts_iterations_var.set(self._round_to_step(90 * scale, 5))
        self.mcts_max_plies_var.set(self._round_to_step(18 * (0.90 + 0.15 * scale), 1))
        self.mcts_settlement_candidates_var.set(
            self._clamp_int(int(round(6 + effort + (1 if profile in ("quality", "exhaustive") else 0))), 5, 18)
        )
        self.mcts_road_candidates_var.set(self._clamp_int(2 + effort // 2, 2, 5))
        self.mcts_exploration_c_var.set(round(self._clamp_float(1.00 + 0.06 * effort, 1.00, 1.50), 2))
        self.mcts_opponent_block_weight_var.set(
            round(self._clamp_float(0.22 + 0.05 * effort, 0.10, 0.70), 2)
        )

        if profile == "speed":
            self.hybrid_include_mc_var.set(False)
            self.hybrid_weight_heuristic_var.set(0.36)
            self.hybrid_weight_phase_var.set(0.32)
            self.hybrid_weight_mcts_var.set(0.32)
            self.hybrid_weight_mc_var.set(0.12)
        elif profile == "balanced":
            self.hybrid_include_mc_var.set(False)
            self.hybrid_weight_heuristic_var.set(0.24)
            self.hybrid_weight_phase_var.set(0.32)
            self.hybrid_weight_mcts_var.set(0.44)
            self.hybrid_weight_mc_var.set(0.18)
        elif profile == "quality":
            self.hybrid_include_mc_var.set(True)
            self.hybrid_weight_heuristic_var.set(0.18)
            self.hybrid_weight_phase_var.set(0.34)
            self.hybrid_weight_mcts_var.set(0.48)
            self.hybrid_weight_mc_var.set(0.16)
        else:
            self.hybrid_include_mc_var.set(True)
            self.hybrid_weight_heuristic_var.set(0.15)
            self.hybrid_weight_phase_var.set(0.32)
            self.hybrid_weight_mcts_var.set(0.53)
            self.hybrid_weight_mc_var.set(0.20)

        self.hybrid_stability_weight_var.set(round(self._clamp_float(0.14 + 0.04 * effort, 0.10, 0.40), 2))
        self.parallel_workers_var.set(0)
        self._refresh_mode_specific_controls()
        self.set_status(f"Applied {profile} profile (effort {effort}) for {mode.value}.")

    @staticmethod
    def _round_to_step(value: float, step: int) -> int:
        return max(step, int(round(value / step)) * step)

    @staticmethod
    def _clamp_int(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _clamp_float(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def apply_theme(self, theme: UiTheme) -> None:
        self._card_fade_start = theme.card_text_start
        self._card_fade_end = theme.card_text_end
        self.mode_card_frame.configure(bg=theme.card_bg, highlightbackground=theme.card_border)
        self.theme_row.configure(bg=theme.card_bg)
        self.theme_label.configure(bg=theme.card_bg, fg=theme.card_text_end)
        self.ui_scale_label.configure(bg=theme.card_bg, fg=theme.card_text_end)
        for widget in self._card_text_widgets:
            widget.configure(bg=theme.card_bg)
        self._update_mode_card(get_mode_description(self.mode_var.get()))

        try:
            self.tuning_effort_scale.configure(
                bg=theme.panel_bg,
                fg=theme.panel_fg,
                highlightbackground=theme.panel_bg,
                troughcolor=theme.heading_bg,
                activebackground=theme.selection_bg,
            )
        except tk.TclError:
            pass

    def _update_mode_card(self, description) -> None:
        mode_key = str(self.mode_var.get())
        title = self._card_title_lookup.get(mode_key, "Custom Analysis Study")
        self.mode_card_heading.configure(text=f"Gallery Placard — {title}")
        self.mode_card_summary.configure(text=f"Summary: {description.summary}")
        self.mode_card_reliability.configure(text=f"Reliability: {description.reliability}")
        self.mode_card_speed.configure(text=f"Speed: {description.speed}")
        self.mode_card_best_use.configure(text=f"Best use: {description.best_use}")
        self._animate_mode_card_fade()

    def _animate_mode_card_fade(self) -> None:
        if self._card_fade_after_id is not None:
            self.after_cancel(self._card_fade_after_id)
            self._card_fade_after_id = None

        start = self._hex_to_rgb(self._card_fade_start)
        end = self._hex_to_rgb(self._card_fade_end)
        steps = 10
        delay_ms = 28

        def step(index: int) -> None:
            ratio = index / steps
            color = self._interpolate_color(start, end, ratio)
            for widget in self._card_text_widgets:
                widget.configure(fg=color)
            if index < steps:
                self._card_fade_after_id = self.after(delay_ms, lambda: step(index + 1))
            else:
                self._card_fade_after_id = None

        step(0)

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple[int, int, int]:
        value = color.strip().lstrip("#")
        if len(value) != 6:
            return (179, 161, 132)
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))

    @staticmethod
    def _interpolate_color(
        start_rgb: tuple[int, int, int],
        end_rgb: tuple[int, int, int],
        ratio: float,
    ) -> str:
        ratio = max(0.0, min(1.0, ratio))
        red = int(round(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio))
        green = int(round(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio))
        blue = int(round(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio))
        return f"#{red:02X}{green:02X}{blue:02X}"


class ResultsPanel(ttk.LabelFrame):
    def __init__(
        self,
        master: tk.Widget,
        *,
        on_vertex_selected: Optional[Callable[[int | None], None]] = None,
        on_toggle_focus: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(master, text="Analysis Output", padding=10)
        self._on_vertex_selected = on_vertex_selected
        self._on_toggle_focus = on_toggle_focus
        self._vertex_row_lookup: dict[int, str] = {}
        self._vertex_score_lookup = {}
        self._sensitivity_lookup = {}
        self._selected_vertex_id: int | None = None
        self._sort_column = "rank"
        self._sort_reverse = False
        self._pinned_vertex_ids: list[int] = []
        self._focus_mode = False
        self._build()

    def _build(self) -> None:
        header_row = ttk.Frame(self)
        header_row.grid(row=0, column=0, sticky="ew")
        header_row.columnconfigure(0, weight=1)

        self.summary_var = tk.StringVar(value="No analysis yet.")
        self.summary_label = ttk.Label(header_row, textvariable=self.summary_var)
        self.summary_label.grid(row=0, column=0, sticky="w")
        self.focus_button = ttk.Button(header_row, text="Focus Results", command=self._handle_focus_toggle)
        self.focus_button.grid(row=0, column=1, sticky="e")
        self.knowledge_test_var = tk.StringVar(value="Knowledge test disabled.")
        self.knowledge_label = ttk.Label(
            self,
            textvariable=self.knowledge_test_var,
            foreground="#5D2E8C",
            wraplength=420,
            justify="left",
        )
        self.knowledge_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.accessibility_hint_var = tk.StringVar(
            value="Keyboard: Tab moves focus. Use Up/Down arrows in ranking to move between vertices."
        )
        self.accessibility_hint_label = ttk.Label(
            self,
            textvariable=self.accessibility_hint_var,
            wraplength=430,
            justify="left",
        )
        self.accessibility_hint_label.grid(row=2, column=0, sticky="w", pady=(2, 0))

        details_container = ttk.Frame(self)
        details_container.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        self.details_canvas = tk.Canvas(details_container, highlightthickness=0)
        self.details_scrollbar = ttk.Scrollbar(details_container, orient="vertical", command=self.details_canvas.yview)
        self.details_canvas.configure(yscrollcommand=self.details_scrollbar.set)
        self.details_canvas.pack(side="left", fill="both", expand=True)
        self.details_scrollbar.pack(side="left", fill="y")

        self.details_inner = ttk.Frame(self.details_canvas)
        self.details_canvas_window_id = self.details_canvas.create_window((0, 0), window=self.details_inner, anchor="nw")
        self.details_inner.bind("<Configure>", self._on_details_inner_configure)
        self.details_canvas.bind("<Configure>", self._on_details_canvas_configure)
        self.details_canvas.bind("<MouseWheel>", self._on_details_mouse_wheel)
        self.details_canvas.bind("<Button-4>", self._on_details_mouse_wheel_linux)
        self.details_canvas.bind("<Button-5>", self._on_details_mouse_wheel_linux)

        columns = (
            "rank",
            "vertex",
            "score",
            "stability",
            "yield",
            "diversity",
            "port",
            "risk",
            "synergy",
            "frontier",
            "best_path",
            "tempo",
            "recipe",
            "fragility",
            "port_conv",
            "robber",
        )
        self.ranking_tree = ttk.Treeview(self.details_inner, columns=columns, show="headings", height=14)
        headings = {
            "rank": "Rank",
            "vertex": "Vertex",
            "score": "Score",
            "stability": "Sensitivity",
            "yield": "Yield",
            "diversity": "Diversity",
            "port": "Port",
            "risk": "Risk",
            "synergy": "Synergy",
            "frontier": "Frontier",
            "best_path": "Best Path",
            "tempo": "Tempo",
            "recipe": "Recipe",
            "fragility": "Fragility",
            "port_conv": "Port Conv",
            "robber": "Robber Risk",
        }
        widths = {
            "rank": 50,
            "vertex": 60,
            "score": 70,
            "stability": 92,
            "yield": 70,
            "diversity": 70,
            "port": 60,
            "risk": 60,
            "synergy": 70,
            "frontier": 70,
            "best_path": 70,
            "tempo": 70,
            "recipe": 70,
            "fragility": 70,
            "port_conv": 70,
            "robber": 70,
        }
        for column in columns:
            self.ranking_tree.heading(column, text=headings[column], command=lambda c=column: self._sort_tree(c))
            self.ranking_tree.column(column, width=widths[column], anchor="center")

        self.ranking_tree.tag_configure("top", background="#E8F3FF")
        self.ranking_tree.tag_configure("stable", foreground="#1B5E20")
        self.ranking_tree.tag_configure("watch", foreground="#8A6D1A")
        self.ranking_tree.tag_configure("volatile", foreground="#8B1A1A")
        self.ranking_tree.bind("<<TreeviewSelect>>", self._handle_rank_selection)
        self.ranking_tree.bind("<Up>", lambda _event: self._navigate_ranking(-1))
        self.ranking_tree.bind("<Down>", lambda _event: self._navigate_ranking(1))
        self.ranking_tree.bind("<Home>", lambda _event: self._navigate_ranking_to_edge(first=True))
        self.ranking_tree.bind("<End>", lambda _event: self._navigate_ranking_to_edge(first=False))
        self.ranking_tree.configure(takefocus=True)
        self.ranking_tree.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        pin_controls = ttk.Frame(self.details_inner)
        pin_controls.grid(row=1, column=0, sticky="ew")
        ttk.Label(pin_controls, text="Pinned comparison (up to 3)").pack(side="left")
        ttk.Button(pin_controls, text="Pin selected", command=self._pin_selected_vertex).pack(side="right")
        ttk.Button(pin_controls, text="Unpin", command=self._unpin_selected_vertex).pack(side="right", padx=(0, 6))
        ttk.Button(pin_controls, text="Clear pins", command=self._clear_pins).pack(side="right", padx=(0, 6))

        self.pinned_tree = ttk.Treeview(self.details_inner, columns=("metric", "p1", "p2", "p3"), show="headings", height=7)
        self.pinned_tree.heading("metric", text="Metric")
        self.pinned_tree.heading("p1", text="Vertex -")
        self.pinned_tree.heading("p2", text="Vertex -")
        self.pinned_tree.heading("p3", text="Vertex -")
        self.pinned_tree.column("metric", width=118, anchor="w")
        self.pinned_tree.column("p1", width=90, anchor="center")
        self.pinned_tree.column("p2", width=90, anchor="center")
        self.pinned_tree.column("p3", width=90, anchor="center")
        self.pinned_tree.grid(row=2, column=0, sticky="nsew", pady=(6, 8))

        ttk.Label(self.details_inner, text="Why this rank?").grid(row=3, column=0, sticky="w")
        self.why_rank_text = tk.Text(self.details_inner, height=8, width=64, wrap="word", state="disabled")
        self.why_rank_text.grid(row=4, column=0, sticky="nsew", pady=(6, 8))

        ttk.Label(self.details_inner, text="Draft-aware pick sequence").grid(row=5, column=0, sticky="w")
        self.sequence_text = tk.Text(self.details_inner, height=8, width=64, wrap="word", state="disabled")
        self.sequence_text.grid(row=6, column=0, sticky="nsew", pady=(6, 0))

        ttk.Label(self.details_inner, text="MCTS-lite explainer").grid(row=7, column=0, sticky="w", pady=(8, 0))
        self.explainer_text = tk.Text(self.details_inner, height=8, width=64, wrap="word", state="disabled")
        self.explainer_text.grid(row=8, column=0, sticky="nsew", pady=(6, 0))

        self.details_inner.columnconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

    def set_focus_mode(self, enabled: bool) -> None:
        self._focus_mode = bool(enabled)
        self.focus_button.configure(text="Exit Focus" if self._focus_mode else "Focus Results")

    def _handle_focus_toggle(self) -> None:
        if self._on_toggle_focus:
            self._on_toggle_focus()

    def _on_details_inner_configure(self, _event: tk.Event) -> None:
        self.details_canvas.configure(scrollregion=self.details_canvas.bbox("all"))

    def _on_details_canvas_configure(self, event: tk.Event) -> None:
        self.details_canvas.itemconfigure(self.details_canvas_window_id, width=event.width)

    def _on_details_mouse_wheel(self, event: tk.Event) -> None:
        if event.delta == 0:
            return
        self.details_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_details_mouse_wheel_linux(self, event: tk.Event) -> None:
        if event.num == 4:
            self.details_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.details_canvas.yview_scroll(1, "units")

    def apply_theme(self, theme: UiTheme, *, scale_factor: float = 1.0) -> None:
        scaled_data_font = (
            theme.font_data[0],
            max(8, int(round(theme.font_data[1] * scale_factor))),
            theme.font_data[2],
        )
        self.configure(style="TLabelframe")
        self.knowledge_label.configure(foreground=theme.accent_fg)
        self.accessibility_hint_label.configure(foreground=theme.muted_fg)
        self.ranking_tree.tag_configure("top", background=theme.top_row_bg)
        self.ranking_tree.tag_configure("stable", foreground=theme.stable_fg)
        self.ranking_tree.tag_configure("watch", foreground=theme.watch_fg)
        self.ranking_tree.tag_configure("volatile", foreground=theme.volatile_fg)

        for text_widget in (self.why_rank_text, self.sequence_text, self.explainer_text):
            text_widget.configure(
                bg=theme.text_bg,
                fg=theme.text_fg,
                insertbackground=theme.text_fg,
                selectbackground=theme.selection_bg,
                selectforeground=theme.selection_fg,
                font=scaled_data_font,
            )
        self.details_canvas.configure(bg=theme.panel_bg)

    def clear(self) -> None:
        self._vertex_row_lookup.clear()
        self._vertex_score_lookup = {}
        self._sensitivity_lookup = {}
        self._selected_vertex_id = None
        self._sort_column = "rank"
        self._sort_reverse = False
        self._pinned_vertex_ids = []
        self.ranking_tree.selection_remove(self.ranking_tree.selection())
        for item in self.ranking_tree.get_children():
            self.ranking_tree.delete(item)
        for item in self.pinned_tree.get_children():
            self.pinned_tree.delete(item)
        self._update_pinned_headings()
        self._render_pinned_comparison()
        self.sequence_text.configure(state="normal")
        self.sequence_text.delete("1.0", tk.END)
        self.sequence_text.configure(state="disabled")
        self.why_rank_text.configure(state="normal")
        self.why_rank_text.delete("1.0", tk.END)
        self.why_rank_text.insert("1.0", "Select a ranked vertex to see contribution/penalty explainability.")
        self.why_rank_text.configure(state="disabled")
        self.explainer_text.configure(state="normal")
        self.explainer_text.delete("1.0", tk.END)
        self.explainer_text.configure(state="disabled")
        self.summary_var.set("No analysis yet.")
        if self._on_vertex_selected:
            self._on_vertex_selected(None)

    def update_knowledge_test_status(
        self,
        *,
        enabled: bool,
        selected_vertices: list[int] | None = None,
        feedback: str | None = None,
    ) -> None:
        if not enabled:
            self.knowledge_test_var.set("Knowledge test disabled.")
            return

        picks = selected_vertices or []
        picks_text = ", ".join(str(vertex_id) for vertex_id in picks) or "none yet"
        line = f"Knowledge test picks ({len(picks)}/4): {picks_text}."
        if feedback:
            line = f"{line} {feedback}"
        self.knowledge_test_var.set(line)

    def update(self, result: AnalysisResult, top_count: int) -> None:
        self.clear()
        shown = result.global_ranking
        self._vertex_score_lookup = {score.vertex_id: score for score in shown}
        self._sensitivity_lookup = compute_sensitivity_badges(shown)
        self.summary_var.set(
            f"Top {top_count} recommended vertices highlighted on board. "
            f"Showing {len(shown)} legal vertices in ranking."
        )
        if result.full_game_summary is not None:
            win_rates = ", ".join(
                f"P{player_id} {rate * 100:.1f}%"
                for player_id, rate in sorted(result.full_game_summary.player_win_rates.items())
            )
            self.summary_var.set(
                f"{self.summary_var.get()} Full-game rollouts: {result.full_game_summary.rollout_count}. "
                f"Win rates: {win_rates}."
            )

        for rank, score in enumerate(shown, start=1):
            sensitivity = self._sensitivity_lookup.get(score.vertex_id)
            tag_parts = []
            if rank <= top_count:
                tag_parts.append("top")
            if sensitivity is not None:
                tag_parts.append(sensitivity.key)
            tags = tuple(tag_parts)
            row_id = self.ranking_tree.insert(
                "",
                "end",
                values=(
                    rank,
                    score.vertex_id,
                    f"{score.total_score:.2f}",
                    sensitivity.label if sensitivity is not None else "",
                    f"{score.expected_yield:.2f}",
                    f"{score.diversity_score:.2f}",
                    f"{score.port_score:.2f}",
                    f"{score.risk_penalty:.2f}",
                    f"{score.synergy_score:.2f}",
                    f"{score.frontier_score:.2f}",
                    f"{score.best_path_score:.2f}",
                    f"{score.tempo_score:.2f}",
                    f"{score.recipe_coverage_score:.2f}",
                    f"{score.fragility_penalty:.2f}",
                    f"{score.port_conversion_score:.2f}",
                    f"{score.robber_penalty:.2f}",
                ),
                tags=tags,
            )
            self._vertex_row_lookup[score.vertex_id] = row_id

        self._sort_tree("rank", preserve_direction=True)

        self.sequence_text.configure(state="normal")
        for pick in result.predicted_sequence:
            self.sequence_text.insert(
                tk.END,
                f"Turn {pick.turn_index:>2} | Player {pick.player_id} -> "
                f"Vertex {pick.vertex_id} (score {pick.score_snapshot.total_score:.2f})\n",
            )
        self.sequence_text.configure(state="disabled")

        self.explainer_text.configure(state="normal")
        if result.explain_lines:
            for line in result.explain_lines:
                self.explainer_text.insert(
                    tk.END,
                    f"Ply {line.ply_index:>2} | P{line.actor} | {line.action} | "
                    f"self={line.self_value:.2f} block={line.blocking_delta:.2f} "
                    f"uct={line.uct_value:.2f} visits={line.visits}\n",
                )
            if result.mcts_summary is not None:
                self.explainer_text.insert(
                    tk.END,
                    "\n"
                    f"Root visits: {result.mcts_summary.root_visits} | "
                    f"Best line: {result.mcts_summary.best_line_score:.2f} | "
                    f"Alt gap: {result.mcts_summary.alt_line_score_gap:.2f} | "
                    f"Runtime: {result.mcts_summary.runtime_ms:.0f}ms",
                )
        else:
            self.explainer_text.insert(
                tk.END,
                "No MCTS-lite explanation for this analysis mode.",
            )
        if result.full_game_summary is not None:
            self.explainer_text.insert(tk.END, "\n\nFull-game summary:\n")
            for player_id, rate in sorted(result.full_game_summary.player_win_rates.items()):
                eta = result.full_game_summary.expected_turns_to_victory.get(player_id)
                eta_text = f"{eta:.1f}" if eta is not None else "n/a"
                self.explainer_text.insert(
                    tk.END,
                    f"  P{player_id}: win {rate * 100:.1f}% | ETA {eta_text} turns\n",
                )
            self.explainer_text.insert(
                tk.END,
                f"  Avg game length: {result.full_game_summary.average_game_length_turns:.1f} turns",
            )
        self.explainer_text.configure(state="disabled")
        self._render_why_rank(None)

    def select_vertex(self, vertex_id: int | None) -> bool:
        if vertex_id is None:
            self._selected_vertex_id = None
            self.ranking_tree.selection_remove(self.ranking_tree.selection())
            self._render_why_rank(None)
            return False

        row_id = self._vertex_row_lookup.get(vertex_id)
        if row_id is None:
            self._selected_vertex_id = None
            self.ranking_tree.selection_remove(self.ranking_tree.selection())
            self._render_why_rank(None)
            return False

        self.ranking_tree.selection_set(row_id)
        self.ranking_tree.focus(row_id)
        self.ranking_tree.see(row_id)
        self._selected_vertex_id = vertex_id
        self._render_why_rank(vertex_id)
        return True

    def _navigate_ranking(self, delta: int) -> str:
        items = list(self.ranking_tree.get_children())
        if not items:
            return "break"

        selected = self.ranking_tree.selection()
        if not selected:
            target_index = 0 if delta >= 0 else len(items) - 1
        else:
            try:
                current_index = items.index(selected[0])
            except ValueError:
                current_index = 0
            target_index = max(0, min(len(items) - 1, current_index + delta))

        target_item = items[target_index]
        self.ranking_tree.selection_set(target_item)
        self.ranking_tree.focus(target_item)
        self.ranking_tree.see(target_item)
        self._handle_rank_selection()
        return "break"

    def _navigate_ranking_to_edge(self, *, first: bool) -> str:
        items = list(self.ranking_tree.get_children())
        if not items:
            return "break"
        target_item = items[0] if first else items[-1]
        self.ranking_tree.selection_set(target_item)
        self.ranking_tree.focus(target_item)
        self.ranking_tree.see(target_item)
        self._handle_rank_selection()
        return "break"

    def _sort_tree(self, column: str, preserve_direction: bool = False) -> None:
        if not preserve_direction:
            if self._sort_column == column:
                self._sort_reverse = not self._sort_reverse
            else:
                self._sort_column = column
                self._sort_reverse = column not in {"rank", "vertex"}

        items = list(self.ranking_tree.get_children())
        if not items:
            return

        def key_for(item_id: str):
            value = self.ranking_tree.set(item_id, column)
            if column in {"rank", "vertex"}:
                try:
                    return int(value)
                except ValueError:
                    return 0
            if column == "stability":
                priority = {"Stable": 3, "Watch": 2, "Volatile": 1}
                return priority.get(value, 0)
            try:
                return float(value)
            except ValueError:
                return value

        items.sort(key=key_for, reverse=self._sort_reverse)
        for index, item_id in enumerate(items):
            self.ranking_tree.move(item_id, "", index)

    def _pin_selected_vertex(self) -> None:
        if self._selected_vertex_id is None:
            return
        if self._selected_vertex_id in self._pinned_vertex_ids:
            return
        if len(self._pinned_vertex_ids) >= 3:
            return
        self._pinned_vertex_ids.append(self._selected_vertex_id)
        self._render_pinned_comparison()

    def _unpin_selected_vertex(self) -> None:
        if self._selected_vertex_id is None:
            return
        if self._selected_vertex_id not in self._pinned_vertex_ids:
            return
        self._pinned_vertex_ids.remove(self._selected_vertex_id)
        self._render_pinned_comparison()

    def _clear_pins(self) -> None:
        if not self._pinned_vertex_ids:
            return
        self._pinned_vertex_ids.clear()
        self._render_pinned_comparison()

    def _render_pinned_comparison(self) -> None:
        for item in self.pinned_tree.get_children():
            self.pinned_tree.delete(item)

        self._update_pinned_headings()
        if not self._pinned_vertex_ids:
            self.pinned_tree.insert("", "end", values=("No pinned vertices yet.", "", "", ""))
            return

        metric_rows = [
            ("Total", "total_score"),
            ("Yield", "expected_yield"),
            ("Diversity", "diversity_score"),
            ("Port", "port_score"),
            ("Risk", "risk_penalty"),
            ("Synergy", "synergy_score"),
            ("Frontier", "frontier_score"),
            ("BestPath", "best_path_score"),
            ("Tempo", "tempo_score"),
            ("Recipe", "recipe_coverage_score"),
            ("Fragility", "fragility_penalty"),
            ("PortConv", "port_conversion_score"),
            ("Robber", "robber_penalty"),
            ("Sensitivity", "__sensitivity"),
        ]

        pinned_scores = [self._vertex_score_lookup.get(vertex_id) for vertex_id in self._pinned_vertex_ids]
        while len(pinned_scores) < 3:
            pinned_scores.append(None)

        for label, metric in metric_rows:
            row_values: list[str] = [label]
            for score in pinned_scores:
                if score is None:
                    row_values.append("-")
                    continue
                if metric == "__sensitivity":
                    badge = self._sensitivity_lookup.get(score.vertex_id)
                    row_values.append(badge.label if badge is not None else "-")
                else:
                    row_values.append(f"{float(getattr(score, metric, 0.0)):.2f}")
            self.pinned_tree.insert("", "end", values=tuple(row_values))

    def _update_pinned_headings(self) -> None:
        labels = [f"Vertex {vertex_id}" for vertex_id in self._pinned_vertex_ids]
        while len(labels) < 3:
            labels.append("Vertex -")
        self.pinned_tree.heading("p1", text=labels[0])
        self.pinned_tree.heading("p2", text=labels[1])
        self.pinned_tree.heading("p3", text=labels[2])

    def _render_why_rank(self, vertex_id: int | None) -> None:
        self.why_rank_text.configure(state="normal")
        self.why_rank_text.delete("1.0", tk.END)
        if vertex_id is None:
            self.why_rank_text.insert("1.0", "Select a ranked vertex to see contribution/penalty explainability.")
        else:
            score = self._vertex_score_lookup.get(vertex_id)
            if score is None:
                self.why_rank_text.insert("1.0", f"Vertex {vertex_id} is not in the current ranking.")
            else:
                badge = self._sensitivity_lookup.get(vertex_id)
                self.why_rank_text.insert("1.0", explain_vertex_score(score, badge))
        self.why_rank_text.configure(state="disabled")

    def _handle_rank_selection(self, _event: tk.Event | None = None) -> None:
        selected = self.ranking_tree.selection()
        if not selected:
            self._selected_vertex_id = None
            self._render_why_rank(None)
            if self._on_vertex_selected:
                self._on_vertex_selected(None)
            return

        item_values = self.ranking_tree.item(selected[0], "values")
        if len(item_values) < 2:
            self._selected_vertex_id = None
            self._render_why_rank(None)
            if self._on_vertex_selected:
                self._on_vertex_selected(None)
            return

        try:
            vertex_id = int(item_values[1])
        except (TypeError, ValueError):
            self._selected_vertex_id = None
            self._render_why_rank(None)
            if self._on_vertex_selected:
                self._on_vertex_selected(None)
            return

        self._selected_vertex_id = vertex_id
        self._render_why_rank(vertex_id)
        if self._on_vertex_selected:
            self._on_vertex_selected(vertex_id)
