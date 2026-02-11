from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from catan_analyzer.analysis.knowledge_test import (
    KnowledgeTestEvaluation,
    evaluate_user_settlement_picks,
)
from catan_analyzer.analysis.runtime import AnalysisCancelled, AnalysisRuntime
from catan_analyzer.analysis.scoring import pip_value
from catan_analyzer.analysis.simulation import create_analyzer
from catan_analyzer.analysis.types import VertexScore
from catan_analyzer.domain.board import EdgeKey, Resource
from catan_analyzer.domain.randomizer import generate_randomized_board
from catan_analyzer.ui.board_canvas import BoardCanvas, ClickMode, PORT_SHORT_LABELS
from catan_analyzer.ui.panels import AnalyzerControls, ResultsPanel
from catan_analyzer.ui.themes import UiTheme, get_theme


class CatanAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Catan Analyzer")
        self.geometry("1450x860")
        self.minsize(1200, 760)

        self.board = generate_randomized_board()
        self._is_syncing_selection = False
        self._analysis_ran_for_current_board = False
        self._knowledge_test_picks: list[int] = []
        self._knowledge_test_feedback: str | None = None
        self._analysis_queue: queue.Queue[tuple[str, int, object]] = queue.Queue()
        self._analysis_worker: threading.Thread | None = None
        self._analysis_in_progress = False
        self._analysis_request_id = 0
        self._analysis_cancel_event: threading.Event | None = None
        self._analysis_started_at = 0.0
        self._last_ranking: list[VertexScore] = []
        self._user_roads: set[EdgeKey] = set()
        self._theme: UiTheme = get_theme("light")
        self._style = ttk.Style(self)
        self._results_focus_mode = False
        self._saved_sash_pos: int | None = None
        self._ui_scale_factor = 1.0

        self._build_layout()
        self.board_canvas.set_board(self.board)
        self.apply_theme(self._theme.key)
        self._refresh_knowledge_test_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)

        self.main_pane = ttk.Panedwindow(container, orient="horizontal")
        self.main_pane.pack(fill="both", expand=True)

        left_frame = ttk.Frame(self.main_pane)
        right_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(left_frame, weight=4)
        self.main_pane.add(right_frame, weight=2)
        self.after(80, self._init_split_pane)

        # ── Click-mode toolbar ──────────────────────────────────
        self._click_mode_var = tk.StringVar(value=ClickMode.INSPECT.name)
        toolbar = ttk.LabelFrame(left_frame, text="Click Mode", padding=(8, 4))
        toolbar.pack(fill="x", pady=(0, 4))

        mode_info = [
            (ClickMode.INSPECT, "🔍 Inspect"),
            (ClickMode.KNOWLEDGE_TEST_PICK, "🧠 Knowledge Test Pick"),
            (ClickMode.COMPARE_TWO_VERTICES, "⚖️ Compare Two Vertices"),
            (ClickMode.PLACE_ROADS, "🛣️ Place Roads"),
        ]
        for mode, label in mode_info:
            rb = ttk.Radiobutton(
                toolbar,
                text=label,
                value=mode.name,
                variable=self._click_mode_var,
                command=self._on_click_mode_changed,
            )
            rb.pack(side="left", padx=(0, 14))

        self._compare_status_var = tk.StringVar(value="")
        self.compare_status_label = ttk.Label(toolbar, textvariable=self._compare_status_var, foreground="#009688")
        self.compare_status_label.pack(
            side="left", padx=(10, 0)
        )

        # ── Board canvas ────────────────────────────────────────
        self.board_canvas = BoardCanvas(
            left_frame,
            on_vertex_selected=self.on_board_vertex_selected,
            on_compare_pair=self._on_compare_pair,
            on_road_selected=self.on_board_road_selected,
        )
        self.board_canvas.pack(fill="both", expand=True)

        # ── Compare tray (shown below canvas when in compare mode) ─
        self._compare_frame = ttk.LabelFrame(left_frame, text="Vertex Comparison", padding=8)
        self._compare_text = tk.Text(self._compare_frame, height=6, width=80, wrap="word", state="disabled",
                                      font=("Segoe UI", 9))
        self._compare_text.pack(fill="both", expand=True)
        # Hidden by default
        self._compare_frame.pack_forget()

        # ── Right-side controls + results ───────────────────────
        self.right_notebook = ttk.Notebook(right_frame)
        self.right_notebook.pack(fill="both", expand=True)

        self.controls_tab = ttk.Frame(self.right_notebook)
        self.results_tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.controls_tab, text="Controls")
        self.right_notebook.add(self.results_tab, text="Results")

        controls_scroll_container = ttk.Frame(self.controls_tab)
        controls_scroll_container.pack(fill="both", expand=True)

        self.controls_canvas = tk.Canvas(controls_scroll_container, highlightthickness=0)
        self.controls_scrollbar = ttk.Scrollbar(
            controls_scroll_container,
            orient="vertical",
            command=self.controls_canvas.yview,
        )
        self.controls_canvas.configure(yscrollcommand=self.controls_scrollbar.set)
        self.controls_canvas.pack(side="left", fill="both", expand=True)
        self.controls_scrollbar.pack(side="left", fill="y")

        self.controls_inner = ttk.Frame(self.controls_canvas)
        self.controls_canvas_window_id = self.controls_canvas.create_window(
            (0, 0),
            window=self.controls_inner,
            anchor="nw",
        )
        self.controls_inner.bind("<Configure>", self._on_controls_inner_configure)
        self.controls_canvas.bind("<Configure>", self._on_controls_canvas_configure)
        self.controls_canvas.bind("<MouseWheel>", self._on_controls_mouse_wheel)
        self.controls_canvas.bind("<Button-4>", self._on_controls_mouse_wheel_linux)
        self.controls_canvas.bind("<Button-5>", self._on_controls_mouse_wheel_linux)

        self.controls = AnalyzerControls(
            self.controls_inner,
            on_randomize=self.on_randomize,
            on_analyze=self.on_analyze,
            on_cancel=self.on_cancel_analysis,
            on_knowledge_test_toggle=self.on_knowledge_test_toggle,
            on_theme_changed=self.on_theme_changed,
            on_ui_scale_changed=self.on_ui_scale_changed,
            on_scroll_request=self._forward_controls_scroll_event,
        )
        self.controls.pack(fill="x")

        self.results_panel = ResultsPanel(
            self.results_tab,
            on_vertex_selected=self.on_vertex_selected,
            on_toggle_focus=self.toggle_results_focus,
        )
        self.results_panel.pack(fill="both", expand=True)

    def _init_split_pane(self) -> None:
        try:
            total_width = self.main_pane.winfo_width()
            if total_width <= 0:
                return
            desired_right = 500
            sash_pos = max(620, total_width - desired_right)
            self.main_pane.sashpos(0, sash_pos)
        except tk.TclError:
            return

    def toggle_results_focus(self) -> None:
        if not self._results_focus_mode:
            try:
                self._saved_sash_pos = self.main_pane.sashpos(0)
                pane_width = max(1, self.main_pane.winfo_width())
                self.main_pane.sashpos(0, max(8, int(pane_width * 0.05)))
            except tk.TclError:
                pass
            self._results_focus_mode = True
            self.right_notebook.select(self.results_tab)
        else:
            try:
                if self._saved_sash_pos is not None:
                    self.main_pane.sashpos(0, self._saved_sash_pos)
            except tk.TclError:
                pass
            self._results_focus_mode = False

        self.results_panel.set_focus_mode(self._results_focus_mode)

    # ── click mode handling ─────────────────────────────────────

    def _on_click_mode_changed(self) -> None:
        mode_name = self._click_mode_var.get()
        mode = ClickMode[mode_name]
        self.board_canvas.set_click_mode(mode)

        if mode == ClickMode.COMPARE_TWO_VERTICES:
            self._compare_frame.pack(fill="x", pady=(4, 0))
            self._compare_status_var.set("Click two vertices to compare.")
        else:
            self._compare_frame.pack_forget()
            self._compare_status_var.set("")

        # Sync knowledge-test checkbox with mode
        if mode == ClickMode.KNOWLEDGE_TEST_PICK:
            if not self.controls.is_knowledge_test_enabled():
                self.controls.knowledge_test_enabled_var.set(True)
                self.on_knowledge_test_toggle(True)
        elif mode == ClickMode.INSPECT:
            if self.controls.is_knowledge_test_enabled():
                self.controls.knowledge_test_enabled_var.set(False)
                self.on_knowledge_test_toggle(False)
        elif mode == ClickMode.PLACE_ROADS:
            self.controls.set_status(
                "Road mode: click an endpoint to set anchor, then click an adjacent endpoint (or edge line) to place a road."
            )

    def _on_controls_inner_configure(self, _event: tk.Event) -> None:
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_controls_canvas_configure(self, event: tk.Event) -> None:
        self.controls_canvas.itemconfigure(self.controls_canvas_window_id, width=event.width)

    def _on_controls_mouse_wheel(self, event: tk.Event) -> None:
        if event.delta == 0:
            return
        self.controls_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_controls_mouse_wheel_linux(self, event: tk.Event) -> None:
        if event.num == 4:
            self.controls_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.controls_canvas.yview_scroll(1, "units")

    def _forward_controls_scroll_event(self, event: tk.Event) -> None:
        if hasattr(event, "delta") and event.delta not in (None, 0):
            self._on_controls_mouse_wheel(event)
            return
        if hasattr(event, "num"):
            self._on_controls_mouse_wheel_linux(event)

    def _on_compare_pair(self, a: int, b: int) -> None:
        """Called when two vertices have been selected in compare mode."""
        lines: list[str] = []
        lines.append(f"{'':8}{'Vertex ' + str(a):>18}  {'Vertex ' + str(b):>18}")
        lines.append("─" * 50)

        # Gather info for both vertices
        for vertex_id, label in [(a, f"V{a}"), (b, f"V{b}")]:
            tiles = self.board.vertex_adjacent_tiles(vertex_id)
            res = ", ".join(
                f"{t.resource.value.title()}" for t in tiles if t.resource is not Resource.DESERT
            )
            pips = sum(pip_value(t.token_number) for t in tiles)
            port = self.board.vertices[vertex_id].port_type
            port_str = PORT_SHORT_LABELS.get(port, "none") if port else "none"
            lines.append(f"{label:8} Resources: {res}")
            lines.append(f"{'':8} Pips: {pips}   Port: {port_str}")

            rank_info = self.board_canvas._score_lookup.get(vertex_id)
            if rank_info:
                rank, score = rank_info
                lines.append(
                    f"{'':8} Rank #{rank}  Score {score.total_score:.2f}  "
                    f"Yield {score.expected_yield:.2f}  Div {score.diversity_score:.2f}  "
                    f"Port {score.port_score:.2f}  Risk {score.risk_penalty:.2f}"
                )
            else:
                lines.append(f"{'':8} Rank: (run analysis)")
            lines.append("")

        # Delta summary if both have scores
        a_info = self.board_canvas._score_lookup.get(a)
        b_info = self.board_canvas._score_lookup.get(b)
        if a_info and b_info:
            _, sa = a_info
            _, sb = b_info
            delta = sa.total_score - sb.total_score
            better = a if delta > 0 else b
            lines.append(f"Score delta: {abs(delta):.2f} in favour of Vertex {better}.")

        self._compare_text.configure(state="normal")
        self._compare_text.delete("1.0", tk.END)
        self._compare_text.insert(tk.END, "\n".join(lines))
        self._compare_text.configure(state="disabled")
        self._compare_status_var.set(f"Comparing V{a} vs V{b}. Click a pick to deselect.")

    def on_theme_changed(self, theme_key: str) -> None:
        self.apply_theme(theme_key)

    def on_ui_scale_changed(self, scale_factor: float) -> None:
        self._ui_scale_factor = max(0.7, min(2.2, float(scale_factor)))
        self.apply_theme(self._theme.key)

    def _scaled_font(self, font_tuple: tuple[str, int, str]) -> tuple[str, int, str]:
        family, size, weight = font_tuple
        scaled_size = max(8, int(round(size * self._ui_scale_factor)))
        return (family, scaled_size, weight)

    def apply_theme(self, theme_key: str) -> None:
        theme = get_theme(theme_key)
        self._theme = theme
        self.controls.set_selected_theme(theme.key)
        self.controls.set_selected_ui_scale(self._ui_scale_factor)

        label_font = self._scaled_font(theme.font_label)
        section_font = self._scaled_font(theme.font_section)
        data_font = self._scaled_font(theme.font_data)
        heading_font = self._scaled_font(theme.font_heading)

        try:
            self._style.theme_use("clam")
        except tk.TclError:
            pass

        self.configure(bg=theme.window_bg)
        self._style.configure(".", background=theme.panel_bg, foreground=theme.panel_fg, font=label_font)
        self._style.configure("TFrame", background=theme.panel_bg)
        self._style.configure("TPanedwindow", background=theme.window_bg)
        self._style.configure("TNotebook", background=theme.panel_bg, bordercolor=theme.border)
        self._style.configure(
            "TNotebook.Tab",
            background=theme.heading_bg,
            foreground=theme.heading_fg,
            font=section_font,
            padding=(10, 6),
        )
        self._style.map(
            "TNotebook.Tab",
            background=[("selected", theme.button_active_bg), ("active", theme.button_bg)],
            foreground=[("selected", theme.heading_fg), ("active", theme.heading_fg)],
        )
        self._style.configure(
            "TLabelframe",
            background=theme.panel_bg,
            foreground=theme.panel_fg,
            bordercolor=theme.border,
            relief="solid",
        )
        self._style.configure(
            "TLabelframe.Label",
            background=theme.panel_bg,
            foreground=theme.panel_fg,
            font=section_font,
        )
        self._style.configure("TLabel", background=theme.panel_bg, foreground=theme.panel_fg, font=label_font)
        self._style.configure(
            "TButton",
            background=theme.button_bg,
            foreground=theme.button_fg,
            font=label_font,
            padding=(8, 4),
            bordercolor=theme.border,
        )
        self._style.map(
            "TButton",
            background=[("active", theme.button_active_bg), ("pressed", theme.button_active_bg)],
            foreground=[("disabled", theme.muted_fg)],
        )
        self._style.configure(
            "TEntry",
            fieldbackground=theme.input_bg,
            foreground=theme.input_fg,
            insertcolor=theme.input_fg,
        )
        self._style.configure(
            "TCombobox",
            fieldbackground=theme.input_bg,
            foreground=theme.input_fg,
            background=theme.input_bg,
            arrowcolor=theme.panel_fg,
        )
        self._style.map(
            "TCombobox",
            fieldbackground=[("readonly", theme.input_readonly_bg)],
            foreground=[("readonly", theme.input_fg)],
            selectbackground=[("readonly", theme.selection_bg)],
            selectforeground=[("readonly", theme.selection_fg)],
        )
        self._style.configure(
            "Treeview",
            background=theme.text_bg,
            fieldbackground=theme.text_bg,
            foreground=theme.text_fg,
            font=data_font,
            rowheight=max(22, int(round(23 * self._ui_scale_factor))),
            bordercolor=theme.border,
        )
        self._style.map(
            "Treeview",
            background=[("selected", theme.selection_bg)],
            foreground=[("selected", theme.selection_fg)],
        )
        self._style.configure(
            "Treeview.Heading",
            background=theme.heading_bg,
            foreground=theme.heading_fg,
            font=(heading_font[0], max(8, heading_font[1] - 2), "bold"),
            relief="raised",
        )
        self._style.map(
            "Treeview.Heading",
            background=[("active", theme.button_active_bg)],
            foreground=[("active", theme.heading_fg)],
        )
        self._style.configure("TCheckbutton", background=theme.panel_bg, foreground=theme.panel_fg, font=label_font)
        self._style.configure("TRadiobutton", background=theme.panel_bg, foreground=theme.panel_fg, font=label_font)

        self.controls.apply_theme(theme)
        self.results_panel.apply_theme(theme, scale_factor=self._ui_scale_factor)
        self.board_canvas.set_visual_theme(theme.key)
        self.controls_canvas.configure(bg=theme.panel_bg)

        self._compare_text.configure(
            bg=theme.text_bg,
            fg=theme.text_fg,
            insertbackground=theme.text_fg,
            selectbackground=theme.selection_bg,
            selectforeground=theme.selection_fg,
            font=data_font,
        )
        self.compare_status_label.configure(foreground=theme.accent_fg)

    # ── existing callbacks ──────────────────────────────────────

    def on_randomize(self) -> None:
        self.board = generate_randomized_board()
        self.board_canvas.set_board(self.board)
        self._analysis_ran_for_current_board = False
        self._knowledge_test_feedback = None
        self._knowledge_test_picks = []
        self._user_roads = set()
        self._last_ranking = []
        self.board_canvas.set_user_picks(self._knowledge_test_picks)
        self.board_canvas.set_user_roads(self._user_roads)
        self.results_panel.clear()
        self._refresh_knowledge_test_ui()
        self.controls.set_status("Board randomized.")
        self.controls.reset_progress()

    def on_analyze(self) -> None:
        if self._analysis_in_progress:
            return

        try:
            config = self.controls.build_config()
        except Exception as exc:  # pragma: no cover - defensive UI guard
            messagebox.showerror("Invalid configuration", str(exc))
            return

        if self.controls.is_knowledge_test_enabled() and len(self._knowledge_test_picks) != 4:
            messagebox.showinfo(
                "Knowledge test",
                "Select exactly 4 settlement vertices on the board before running analysis.",
            )
            self.controls.set_status("Knowledge test: choose 4 vertices first.")
            return

        self._analysis_in_progress = True
        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        self._analysis_cancel_event = threading.Event()
        self._analysis_started_at = time.perf_counter()
        while True:
            try:
                self._analysis_queue.get_nowait()
            except queue.Empty:
                break
        self.controls.set_busy(True)
        self.controls.set_status("Analyzing board on background workers...")
        self.controls.set_progress(0.0, "Queued analysis…", elapsed_seconds=0.0, eta_seconds=None)
        self.update_idletasks()

        self._analysis_worker = threading.Thread(
            target=self._run_analysis_worker,
            args=(request_id, self.board, config, self._analysis_cancel_event),
            daemon=True,
        )
        self._analysis_worker.start()
        self.after(60, self._poll_analysis_queue)

    def on_cancel_analysis(self) -> None:
        if not self._analysis_in_progress:
            return
        if self._analysis_cancel_event is None:
            return
        if self._analysis_cancel_event.is_set():
            return
        self._analysis_cancel_event.set()
        elapsed = max(0.0, time.perf_counter() - self._analysis_started_at)
        self.controls.set_status("Cancelling analysis…")
        self.controls.set_progress(99.0, "Cancelling analysis…", elapsed_seconds=elapsed, eta_seconds=None)

    def on_vertex_selected(self, vertex_id: int | None) -> None:
        self._sync_selection(vertex_id, source="table")

    def on_board_vertex_selected(self, vertex_id: int | None) -> None:
        if self._analysis_in_progress:
            return

        current_mode = self.board_canvas.get_click_mode()

        if current_mode == ClickMode.KNOWLEDGE_TEST_PICK:
            if not self._analysis_ran_for_current_board:
                self._toggle_knowledge_test_pick(vertex_id)
                self._sync_selection(vertex_id, source="board", update_status=False)
                return

        self._sync_selection(vertex_id, source="board")

    def on_board_road_selected(self, edge: EdgeKey) -> None:
        if self._analysis_in_progress:
            return

        normalized = self.board.normalize_edge_key(edge[0], edge[1])
        if normalized in self._user_roads:
            self._user_roads.remove(normalized)
            self.board_canvas.set_user_roads(self._user_roads)
            self.controls.set_status(
                f"Road removed: V{normalized[0]}–V{normalized[1]}."
            )
            return

        settlements = set(self._knowledge_test_picks)
        self._user_roads.add(normalized)
        self.board_canvas.set_user_roads(self._user_roads)
        if settlements and not self.board.is_legal_road(normalized, settlements=settlements, roads=self._user_roads - {normalized}):
            self.controls.set_status(
                f"Road sketched: V{normalized[0]}–V{normalized[1]} (off-network from current test settlements)."
            )
        else:
            self.controls.set_status(
                f"Road placed: V{normalized[0]}–V{normalized[1]}."
            )

    def _run_analysis_worker(self, request_id: int, board, config, cancel_event: threading.Event) -> None:
        try:
            started = time.perf_counter()

            def _handle_progress(stage: str, fraction: float) -> None:
                self._analysis_queue.put(("progress", request_id, (stage, fraction)))

            runtime = AnalysisRuntime(cancel_event=cancel_event, on_progress=_handle_progress)
            runtime.report_progress("Initializing analyzer…", 0.01, force=True)
            analyzer = create_analyzer(config.mode)
            result = analyzer.analyze(board, config, runtime=runtime)
            runtime.report_progress("Finalizing results…", 1.0, force=True)
            elapsed = time.perf_counter() - started
            self._analysis_queue.put(("success", request_id, (result, config, elapsed)))
        except AnalysisCancelled as exc:
            self._analysis_queue.put(("cancelled", request_id, exc))
        except Exception as exc:  # pragma: no cover - defensive worker guard
            self._analysis_queue.put(("error", request_id, exc))

    def _poll_analysis_queue(self) -> None:
        if not self._analysis_in_progress:
            return

        terminal_event: tuple[str, object] | None = None
        latest_progress: tuple[str, float] | None = None

        while True:
            try:
                status, request_id, payload = self._analysis_queue.get_nowait()
            except queue.Empty:
                break

            if request_id != self._analysis_request_id:
                continue

            if status == "progress":
                if isinstance(payload, tuple) and len(payload) >= 2:
                    stage = str(payload[0])
                    try:
                        fraction = float(payload[1])
                    except (TypeError, ValueError):
                        fraction = 0.0
                    latest_progress = (stage, fraction)
                continue

            terminal_event = (status, payload)
            break

        if latest_progress is not None:
            stage, fraction = latest_progress
            percent = max(0.0, min(100.0, fraction * 100.0))
            elapsed = max(0.0, time.perf_counter() - self._analysis_started_at)
            eta_seconds: float | None = None
            if 0.001 < fraction < 0.999:
                eta_seconds = max(0.0, (elapsed / fraction) - elapsed)
            self.controls.set_progress(percent, stage, elapsed_seconds=elapsed, eta_seconds=eta_seconds)

        if terminal_event is None:
            self.after(60, self._poll_analysis_queue)
            return

        status, payload = terminal_event
        try:
            if status == "success":
                result, config, elapsed = payload  # type: ignore[misc]
                self.controls.set_progress(100.0, "Analysis complete.", elapsed_seconds=elapsed, eta_seconds=0.0)
                self._apply_analysis_result(result, config, elapsed)
            elif status == "cancelled":
                elapsed = max(0.0, time.perf_counter() - self._analysis_started_at)
                self.controls.set_status("Analysis cancelled. Previous results are preserved.")
                self.controls.set_progress(0.0, "Cancelled", elapsed_seconds=elapsed, eta_seconds=None)
            else:
                error = payload if isinstance(payload, Exception) else RuntimeError("Unknown analysis failure.")
                messagebox.showerror("Analysis failed", str(error))
                self.controls.set_status("Analysis failed.")
        finally:
            self._analysis_in_progress = False
            self._analysis_cancel_event = None
            self.controls.set_busy(False)

    def _apply_analysis_result(self, result, config, elapsed: float) -> None:
        top_count = 2 * config.player_count
        self.right_notebook.select(self.results_tab)
        self.results_panel.update(result, top_count=top_count)
        self.board_canvas.set_overlays(result.top_recommendations, result.predicted_sequence)
        self.board_canvas.set_clickable_vertices(score.vertex_id for score in result.global_ranking)
        self.board_canvas.set_score_lookup(result.global_ranking)
        self._last_ranking = result.global_ranking
        self._analysis_ran_for_current_board = True

        if self.controls.is_knowledge_test_enabled():
            evaluation = evaluate_user_settlement_picks(
                self._knowledge_test_picks,
                result.global_ranking,
                top_n=4,
            )
            self._knowledge_test_feedback = self._build_knowledge_feedback(evaluation)
        else:
            self._knowledge_test_feedback = None
        self._refresh_knowledge_test_ui()
        self.controls.set_status(f"Analysis complete in {elapsed:.2f}s ({config.mode.value}).")

    def on_knowledge_test_toggle(self, enabled: bool) -> None:
        if not enabled:
            self._knowledge_test_picks = []
            self._knowledge_test_feedback = None
            self._user_roads = set()
            self.board_canvas.set_user_picks(self._knowledge_test_picks)
            self.board_canvas.set_user_roads(self._user_roads)
        else:
            self._analysis_ran_for_current_board = False
            self._knowledge_test_feedback = None
        self._refresh_knowledge_test_ui()

    def _sync_selection(self, vertex_id: int | None, *, source: str, update_status: bool = True) -> None:
        if self._is_syncing_selection:
            return

        self._is_syncing_selection = True
        try:
            self.board_canvas.select_vertex(vertex_id)
            if source != "table":
                self.results_panel.select_vertex(vertex_id)
        finally:
            self._is_syncing_selection = False

        if not update_status:
            return

        if vertex_id is None:
            return

        adjacent_tiles = self.board.vertex_adjacent_tiles(vertex_id)
        resource_labels = ", ".join(tile.resource.value.title() for tile in adjacent_tiles)
        self.controls.set_status(
            f"Selected vertex {vertex_id} controls: {resource_labels or 'No resources'}."
        )

    def _toggle_knowledge_test_pick(self, vertex_id: int | None) -> None:
        if vertex_id is None:
            return

        if vertex_id in self._knowledge_test_picks:
            self._knowledge_test_picks.remove(vertex_id)
            self.controls.set_status(f"Knowledge test: removed vertex {vertex_id}.")
        elif len(self._knowledge_test_picks) < 4:
            self._knowledge_test_picks.append(vertex_id)
            self.controls.set_status(
                f"Knowledge test: added vertex {vertex_id} "
                f"({len(self._knowledge_test_picks)}/4)."
            )
        else:
            self.controls.set_status(
                "Knowledge test already has 4 picks. Click a picked vertex to remove it."
            )
            return

        self._knowledge_test_feedback = None
        self.board_canvas.set_user_picks(self._knowledge_test_picks)
        self._refresh_knowledge_test_ui()

    def _refresh_knowledge_test_ui(self) -> None:
        self.results_panel.update_knowledge_test_status(
            enabled=self.controls.is_knowledge_test_enabled(),
            selected_vertices=self._knowledge_test_picks,
            feedback=self._knowledge_test_feedback,
        )

    def _build_knowledge_feedback(self, evaluation: KnowledgeTestEvaluation) -> str:
        top_vertices = ", ".join(str(vertex_id) for vertex_id in evaluation.top_vertices) or "n/a"
        hits = ", ".join(str(vertex_id) for vertex_id in evaluation.hits) or "none"
        avg_rank = (
            f"{evaluation.average_rank:.1f}"
            if evaluation.average_rank is not None
            else "n/a"
        )
        return (
            f"Result: {evaluation.hit_count}/4 picks in the engine top-4. "
            f"Score: {evaluation.numeric_score:.1f}/100 ({evaluation.letter_grade}). "
            f"Top-4 vertices: {top_vertices}. "
            f"Your hits: {hits}. "
            f"Avg rank of your picks: {avg_rank}."
        )

    def _on_close(self) -> None:
        if self._analysis_cancel_event is not None:
            self._analysis_cancel_event.set()
        self.destroy()


def run_app() -> None:
    app = CatanAnalyzerApp()
    app.mainloop()
