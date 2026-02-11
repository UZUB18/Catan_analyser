from __future__ import annotations

import base64
import io
import math
import tkinter as tk
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Iterable, Optional
from xml.etree import ElementTree as ET

try:  # pragma: no cover - optional dependency path for local desktop UI only
    from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageTk
except Exception:  # pragma: no cover - fallback when PIL isn't available
    Image = None
    ImageDraw = None
    ImageFilter = None
    ImageOps = None
    ImageTk = None

from catan_analyzer.analysis.scoring import pip_value
from catan_analyzer.analysis.types import DraftPick, VertexScore
from catan_analyzer.domain.board import BoardState, EdgeKey, PortType, Resource

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TILES_DIR = _PROJECT_ROOT / "tiles"

_RESOURCE_COLORS_COLOR_SAFE = {
    Resource.WOOD: "#009E73",
    Resource.BRICK: "#D55E00",
    Resource.SHEEP: "#56B4E9",
    Resource.WHEAT: "#E69F00",
    Resource.ORE: "#7A7A7A",
    Resource.DESERT: "#C8A46A",
}

_RESOURCE_COLORS_DARK = {
    Resource.WOOD: "#3BC694",
    Resource.BRICK: "#FF9158",
    Resource.SHEEP: "#73C7FF",
    Resource.WHEAT: "#FFCA4E",
    Resource.ORE: "#A5A5A5",
    Resource.DESERT: "#A18454",
}

PORT_SHORT_LABELS = {
    PortType.ANY_3TO1: "3:1",
    PortType.WOOD_2TO1: "W 2:1",
    PortType.BRICK_2TO1: "B 2:1",
    PortType.SHEEP_2TO1: "S 2:1",
    PortType.WHEAT_2TO1: "Wh 2:1",
    PortType.ORE_2TO1: "O 2:1",
}

_BOARD_VISUAL_THEMES: dict[str, dict[str, object]] = {
    "light": {
        "canvas_bg": "#F8FBFF",
        "resource_colors": _RESOURCE_COLORS_COLOR_SAFE,
        "tile_outline": "#2F3E46",
        "selected_hex_outline": "#FFBF00",
        "token_fill": "#FFF8E1",
        "token_outline": "#444444",
        "token_selected_outline": "#FF8F00",
        "token_text": "#222222",
        "token_hot_text": "#C62828",
        "desert_text": "#6B4F2D",
        "port_line": "#0072B2",
        "port_text": "#004D78",
        "vertex_click_fill": "#D9EEFF",
        "vertex_click_outline": "#0072B2",
        "vertex_base_fill": "#2F3E46",
        "engine_ring_color": "#0072B2",
        "user_ring_color": "#CC79A7",
        "selected_ring_color": "#E69F00",
        "selected_glow_color": "#FFD580",
        "compare_ring_color": "#009E73",
        "road_color": "#1565C0",
        "road_outline_color": "#0B2A52",
        "road_preview_color": "#64B5F6",
        "road_anchor_color": "#0D47A1",
        "player_colors": {1: "#D55E00", 2: "#0072B2", 3: "#009E73", 4: "#CC79A7"},
        "tooltip_bg": "#1E293B",
        "tooltip_fg": "#E2E8F0",
        "tooltip_border": "#475569",
    },
    "dark": {
        "canvas_bg": "#0F1722",
        "resource_colors": _RESOURCE_COLORS_DARK,
        "tile_outline": "#8AA2BD",
        "selected_hex_outline": "#FFD166",
        "token_fill": "#1E2B3A",
        "token_outline": "#B6C5D6",
        "token_selected_outline": "#FFBF47",
        "token_text": "#F0F5FA",
        "token_hot_text": "#FF8D8D",
        "desert_text": "#F1D4A6",
        "port_line": "#65B8FF",
        "port_text": "#9DD3FF",
        "vertex_click_fill": "#244660",
        "vertex_click_outline": "#9ED3FF",
        "vertex_base_fill": "#9AB1C8",
        "engine_ring_color": "#56B4E9",
        "user_ring_color": "#F19BD4",
        "selected_ring_color": "#FFD166",
        "selected_glow_color": "#FFE9A9",
        "compare_ring_color": "#74E3BE",
        "road_color": "#7BC0FF",
        "road_outline_color": "#2D4E6A",
        "road_preview_color": "#9DD3FF",
        "road_anchor_color": "#D2E9FF",
        "player_colors": {1: "#FF9A65", 2: "#6DC8FF", 3: "#70DDB7", 4: "#F2A7D8"},
        "tooltip_bg": "#0B131D",
        "tooltip_fg": "#F1F6FF",
        "tooltip_border": "#5C7490",
    },
    "high_contrast": {
        "canvas_bg": "#000000",
        "resource_colors": {
            Resource.WOOD: "#00FF9C",
            Resource.BRICK: "#FF8A00",
            Resource.SHEEP: "#00C2FF",
            Resource.WHEAT: "#FFF200",
            Resource.ORE: "#CFCFCF",
            Resource.DESERT: "#B28B42",
        },
        "tile_outline": "#FFFFFF",
        "selected_hex_outline": "#FFD400",
        "token_fill": "#000000",
        "token_outline": "#FFFFFF",
        "token_selected_outline": "#FFD400",
        "token_text": "#FFFFFF",
        "token_hot_text": "#FF4D4D",
        "desert_text": "#FFFFFF",
        "port_line": "#00FFFF",
        "port_text": "#00FFFF",
        "vertex_click_fill": "#FFFFFF",
        "vertex_click_outline": "#000000",
        "vertex_base_fill": "#FFFFFF",
        "engine_ring_color": "#00C2FF",
        "user_ring_color": "#FF7EEB",
        "selected_ring_color": "#FFD400",
        "selected_glow_color": "#FFFFFF",
        "compare_ring_color": "#6CFFAA",
        "road_color": "#00FFFF",
        "road_outline_color": "#003D3D",
        "road_preview_color": "#9FFFFF",
        "road_anchor_color": "#FFFFFF",
        "player_colors": {1: "#FF8A00", 2: "#00C2FF", 3: "#6CFFAA", 4: "#FF7EEB"},
        "tooltip_bg": "#000000",
        "tooltip_fg": "#FFFFFF",
        "tooltip_border": "#FFFFFF",
    },
}

_RESOURCE_TEXTURE_KEYWORDS = {
    Resource.WOOD: "wood",
    Resource.BRICK: "brick",
    Resource.SHEEP: "sheep",
    Resource.WHEAT: "wheat",
    Resource.ORE: "ore",
}

# Zoom textured artwork slightly so built-in matte/borders in source images
# don't leave pale bands near hex edges.
_TILE_TEXTURE_ZOOM = 1.34
_TILE_BASE_OVERDRAW_WIDTH = 3
_TILE_OUTLINE_WIDTH = 1.5
_TEXTURE_MASK_THRESHOLD = 20

# ── Ring style constants ────────────────────────────────────────
# Engine top-pick ring
ENGINE_RING_RADIUS = 16
ENGINE_RING_WIDTH = 3.0
ENGINE_RING_DASH = (6, 3)            # dashed

# Draft sequence ring
DRAFT_RING_RADIUS = 21
DRAFT_RING_WIDTH = 3.0
DRAFT_RING_DASH = (8, 3, 2, 3)      # dash-dot

# User test-pick ring (double-ring)
USER_OUTER_RADIUS = 20
USER_INNER_RADIUS = 14
USER_RING_WIDTH = 2.5

# Selected vertex ring (thick double-ring)
SELECT_OUTER_RADIUS = 25
SELECT_INNER_RADIUS = 11
SELECT_OUTER_WIDTH = 3
SELECT_INNER_WIDTH = 2.0

# Compare mode ring
COMPARE_RING_RADIUS = 23
COMPARE_RING_WIDTH = 3.0
COMPARE_RING_DASH = (4, 4)

ROAD_LINE_WIDTH = 12
ROAD_LINE_OUTLINE_WIDTH = 18
ROAD_ANCHOR_RADIUS = 14
ROAD_PICK_TOLERANCE = 14.0


class ClickMode(Enum):
    """Determines what happens when the user clicks on the board."""
    INSPECT = auto()
    KNOWLEDGE_TEST_PICK = auto()
    COMPARE_TWO_VERTICES = auto()
    PLACE_ROADS = auto()


class BoardCanvas(tk.Canvas):
    def __init__(
        self,
        master: tk.Widget,
        *,
        on_vertex_selected: Callable[[int | None], None] | None = None,
        on_compare_pair: Callable[[int, int], None] | None = None,
        on_road_selected: Callable[[EdgeKey], None] | None = None,
        **kwargs,
    ) -> None:
        self._theme_key = "light"
        self._colors = _BOARD_VISUAL_THEMES[self._theme_key]
        super().__init__(
            master,
            width=860,
            height=700,
            bg=str(self._colors["canvas_bg"]),
            highlightthickness=1,
            **kwargs,
        )
        self._board: BoardState | None = None
        self._top_scores: list[VertexScore] = []
        self._sequence: list[DraftPick] = []
        self._selected_vertex_id: int | None = None
        self._selected_hex_ids: set[int] = set()
        self._clickable_vertex_ids: set[int] = set()
        self._user_picks: list[int] = []
        self._compare_picks: list[int] = []
        self._user_roads: set[EdgeKey] = set()
        self._road_anchor_vertex_id: int | None = None
        self._road_hover_vertex_id: int | None = None
        self._on_vertex_selected = on_vertex_selected
        self._on_compare_pair = on_compare_pair
        self._on_road_selected = on_road_selected

        # Click mode
        self._click_mode: ClickMode = ClickMode.INSPECT

        # Hover tooltip state
        self._hover_vertex_id: int | None = None
        self._tooltip: tk.Toplevel | None = None

        # Score lookup (vertex_id → VertexScore) for rank snapshot
        self._score_lookup: dict[int, tuple[int, VertexScore]] = {}
        self._tile_texture_sources = self._load_tile_texture_sources()
        self._tile_texture_cache: dict[tuple[Resource, int, int], object] = {}
        self._tile_texture_refs: list[object] = []

        self.bind("<Configure>", lambda _event: self.draw())
        self.bind("<Button-1>", self._handle_click)
        self.bind("<Motion>", self._handle_motion)
        self.bind("<Leave>", self._handle_leave)

    # ── public API ──────────────────────────────────────────────

    def set_click_mode(self, mode: ClickMode) -> None:
        """Switch the active click behaviour."""
        self._click_mode = mode
        # Clear compare picks when leaving compare mode
        if mode != ClickMode.COMPARE_TWO_VERTICES:
            self._compare_picks.clear()
        if mode != ClickMode.PLACE_ROADS:
            self._road_anchor_vertex_id = None
            self._road_hover_vertex_id = None
        self.draw()

    def get_click_mode(self) -> ClickMode:
        return self._click_mode

    def set_visual_theme(self, theme_key: str) -> None:
        normalized = str(theme_key).strip().lower()
        if normalized not in _BOARD_VISUAL_THEMES:
            normalized = "light"
        self._theme_key = normalized
        self._colors = _BOARD_VISUAL_THEMES[normalized]
        self.configure(bg=str(self._colors["canvas_bg"]))
        self._hide_tooltip()
        self.draw()

    def set_board(self, board: BoardState) -> None:
        self._board = board
        self._top_scores = []
        self._sequence = []
        self._selected_vertex_id = None
        self._selected_hex_ids = set()
        self._clickable_vertex_ids = set(board.vertices.keys())
        self._user_picks = []
        self._compare_picks = []
        self._user_roads = set()
        self._road_anchor_vertex_id = None
        self._road_hover_vertex_id = None
        self._score_lookup = {}
        self.draw()

    def set_overlays(self, top_scores: Iterable[VertexScore], sequence: Iterable[DraftPick]) -> None:
        self._top_scores = list(top_scores)
        self._sequence = list(sequence)
        self.draw()

    def set_score_lookup(self, ranking: list[VertexScore]) -> None:
        """Store full ranking so hover can show rank snapshot."""
        self._score_lookup = {
            score.vertex_id: (rank, score)
            for rank, score in enumerate(ranking, start=1)
        }

    def set_clickable_vertices(self, vertex_ids: Iterable[int] | None) -> None:
        if self._board is None:
            self._clickable_vertex_ids = set(vertex_ids or ())
            return

        if vertex_ids is None:
            self._clickable_vertex_ids = set(self._board.vertices.keys())
        else:
            self._clickable_vertex_ids = {
                vertex_id
                for vertex_id in vertex_ids
                if vertex_id in self._board.vertices
            }
            if not self._clickable_vertex_ids:
                self._clickable_vertex_ids = set(self._board.vertices.keys())
        self.draw()

    def set_user_picks(self, vertex_ids: Iterable[int]) -> None:
        if self._board is None:
            self._user_picks = list(vertex_ids)
            return

        self._user_picks = [
            vertex_id
            for vertex_id in vertex_ids
            if vertex_id in self._board.vertices
        ]
        self.draw()

    def set_user_roads(self, edges: Iterable[EdgeKey]) -> None:
        if self._board is None:
            self._user_roads = {
                tuple(sorted((int(edge[0]), int(edge[1]))))
                for edge in edges
                if len(edge) == 2
            }
            self.draw()
            return

        filtered: set[EdgeKey] = set()
        for edge in edges:
            if len(edge) != 2:
                continue
            normalized = self._board.normalize_edge_key(edge[0], edge[1])
            if self._board.edge_exists(normalized):
                filtered.add(normalized)
        self._user_roads = filtered
        self.draw()

    def clear_user_roads(self) -> None:
        self._user_roads.clear()
        self._road_anchor_vertex_id = None
        self._road_hover_vertex_id = None
        self.draw()

    def clear_overlays(self) -> None:
        self._top_scores = []
        self._sequence = []
        self._score_lookup = {}
        self.draw()

    def select_vertex(self, vertex_id: int | None) -> None:
        self._selected_vertex_id = vertex_id
        self._selected_hex_ids = set()
        if self._board is not None and vertex_id is not None and vertex_id in self._board.vertices:
            self._selected_hex_ids = set(self._board.vertices[vertex_id].adjacent_hex_ids)
        self.draw()

    def clear_selection(self) -> None:
        self.select_vertex(None)

    def clear_compare_picks(self) -> None:
        self._compare_picks.clear()
        self.draw()

    def clear_road_anchor(self) -> None:
        self._road_anchor_vertex_id = None
        self._road_hover_vertex_id = None
        self.draw()

    # ── draw ────────────────────────────────────────────────────

    def draw(self) -> None:
        self.delete("all")
        self._tile_texture_refs = []
        if self._board is None:
            return

        scale, offset_x, offset_y = self._layout()

        def to_canvas(point: tuple[float, float]) -> tuple[float, float]:
            return (point[0] * scale + offset_x, point[1] * scale + offset_y)

        # ── tiles ───────────────────────────────────────────────
        for tile in self._board.tiles:
            polygon_points: list[float] = []
            polygon_tuples: list[tuple[float, float]] = []
            for corner in tile.corner_points:
                x, y = to_canvas(corner)
                polygon_points.extend([x, y])
                polygon_tuples.append((x, y))

            bounds = self._polygon_int_bounds(polygon_tuples)
            texture_image = self._get_tile_texture_image(tile.resource, polygon_tuples, bounds=bounds)
            if texture_image is not None:
                # Draw a base fill first to eliminate white seams between adjacent textured hexes.
                base_color = self._resource_color(tile.resource)
                self.create_polygon(
                    polygon_points,
                    fill=base_color,
                    outline=base_color,
                    width=_TILE_BASE_OVERDRAW_WIDTH,
                    joinstyle=tk.ROUND,
                )
                self.create_image(bounds[0], bounds[1], image=texture_image, anchor="nw")
                self._tile_texture_refs.append(texture_image)
                self.create_polygon(
                    polygon_points,
                    fill="",
                    outline=str(self._colors["tile_outline"]),
                    width=_TILE_OUTLINE_WIDTH,
                    joinstyle=tk.ROUND,
                )
            else:
                self.create_polygon(
                    polygon_points,
                    fill=self._resource_color(tile.resource),
                    outline=str(self._colors["tile_outline"]),
                    width=_TILE_OUTLINE_WIDTH,
                    joinstyle=tk.ROUND,
                )
            if tile.id in self._selected_hex_ids:
                self.create_polygon(
                    polygon_points,
                    fill="",
                    outline=str(self._colors["selected_hex_outline"]),
                    width=5,
                )
            center_x, center_y = to_canvas(tile.center)
            if tile.token_number is None:
                self.create_text(
                    center_x,
                    center_y,
                    text="DESERT",
                    font=("Segoe UI", 11, "bold"),
                    fill=str(self._colors["desert_text"]),
                )
            else:
                token_outline = (
                    str(self._colors["token_selected_outline"])
                    if tile.id in self._selected_hex_ids
                    else str(self._colors["token_outline"])
                )
                self.create_oval(
                    center_x - 21,
                    center_y - 21,
                    center_x + 21,
                    center_y + 21,
                    fill=str(self._colors["token_fill"]),
                    outline=token_outline,
                    width=2 if tile.id in self._selected_hex_ids else 1,
                )
                token_color = (
                    str(self._colors["token_hot_text"])
                    if tile.token_number in (6, 8)
                    else str(self._colors["token_text"])
                )
                self.create_text(
                    center_x,
                    center_y,
                    text=str(tile.token_number),
                    font=("Segoe UI", 13, "bold"),
                    fill=token_color,
                )

        center_x = self.winfo_width() / 2
        center_y = self.winfo_height() / 2

        # ── ports ───────────────────────────────────────────────
        for port in self._board.ports:
            first, second = port.vertex_ids
            first_point = to_canvas(self._board.vertices[first].point)
            second_point = to_canvas(self._board.vertices[second].point)
            midpoint = to_canvas(port.midpoint)
            self.create_line(*first_point, *second_point, fill=str(self._colors["port_line"]), width=4)

            dir_x = midpoint[0] - center_x
            dir_y = midpoint[1] - center_y
            magnitude = (dir_x**2 + dir_y**2) ** 0.5 or 1.0
            label_x = midpoint[0] + (dir_x / magnitude) * 22
            label_y = midpoint[1] + (dir_y / magnitude) * 22
            self.create_text(
                label_x,
                label_y,
                text=PORT_SHORT_LABELS[port.port_type],
                font=("Segoe UI", 9, "bold"),
                fill=str(self._colors["port_text"]),
            )

        # ── vertex dots ─────────────────────────────────────────
        # ── user roads ───────────────────────────────────────────
        road_color = str(self._colors.get("road_color", self._colors.get("engine_ring_color", "#1565C0")))
        road_outline_color = str(self._colors.get("road_outline_color", "#0B2A52"))
        for edge in sorted(self._user_roads):
            first, second = edge
            if first not in self._board.vertices or second not in self._board.vertices:
                continue
            first_point = to_canvas(self._board.vertices[first].point)
            second_point = to_canvas(self._board.vertices[second].point)
            self.create_line(
                *first_point,
                *second_point,
                fill=road_outline_color,
                width=ROAD_LINE_OUTLINE_WIDTH,
                capstyle=tk.ROUND,
            )
            self.create_line(
                *first_point,
                *second_point,
                fill=road_color,
                width=ROAD_LINE_WIDTH,
                capstyle=tk.ROUND,
            )

        # ── road placement preview ───────────────────────────────
        if (
            self._click_mode == ClickMode.PLACE_ROADS
            and self._road_anchor_vertex_id is not None
            and self._road_anchor_vertex_id in self._board.vertices
        ):
            anchor = self._road_anchor_vertex_id
            anchor_x, anchor_y = to_canvas(self._board.vertices[anchor].point)
            anchor_color = str(self._colors.get("road_anchor_color", self._colors.get("road_color", "#0D47A1")))
            self.create_oval(
                anchor_x - ROAD_ANCHOR_RADIUS,
                anchor_y - ROAD_ANCHOR_RADIUS,
                anchor_x + ROAD_ANCHOR_RADIUS,
                anchor_y + ROAD_ANCHOR_RADIUS,
                outline=anchor_color,
                width=3,
            )
            self.create_text(
                anchor_x,
                anchor_y - ROAD_ANCHOR_RADIUS - 10,
                text="ANCHOR",
                fill=anchor_color,
                font=("Segoe UI", 8, "bold"),
            )
            if (
                self._road_hover_vertex_id is not None
                and self._road_hover_vertex_id in self._board.vertices
                and self._road_hover_vertex_id in self._board.vertices[anchor].adjacent_vertex_ids
            ):
                hover_x, hover_y = to_canvas(self._board.vertices[self._road_hover_vertex_id].point)
                preview_outline = str(self._colors.get("road_outline_color", "#0B2A52"))
                preview_color = str(self._colors.get("road_preview_color", "#64B5F6"))
                self.create_line(
                    anchor_x,
                    anchor_y,
                    hover_x,
                    hover_y,
                    fill=preview_outline,
                    width=ROAD_LINE_OUTLINE_WIDTH,
                    capstyle=tk.ROUND,
                )
                self.create_line(
                    anchor_x,
                    anchor_y,
                    hover_x,
                    hover_y,
                    fill=preview_color,
                    width=ROAD_LINE_WIDTH,
                    capstyle=tk.ROUND,
                )

        clickable_vertex_ids = self._clickable_vertex_ids or set(self._board.vertices.keys())
        for vertex in self._board.vertices.values():
            vx, vy = to_canvas(vertex.point)
            if vertex.id in clickable_vertex_ids:
                self.create_oval(
                    vx - 5,
                    vy - 5,
                    vx + 5,
                    vy + 5,
                    fill=str(self._colors["vertex_click_fill"]),
                    outline=str(self._colors["vertex_click_outline"]),
                    width=1,
                )
            else:
                self.create_oval(
                    vx - 3,
                    vy - 3,
                    vx + 3,
                    vy + 3,
                    fill=str(self._colors["vertex_base_fill"]),
                    outline="",
                )

        # ── engine top picks (dashed cyan ring) ─────────────────
        for rank, score in enumerate(self._top_scores, start=1):
            vertex = self._board.vertices[score.vertex_id]
            vx, vy = to_canvas(vertex.point)
            r = ENGINE_RING_RADIUS
            self.create_oval(
                vx - r, vy - r, vx + r, vy + r,
                outline=str(self._colors["engine_ring_color"]),
                width=ENGINE_RING_WIDTH,
                dash=ENGINE_RING_DASH,
            )
            # Rank badge (filled small circle with text)
            badge_r = 11
            self.create_oval(
                vx - badge_r, vy - badge_r, vx + badge_r, vy + badge_r,
                fill=str(self._colors["engine_ring_color"]), outline="#FFFFFF", width=1.5,
            )
            self.create_text(vx, vy, text=str(rank), fill="#FFFFFF", font=("Segoe UI", 10, "bold"))

        # ── user test picks (double-ring purple) ────────────────
        for pick_index, vertex_id in enumerate(self._user_picks, start=1):
            vertex = self._board.vertices.get(vertex_id)
            if vertex is None:
                continue
            vx, vy = to_canvas(vertex.point)
            # Outer ring
            r = USER_OUTER_RADIUS
            self.create_oval(
                vx - r, vy - r, vx + r, vy + r,
                outline=str(self._colors["user_ring_color"]), width=USER_RING_WIDTH,
            )
            # Inner ring
            r2 = USER_INNER_RADIUS
            self.create_oval(
                vx - r2, vy - r2, vx + r2, vy + r2,
                outline=str(self._colors["user_ring_color"]), width=1.5, dash=(3, 2),
            )
            # Label
            self.create_text(
                vx + 23, vy - 23,
                text=f"U{pick_index}",
                fill=str(self._colors["user_ring_color"]),
                font=("Segoe UI", 9, "bold"),
            )

        # ── draft sequence picks (dash-dot player-colored ring) ─
        for pick in self._sequence:
            vertex = self._board.vertices[pick.vertex_id]
            vx, vy = to_canvas(vertex.point)
            player_colors = self._colors.get("player_colors", {})
            color = player_colors.get(pick.player_id, "#A33D00")
            r = DRAFT_RING_RADIUS
            self.create_oval(
                vx - r, vy - r, vx + r, vy + r,
                outline=color, width=DRAFT_RING_WIDTH, dash=DRAFT_RING_DASH,
            )
            self.create_text(
                vx, vy - r - 8,
                text=f"T{pick.turn_index}",
                fill=color,
                font=("Segoe UI", 10, "bold"),
            )

        # ── compare picks (teal crosshair ring) ────────────────
        for idx, vertex_id in enumerate(self._compare_picks, start=1):
            vertex = self._board.vertices.get(vertex_id)
            if vertex is None:
                continue
            vx, vy = to_canvas(vertex.point)
            r = COMPARE_RING_RADIUS
            self.create_oval(
                vx - r, vy - r, vx + r, vy + r,
                outline=str(self._colors["compare_ring_color"]),
                width=COMPARE_RING_WIDTH,
                dash=COMPARE_RING_DASH,
            )
            # Crosshair lines
            arm = r + 5
            compare_color = str(self._colors["compare_ring_color"])
            self.create_line(vx - arm, vy, vx - r + 2, vy, fill=compare_color, width=1.5)
            self.create_line(vx + r - 2, vy, vx + arm, vy, fill=compare_color, width=1.5)
            self.create_line(vx, vy - arm, vx, vy - r + 2, fill=compare_color, width=1.5)
            self.create_line(vx, vy + r - 2, vx, vy + arm, fill=compare_color, width=1.5)
            # Label
            self.create_text(
                vx + r + 8, vy - r - 4,
                text=f"C{idx}",
                fill=str(self._colors["compare_ring_color"]),
                font=("Segoe UI", 9, "bold"),
            )

        # ── selected vertex (thick amber double-ring) ──────────
        if self._selected_vertex_id is not None and self._selected_vertex_id in self._board.vertices:
            selected_vertex = self._board.vertices[self._selected_vertex_id]
            vx, vy = to_canvas(selected_vertex.point)
            r = SELECT_OUTER_RADIUS
            self.create_oval(
                vx - r, vy - r, vx + r, vy + r,
                outline=str(self._colors["selected_ring_color"]), width=SELECT_OUTER_WIDTH,
            )
            # Glow ring (slightly larger, semi-transparent feel via stipple)
            glow_r = r + 4
            self.create_oval(
                vx - glow_r, vy - glow_r, vx + glow_r, vy + glow_r,
                outline=str(self._colors["selected_glow_color"]), width=1.5, dash=(2, 2),
            )
            # Inner filled dot
            r2 = SELECT_INNER_RADIUS
            self.create_oval(
                vx - r2, vy - r2, vx + r2, vy + r2,
                fill=str(self._colors["selected_ring_color"]),
                outline="#FFFFFF",
                width=SELECT_INNER_WIDTH,
            )

    # ── hover tooltip ───────────────────────────────────────────

    def _handle_motion(self, event: tk.Event) -> None:
        if self._board is None:
            return

        if self._click_mode == ClickMode.PLACE_ROADS and self._road_anchor_vertex_id is not None:
            anchor = self._road_anchor_vertex_id
            anchor_vertex = self._board.vertices.get(anchor)
            if anchor_vertex is not None:
                hover_vertex = self._closest_vertex(
                    click_x=float(event.x),
                    click_y=float(event.y),
                    candidate_vertex_ids=anchor_vertex.adjacent_vertex_ids,
                    max_distance=24.0,
                )
                if hover_vertex != self._road_hover_vertex_id:
                    self._road_hover_vertex_id = hover_vertex
                    self.draw()

        vertex_id = self._closest_vertex(
            click_x=float(event.x),
            click_y=float(event.y),
            candidate_vertex_ids=self._board.vertices.keys(),
            max_distance=20.0,
        )

        if vertex_id == self._hover_vertex_id:
            # Move tooltip if it exists
            if self._tooltip is not None:
                self._position_tooltip(event)
            return

        self._hover_vertex_id = vertex_id
        self._hide_tooltip()

        if vertex_id is None:
            return

        self._show_tooltip(vertex_id, event)

    def _handle_leave(self, _event: tk.Event) -> None:
        self._hover_vertex_id = None
        self._road_hover_vertex_id = None
        self._hide_tooltip()

    def _show_tooltip(self, vertex_id: int, event: tk.Event) -> None:
        if self._board is None:
            return

        vertex = self._board.vertices.get(vertex_id)
        if vertex is None:
            return

        # Build tooltip content
        lines: list[str] = [f"Vertex {vertex_id}"]

        # Adjacent resources & pip total
        adjacent_tiles = self._board.vertex_adjacent_tiles(vertex_id)
        res_parts: list[str] = []
        total_pips = 0
        for tile in adjacent_tiles:
            pips = pip_value(tile.token_number)
            total_pips += pips
            if tile.resource is Resource.DESERT:
                res_parts.append("Desert")
            else:
                num_str = str(tile.token_number) if tile.token_number else "?"
                res_parts.append(f"{tile.resource.value.title()} ({num_str}, {pips}♦)")
        if res_parts:
            lines.append("Resources: " + ", ".join(res_parts))
        lines.append(f"Pip total: {total_pips}")

        # Port info
        if vertex.port_type is not None:
            lines.append(f"Port: {PORT_SHORT_LABELS.get(vertex.port_type, str(vertex.port_type))}")
        else:
            lines.append("Port: none")

        # Rank snapshot
        rank_info = self._score_lookup.get(vertex_id)
        if rank_info is not None:
            rank, score = rank_info
            lines.append(f"Rank: #{rank}  Score: {score.total_score:.2f}")
            lines.append(f"  Yield={score.expected_yield:.2f}  Div={score.diversity_score:.2f}  Port={score.port_score:.2f}")
        else:
            lines.append("Rank: (run analysis first)")

        text = "\n".join(lines)

        # Create tooltip window
        self._tooltip = tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        tw_bg = str(self._colors["tooltip_bg"])
        tw_fg = str(self._colors["tooltip_fg"])
        tw_border = str(self._colors["tooltip_border"])
        tw.configure(bg=tw_bg)

        frame = tk.Frame(
            tw,
            bg=tw_bg,
            padx=10,
            pady=8,
            bd=1,
            relief="solid",
            highlightbackground=tw_border,
            highlightthickness=1,
        )
        frame.pack()

        label = tk.Label(
            frame,
            text=text,
            justify="left",
            bg=tw_bg,
            fg=tw_fg,
            font=("Segoe UI", 9),
            wraplength=320,
        )
        label.pack()

        self._position_tooltip(event)

    def _position_tooltip(self, event: tk.Event) -> None:
        if self._tooltip is None:
            return
        x = event.x_root + 16
        y = event.y_root + 12
        self._tooltip.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self) -> None:
        if self._tooltip is not None:
            self._tooltip.destroy()
            self._tooltip = None

    # ── click handling ──────────────────────────────────────────

    def _handle_click(self, event: tk.Event) -> None:
        if self._board is None:
            return

        if self._click_mode == ClickMode.PLACE_ROADS:
            self._handle_road_click(click_x=float(event.x), click_y=float(event.y))
            return

        clickable_vertex_ids = self._clickable_vertex_ids or set(self._board.vertices.keys())
        vertex_id = self._closest_vertex(
            click_x=float(event.x),
            click_y=float(event.y),
            candidate_vertex_ids=clickable_vertex_ids,
            max_distance=22.0,
        )

        if self._click_mode == ClickMode.INSPECT:
            self._handle_inspect_click(vertex_id)
        elif self._click_mode == ClickMode.KNOWLEDGE_TEST_PICK:
            self._handle_knowledge_click(vertex_id)
        elif self._click_mode == ClickMode.COMPARE_TWO_VERTICES:
            self._handle_compare_click(vertex_id)

    def _handle_inspect_click(self, vertex_id: int | None) -> None:
        """Default inspect behaviour (original logic)."""
        if self._on_vertex_selected is None:
            if vertex_id is None:
                self.clear_selection()
            else:
                self.select_vertex(vertex_id)
            return
        self._on_vertex_selected(vertex_id)

    def _handle_knowledge_click(self, vertex_id: int | None) -> None:
        """Forward to the vertex-selected callback (app decides toggle logic)."""
        if self._on_vertex_selected is not None:
            self._on_vertex_selected(vertex_id)

    def _handle_compare_click(self, vertex_id: int | None) -> None:
        if vertex_id is None:
            return

        # Toggle: if already in compare picks, remove it
        if vertex_id in self._compare_picks:
            self._compare_picks.remove(vertex_id)
            self.draw()
            return

        self._compare_picks.append(vertex_id)
        self.draw()

        if len(self._compare_picks) >= 2:
            a, b = self._compare_picks[0], self._compare_picks[1]
            if self._on_compare_pair is not None:
                self._on_compare_pair(a, b)
            # Keep picks visible for the user; caller can clear via clear_compare_picks()

    def _handle_road_click(self, *, click_x: float, click_y: float) -> None:
        if self._board is None:
            return

        clickable_vertex_ids = self._clickable_vertex_ids or set(self._board.vertices.keys())
        vertex_id = self._closest_vertex(
            click_x=click_x,
            click_y=click_y,
            candidate_vertex_ids=clickable_vertex_ids,
            max_distance=22.0,
        )

        if self._road_anchor_vertex_id is None:
            if vertex_id is None:
                return
            self._road_anchor_vertex_id = vertex_id
            self._road_hover_vertex_id = None
            self.draw()
            return

        anchor = self._road_anchor_vertex_id
        if vertex_id == anchor:
            self._road_anchor_vertex_id = None
            self._road_hover_vertex_id = None
            self.draw()
            return

        if (
            vertex_id is not None
            and anchor in self._board.vertices
            and vertex_id in self._board.vertices[anchor].adjacent_vertex_ids
        ):
            edge = self._board.normalize_edge_key(anchor, vertex_id)
            if self._on_road_selected is not None:
                self._on_road_selected(edge)
            self._road_anchor_vertex_id = vertex_id
            self._road_hover_vertex_id = None
            self.draw()
            return

        edge = self._closest_adjacent_edge(
            anchor_vertex_id=anchor,
            click_x=click_x,
            click_y=click_y,
            max_distance=ROAD_PICK_TOLERANCE,
        )
        if edge is not None:
            if self._on_road_selected is not None:
                self._on_road_selected(edge)
            self._road_hover_vertex_id = None
            self.draw()
            return

        if vertex_id is None:
            return

        # Re-anchor if user clicks another vertex.
        self._road_anchor_vertex_id = vertex_id
        self._road_hover_vertex_id = None
        self.draw()

    # ── geometry helpers ────────────────────────────────────────

    def _closest_vertex(
        self,
        *,
        click_x: float,
        click_y: float,
        candidate_vertex_ids: Iterable[int],
        max_distance: float,
    ) -> int | None:
        if self._board is None:
            return None

        scale, offset_x, offset_y = self._layout()
        max_distance_sq = max_distance**2
        closest_vertex_id: int | None = None

        for vertex_id in candidate_vertex_ids:
            vertex = self._board.vertices.get(vertex_id)
            if vertex is None:
                continue
            vx = vertex.point[0] * scale + offset_x
            vy = vertex.point[1] * scale + offset_y
            distance_sq = (vx - click_x) ** 2 + (vy - click_y) ** 2
            if distance_sq <= max_distance_sq:
                max_distance_sq = distance_sq
                closest_vertex_id = vertex_id

        return closest_vertex_id

    def _closest_adjacent_edge(
        self,
        *,
        anchor_vertex_id: int,
        click_x: float,
        click_y: float,
        max_distance: float,
    ) -> EdgeKey | None:
        if self._board is None:
            return None

        anchor_vertex = self._board.vertices.get(anchor_vertex_id)
        if anchor_vertex is None:
            return None

        scale, offset_x, offset_y = self._layout()
        anchor_x = anchor_vertex.point[0] * scale + offset_x
        anchor_y = anchor_vertex.point[1] * scale + offset_y

        best_edge: EdgeKey | None = None
        best_distance_sq = max_distance**2

        for neighbor_id in anchor_vertex.adjacent_vertex_ids:
            neighbor_vertex = self._board.vertices.get(neighbor_id)
            if neighbor_vertex is None:
                continue
            neighbor_x = neighbor_vertex.point[0] * scale + offset_x
            neighbor_y = neighbor_vertex.point[1] * scale + offset_y
            distance_sq = self._point_to_segment_distance_sq(
                point_x=click_x,
                point_y=click_y,
                segment_start_x=anchor_x,
                segment_start_y=anchor_y,
                segment_end_x=neighbor_x,
                segment_end_y=neighbor_y,
            )
            if distance_sq <= best_distance_sq:
                best_distance_sq = distance_sq
                best_edge = self._board.normalize_edge_key(anchor_vertex_id, neighbor_id)

        return best_edge

    @staticmethod
    def _point_to_segment_distance_sq(
        *,
        point_x: float,
        point_y: float,
        segment_start_x: float,
        segment_start_y: float,
        segment_end_x: float,
        segment_end_y: float,
    ) -> float:
        segment_dx = segment_end_x - segment_start_x
        segment_dy = segment_end_y - segment_start_y
        segment_length_sq = segment_dx * segment_dx + segment_dy * segment_dy
        if segment_length_sq <= 1e-9:
            dx = point_x - segment_start_x
            dy = point_y - segment_start_y
            return dx * dx + dy * dy

        projection = (
            (point_x - segment_start_x) * segment_dx
            + (point_y - segment_start_y) * segment_dy
        ) / segment_length_sq
        projection = max(0.0, min(1.0, projection))
        nearest_x = segment_start_x + projection * segment_dx
        nearest_y = segment_start_y + projection * segment_dy
        dx = point_x - nearest_x
        dy = point_y - nearest_y
        return dx * dx + dy * dy

    def _resource_color(self, resource: Resource) -> str:
        resource_colors = self._colors.get("resource_colors", {})
        if isinstance(resource_colors, dict):
            return str(resource_colors.get(resource, "#BDBDBD"))
        return "#BDBDBD"

    @staticmethod
    def _polygon_int_bounds(points: list[tuple[float, float]]) -> tuple[int, int, int, int]:
        min_x = math.floor(min(point[0] for point in points))
        min_y = math.floor(min(point[1] for point in points))
        max_x = math.ceil(max(point[0] for point in points))
        max_y = math.ceil(max(point[1] for point in points))
        return (min_x, min_y, max_x, max_y)

    def _load_tile_texture_sources(self) -> dict[Resource, object]:
        if Image is None or ImageDraw is None or ImageFilter is None or ImageOps is None or ImageTk is None:
            return {}
        if not _TILES_DIR.exists():
            return {}

        svg_files = list(_TILES_DIR.glob("*.svg"))
        if not svg_files:
            return {}

        sources: dict[Resource, object] = {}
        for resource, keyword in _RESOURCE_TEXTURE_KEYWORDS.items():
            matches = [
                svg_path
                for svg_path in svg_files
                if keyword in svg_path.name.lower()
            ]
            if not matches:
                continue
            matches.sort(key=lambda path: len(path.name))
            image = self._decode_embedded_png_from_svg(matches[0])
            if image is not None:
                sources[resource] = image
        return sources

    def _decode_embedded_png_from_svg(self, svg_path: Path) -> object | None:
        if Image is None or ImageDraw is None:
            return None
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
        except Exception:
            return None

        image_element = root.find("{http://www.w3.org/2000/svg}image")
        if image_element is None:
            return None

        href = image_element.attrib.get("href") or image_element.attrib.get("{http://www.w3.org/1999/xlink}href")
        if not href:
            return None
        compact_href = "".join(href.split())
        if not compact_href.startswith("data:image/png;base64,"):
            return None

        try:
            encoded = compact_href.split("base64,", 1)[1]
            image_bytes = base64.b64decode(encoded)
            source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None

        # If the SVG defines a hex clipPath, apply it so the texture fills the hex better.
        try:
            clip_polygon = None
            for clip_path in root.findall(".//{http://www.w3.org/2000/svg}clipPath"):
                polygon = clip_path.find("{http://www.w3.org/2000/svg}polygon")
                if polygon is not None and polygon.attrib.get("points"):
                    clip_polygon = polygon
                    break
            if clip_polygon is None:
                return source

            points_attr = clip_polygon.attrib.get("points", "")
            raw_points = [chunk.strip() for chunk in points_attr.replace("\n", " ").split(" ") if chunk.strip()]
            svg_points: list[tuple[float, float]] = []
            for raw_point in raw_points:
                if "," not in raw_point:
                    continue
                x_str, y_str = raw_point.split(",", 1)
                svg_points.append((float(x_str), float(y_str)))
            if len(svg_points) < 3:
                return source

            svg_width = self._svg_numeric_attr(root.attrib.get("width"))
            svg_height = self._svg_numeric_attr(root.attrib.get("height"))
            if svg_width <= 0.0 or svg_height <= 0.0:
                view_box = root.attrib.get("viewBox", "").split()
                if len(view_box) == 4:
                    svg_width = float(view_box[2])
                    svg_height = float(view_box[3])
            if svg_width <= 0.0 or svg_height <= 0.0:
                return source

            scale_x = source.width / svg_width
            scale_y = source.height / svg_height
            image_points = [(x * scale_x, y * scale_y) for x, y in svg_points]

            mask = Image.new("L", source.size, 0)
            ImageDraw.Draw(mask).polygon(image_points, fill=255)
            clipped = source.copy()
            clipped.putalpha(mask)

            min_x, min_y, max_x, max_y = self._polygon_int_bounds(image_points)
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(clipped.width, max_x)
            max_y = min(clipped.height, max_y)
            if max_x - min_x > 4 and max_y - min_y > 4:
                clipped = clipped.crop((min_x, min_y, max_x, max_y))
            return clipped.convert("RGBA")
        except Exception:
            return source.convert("RGBA")

    @staticmethod
    def _svg_numeric_attr(raw_value: str | None) -> float:
        if not raw_value:
            return 0.0
        numeric = "".join(ch for ch in raw_value.strip() if ch.isdigit() or ch in ".-")
        if not numeric:
            return 0.0
        try:
            return float(numeric)
        except ValueError:
            return 0.0

    def _get_tile_texture_image(
        self,
        resource: Resource,
        polygon_points: list[tuple[float, float]],
        *,
        bounds: tuple[int, int, int, int],
    ) -> object | None:
        source = self._tile_texture_sources.get(resource)
        if (
            source is None
            or Image is None
            or ImageDraw is None
            or ImageFilter is None
            or ImageOps is None
            or ImageTk is None
        ):
            return None
        if not polygon_points:
            return None

        min_x, min_y, max_x, max_y = bounds
        width = max(4, int(max_x - min_x + 1))
        height = max(4, int(max_y - min_y + 1))
        cache_key = (resource, width, height)
        cached = self._tile_texture_cache.get(cache_key)
        if cached is not None:
            return cached

        resample_lanczos = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        fitted = ImageOps.fit(source, (width, height), method=resample_lanczos, centering=(0.5, 0.5)).convert("RGBA")
        if _TILE_TEXTURE_ZOOM > 1.0:
            zoomed_w = max(width, int(round(width * _TILE_TEXTURE_ZOOM)))
            zoomed_h = max(height, int(round(height * _TILE_TEXTURE_ZOOM)))
            zoomed = fitted.resize((zoomed_w, zoomed_h), resample=resample_lanczos)
            left = max(0, (zoomed_w - width) // 2)
            top = max(0, (zoomed_h - height) // 2)
            fitted = zoomed.crop((left, top, left + width, top + height))

        anti_alias = 3
        mask = Image.new("L", (width * anti_alias, height * anti_alias), 0)
        mask_points = [
            ((point[0] - min_x) * anti_alias, (point[1] - min_y) * anti_alias)
            for point in polygon_points
        ]
        ImageDraw.Draw(mask).polygon(mask_points, fill=255)
        mask = mask.resize((width, height), resample=resample_lanczos)
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.point(lambda value: 255 if value >= _TEXTURE_MASK_THRESHOLD else 0)
        fitted.putalpha(mask)

        photo_image = ImageTk.PhotoImage(fitted)
        if len(self._tile_texture_cache) > 80:
            self._tile_texture_cache.clear()
        self._tile_texture_cache[cache_key] = photo_image
        return photo_image

    def _layout(self) -> tuple[float, float, float]:
        if self._board is None:
            return (1.0, 0.0, 0.0)

        all_points = [corner for tile in self._board.tiles for corner in tile.corner_points]
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)

        width = max(self.winfo_width(), int(self.cget("width")))
        height = max(self.winfo_height(), int(self.cget("height")))
        padding = 40

        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)
        offset_x = padding - min_x * scale + ((width - 2 * padding) - span_x * scale) / 2
        offset_y = padding - min_y * scale + ((height - 2 * padding) - span_y * scale) / 2
        return (scale, offset_x, offset_y)
