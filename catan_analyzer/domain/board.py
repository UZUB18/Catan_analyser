from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
EdgeKey = Tuple[int, int]

BOARD_RADIUS = 2
CORNER_ROUNDING = 6


class Resource(str, Enum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"
    DESERT = "desert"


class PortType(str, Enum):
    ANY_3TO1 = "3:1"
    WOOD_2TO1 = "wood 2:1"
    BRICK_2TO1 = "brick 2:1"
    SHEEP_2TO1 = "sheep 2:1"
    WHEAT_2TO1 = "wheat 2:1"
    ORE_2TO1 = "ore 2:1"


@dataclass
class HexTile:
    id: int
    q: int
    r: int
    resource: Resource
    token_number: Optional[int]
    center: Point
    corner_points: Tuple[Point, ...]


@dataclass
class Vertex:
    id: int
    point: Point
    adjacent_hex_ids: Tuple[int, ...]
    adjacent_vertex_ids: Tuple[int, ...]
    port_type: Optional[PortType] = None


@dataclass
class Port:
    id: int
    vertex_ids: Tuple[int, int]
    port_type: PortType
    midpoint: Point


@dataclass
class BoardState:
    tiles: List[HexTile]
    vertices: Dict[int, Vertex]
    ports: List[Port]
    edges: Dict[EdgeKey, Tuple[int, ...]]
    _tile_lookup: Dict[int, HexTile] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tile_lookup = {tile.id: tile for tile in self.tiles}

    def get_tile(self, tile_id: int) -> HexTile:
        return self._tile_lookup[tile_id]

    def vertex_adjacent_tiles(self, vertex_id: int) -> List[HexTile]:
        return [self.get_tile(tile_id) for tile_id in self.vertices[vertex_id].adjacent_hex_ids]

    def blocked_vertices(self, occupied_vertices: Iterable[int]) -> set[int]:
        occupied = set(occupied_vertices)
        blocked = set(occupied)
        for vertex_id in occupied:
            blocked.update(self.vertices[vertex_id].adjacent_vertex_ids)
        return blocked

    def legal_settlement_vertices(self, occupied_vertices: Iterable[int]) -> List[int]:
        blocked = self.blocked_vertices(occupied_vertices)
        return [vertex_id for vertex_id in sorted(self.vertices.keys()) if vertex_id not in blocked]

    def is_legal_settlement(self, vertex_id: int, occupied_vertices: Iterable[int]) -> bool:
        return vertex_id in self.legal_settlement_vertices(occupied_vertices)

    @staticmethod
    def normalize_edge_key(vertex_a: int, vertex_b: int) -> EdgeKey:
        return tuple(sorted((int(vertex_a), int(vertex_b))))

    def edge_exists(self, edge: EdgeKey) -> bool:
        if len(edge) != 2:
            return False
        first, second = int(edge[0]), int(edge[1])
        if first == second:
            return False
        if first not in self.vertices or second not in self.vertices:
            return False
        return self.normalize_edge_key(first, second) in self.edges

    def road_touches_settlement_or_network(
        self,
        edge: EdgeKey,
        settlements: Iterable[int],
        roads: Iterable[EdgeKey],
    ) -> bool:
        if not self.edge_exists(edge):
            return False

        first, second = self.normalize_edge_key(edge[0], edge[1])
        settlement_vertices = {vertex_id for vertex_id in settlements if vertex_id in self.vertices}
        if first in settlement_vertices or second in settlement_vertices:
            return True

        network_vertices: set[int] = set()
        for road in roads:
            if len(road) != 2:
                continue
            if not self.edge_exists(road):
                continue
            road_first, road_second = self.normalize_edge_key(road[0], road[1])
            network_vertices.add(road_first)
            network_vertices.add(road_second)

        return first in network_vertices or second in network_vertices

    def is_legal_road(
        self,
        edge: EdgeKey,
        settlements: Iterable[int],
        roads: Iterable[EdgeKey],
    ) -> bool:
        if len(edge) != 2:
            return False
        if not self.edge_exists(edge):
            return False

        normalized_edge = self.normalize_edge_key(edge[0], edge[1])
        normalized_roads = {
            self.normalize_edge_key(road[0], road[1])
            for road in roads
            if len(road) == 2 and self.edge_exists(road)
        }
        if normalized_edge in normalized_roads:
            return False

        return self.road_touches_settlement_or_network(
            normalized_edge,
            settlements=settlements,
            roads=normalized_roads,
        )


def build_standard_board(
    resource_order: Optional[Sequence[Resource]] = None,
    token_order: Optional[Sequence[int]] = None,
) -> BoardState:
    coords = _generate_axial_coords(BOARD_RADIUS)
    resources = list(resource_order) if resource_order is not None else _default_resource_order()
    numbers = list(token_order) if token_order is not None else _default_number_order()

    if len(resources) != len(coords):
        raise ValueError(f"Expected {len(coords)} resources, received {len(resources)}.")

    if len(numbers) != len(coords) - 1:
        raise ValueError(f"Expected {len(coords) - 1} number tokens, received {len(numbers)}.")

    tiles: List[HexTile] = []
    number_index = 0
    for tile_id, (q, r) in enumerate(coords):
        center = _axial_to_pixel(q, r)
        corners = tuple(_hex_corner(center, corner_index) for corner_index in range(6))
        resource = resources[tile_id]
        if resource is Resource.DESERT:
            token_number = None
        else:
            token_number = numbers[number_index]
            number_index += 1

        tiles.append(
            HexTile(
                id=tile_id,
                q=q,
                r=r,
                resource=resource,
                token_number=token_number,
                center=center,
                corner_points=corners,
            )
        )

    if number_index != len(numbers):
        raise ValueError("Number token assignment does not match non-desert tile count.")

    vertices, edges = _build_vertices_and_edges(tiles)
    ports = _assign_ports(vertices, edges)

    return BoardState(
        tiles=tiles,
        vertices=vertices,
        ports=ports,
        edges={edge_key: tuple(sorted(hex_ids)) for edge_key, hex_ids in edges.items()},
    )


def _default_resource_order() -> List[Resource]:
    return [
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.ORE,
        Resource.WOOD,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.BRICK,
        Resource.DESERT,
        Resource.ORE,
        Resource.WOOD,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.BRICK,
        Resource.ORE,
        Resource.WOOD,
        Resource.SHEEP,
        Resource.WHEAT,
    ]


def _default_number_order() -> List[int]:
    return [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]


def _generate_axial_coords(radius: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            coords.append((q, r))
    coords.sort(key=lambda item: (item[1], item[0]))
    return coords


def _axial_to_pixel(q: int, r: int) -> Point:
    x = math.sqrt(3) * (q + r / 2)
    y = 1.5 * r
    return (x, y)


def _hex_corner(center: Point, corner_index: int) -> Point:
    angle_deg = 60 * corner_index - 30
    angle_rad = math.radians(angle_deg)
    return (
        center[0] + math.cos(angle_rad),
        center[1] + math.sin(angle_rad),
    )


def _build_vertices_and_edges(
    tiles: Sequence[HexTile],
) -> Tuple[Dict[int, Vertex], Dict[EdgeKey, set[int]]]:
    vertex_lookup: Dict[Point, int] = {}
    vertex_points: Dict[int, Point] = {}
    vertex_hexes: Dict[int, set[int]] = {}
    vertex_neighbors: Dict[int, set[int]] = {}
    edge_hexes: Dict[EdgeKey, set[int]] = {}

    next_vertex_id = 0

    for tile in tiles:
        tile_vertex_ids: List[int] = []
        for corner_point in tile.corner_points:
            key = (round(corner_point[0], CORNER_ROUNDING), round(corner_point[1], CORNER_ROUNDING))
            vertex_id = vertex_lookup.get(key)
            if vertex_id is None:
                vertex_id = next_vertex_id
                next_vertex_id += 1
                vertex_lookup[key] = vertex_id
                vertex_points[vertex_id] = corner_point
                vertex_hexes[vertex_id] = set()
                vertex_neighbors[vertex_id] = set()
            tile_vertex_ids.append(vertex_id)
            vertex_hexes[vertex_id].add(tile.id)

        for first, second in zip(tile_vertex_ids, tile_vertex_ids[1:] + tile_vertex_ids[:1]):
            edge_key = tuple(sorted((first, second)))
            edge_hexes.setdefault(edge_key, set()).add(tile.id)
            vertex_neighbors[first].add(second)
            vertex_neighbors[second].add(first)

    vertices: Dict[int, Vertex] = {}
    for vertex_id in sorted(vertex_points.keys()):
        vertices[vertex_id] = Vertex(
            id=vertex_id,
            point=vertex_points[vertex_id],
            adjacent_hex_ids=tuple(sorted(vertex_hexes[vertex_id])),
            adjacent_vertex_ids=tuple(sorted(vertex_neighbors[vertex_id])),
            port_type=None,
        )

    return vertices, edge_hexes


def _assign_ports(vertices: Dict[int, Vertex], edge_hexes: Dict[EdgeKey, set[int]]) -> List[Port]:
    coastal_edges: List[Tuple[float, EdgeKey, Point]] = []
    for edge_key, adjacent_hexes in edge_hexes.items():
        if len(adjacent_hexes) != 1:
            continue
        first, second = edge_key
        first_point = vertices[first].point
        second_point = vertices[second].point
        midpoint = ((first_point[0] + second_point[0]) / 2, (first_point[1] + second_point[1]) / 2)
        angle = math.atan2(midpoint[1], midpoint[0])
        coastal_edges.append((angle, edge_key, midpoint))

    coastal_edges.sort(key=lambda item: item[0])
    if not coastal_edges:
        return []

    raw_indices = _preferred_port_indices(len(coastal_edges))
    port_types = [
        PortType.ANY_3TO1,
        PortType.BRICK_2TO1,
        PortType.ANY_3TO1,
        PortType.ORE_2TO1,
        PortType.ANY_3TO1,
        PortType.SHEEP_2TO1,
        PortType.ANY_3TO1,
        PortType.WHEAT_2TO1,
        PortType.WOOD_2TO1,
    ]

    ports: List[Port] = []
    for port_id, (index, port_type) in enumerate(zip(raw_indices, port_types)):
        _, edge_key, midpoint = coastal_edges[index]
        first, second = edge_key
        if vertices[first].port_type is None:
            vertices[first].port_type = port_type
        if vertices[second].port_type is None:
            vertices[second].port_type = port_type
        ports.append(Port(id=port_id, vertex_ids=(first, second), port_type=port_type, midpoint=midpoint))
    return ports


def _preferred_port_indices(coastal_edge_count: int) -> List[int]:
    template = [0, 3, 6, 10, 13, 16, 20, 23, 26]
    if coastal_edge_count < 27:
        step = coastal_edge_count / 9.0
        template = [int(round(step * index)) % coastal_edge_count for index in range(9)]

    indices: List[int] = []
    used: set[int] = set()
    for raw_index in template:
        index = raw_index % coastal_edge_count
        while index in used:
            index = (index + 1) % coastal_edge_count
        used.add(index)
        indices.append(index)
    return indices
