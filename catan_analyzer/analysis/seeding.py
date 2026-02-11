from __future__ import annotations

import hashlib
from typing import Any

from catan_analyzer.domain.board import BoardState


def analysis_seed(board: BoardState, requested_seed: int | None, *, salt: str) -> int:
    """Return deterministic seed for analysis passes.

    If the user supplied `requested_seed`, preserve it exactly. Otherwise derive
    a stable seed from board state + salt so repeated runs on the same board are
    reproducible.
    """
    if requested_seed is not None:
        return int(requested_seed)

    signature = _board_signature(board)
    payload = f"{signature}|{salt}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def derive_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _board_signature(board: BoardState) -> str:
    tile_bits = [
        f"{tile.id}:{tile.resource.value}:{tile.token_number if tile.token_number is not None else 'D'}"
        for tile in sorted(board.tiles, key=lambda item: item.id)
    ]
    port_bits = [
        f"{port.id}:{port.port_type.value}:{port.vertex_ids[0]}-{port.vertex_ids[1]}"
        for port in sorted(board.ports, key=lambda item: item.id)
    ]
    return ";".join([*tile_bits, *port_bits])
