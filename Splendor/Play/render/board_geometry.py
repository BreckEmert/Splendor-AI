# Splendor/Play/render/board_geometry.py

from dataclasses import dataclass
from typing import Tuple, NamedTuple


class Coord(NamedTuple):
    x: int
    y: int


@dataclass(frozen=True)
class BoardGeometry:
    canvas: Coord = Coord(5000, 3000)
    card: Coord = Coord(300, 400)
    card_gap: Coord = Coord(10, 50)
    gem: Coord = Coord(200, 200)
    deck_origin: Coord = Coord(200, 680)
    deck_gap: Coord = Coord(50, 50)
    player_origins: Tuple[Coord, ...] = (
        Coord(2600, 200), Coord(2600, 1700)
    )
    reserve_origins: Tuple[Coord, ...] = (
        Coord(4350, 800), Coord(4350, 2300)
    )
