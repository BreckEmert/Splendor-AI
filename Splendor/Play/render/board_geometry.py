# Splendor/Play/render/board_geometry.py

from dataclasses import dataclass
from typing import Tuple, NamedTuple, TYPE_CHECKING
if TYPE_CHECKING:
    import pygame


class Coord(NamedTuple):
    """Coordinate."""
    x: int
    y: int


class Size(NamedTuple):
    """Width and height of Rect."""
    w: int
    h: int


class Rect(NamedTuple):
    """Rectangle."""
    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def from_size(cls, x0: int, y0: int, w: int, h: int) -> "Rect":
        return cls(x0, y0, x0+w, y0+h)

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0
    
    @property
    def size(self) -> tuple:
        return self.w, self.h

    def contains(self, x: int, y: int) -> bool:
        """Whether or not a point is contained within the Rect."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def scaled(self, sx: float, sy: float) -> "Rect":
        """Return a copy in scaled window-space."""
        return Rect(
            int(self.x0 * sx), int(self.y0 * sy),
            int(self.x1 * sx), int(self.y1 * sy),
        )

    def to_pygame(self) -> "pygame.Rect":
        import pygame
        return pygame.Rect(self.x0, self.y0, self.w, self.h)


@dataclass(frozen=True)
class BoardGeometry:
    """Relevant coordinates for the board."""
    # Board elements
    canvas: Coord = Coord(2500, 1500)
    default_canvas_scale: Coord = Coord(1600, 960)
    card: Coord = Coord(150, 200)
    noble: Coord = Coord(140, 140)
    gem: Coord = Coord(95, 95)

    # Offsets
    deck_offset: Size = Size(25, 0)
    board_card_offset: Size = Size(8, 25)
    noble_offset: Size = Size(29, 0)
    board_gem_offset: Size = Size(0, 25)
    board_gem_text_offset: Size = Size(10, 0)
    player_card_offset: Size = Size(50, 50)
    reserve_offset: Size = Size(50, 66)

    # UI elements
    button: Coord = Coord(155, 50)
    confirm_origin: Coord = Coord(500, 1300)
    reward_origin: Coord = Coord(50, 1420)

    # Origins
    gem_origin: Coord = Coord(950, 470)
    deck_origin: Coord = Coord(100, 470)
    shop_origin: Coord = Coord(275, 470)
    nobles_origin: Coord = Coord(335, 290)
    player_origins: Tuple[Coord, Coord] = (
        Coord(1250, 100), 
        Coord(1250, 850)
    )
    reserve_origins: Tuple[Coord, Coord] = (
        Coord(2225, 400), 
        Coord(2225, 1150)
    )
    move_text_origins: Tuple[Coord, Coord] = (
        Coord(1250, 10), 
        Coord(1250, 1400)
    )

    def player_origin(self, pos: int) -> Coord:
        return self.player_origins[pos]

    def reserve_origin(self, pos: int) -> Coord:
        return self.reserve_origins[pos]
    
    def move_text_origin(self, pos: int) -> Coord:
        return self.move_text_origins[pos]
