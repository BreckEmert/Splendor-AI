# Splendor/Play/common_types.py

from dataclasses import dataclass
from typing import Tuple, NamedTuple, Literal, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .render import Rect


class CardIndex(NamedTuple):
    tier: int
    pos: int


@dataclass(frozen=True)
class FocusTarget:
    kind: Literal["shop", "deck", "reserved"]
    tier: Optional[int] = None  # shop/deck
    pos: Optional[int] = None  # shop (0‑3)
    reserve_idx: Optional[int] = None

    @classmethod
    def from_index(cls, tier: int, pos: int) -> "FocusTarget":
        return cls(
            "shop" if pos < 4 else "deck",
            tier=tier,
            pos=pos if pos < 4 else None,
        )

    @property
    def card_index(self) -> CardIndex | None:
        return CardIndex(self.tier, self.pos) if self.kind == "shop" else None  # type: ignore


# Classes for the clickmap - hit targets and their payloads
BoardCardToken = Tuple[Literal["board_card"], int, int]  # tier, pos (0‑4 / 4 = deck)
ReservedCardToken = Tuple[Literal["reserved_card"], int, int]  # reserve_idx, move_idx
GemToken = Tuple[Literal["gem"], int]  # gem_index 0‑4 (no gold)

ClickToken = Union[BoardCardToken, ReservedCardToken, GemToken]
ClickMap = dict["Rect", ClickToken]  # per‑frame hit‑test atlas
