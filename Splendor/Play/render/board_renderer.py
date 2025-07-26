# Splendor/Play/render/board_renderer.py 

import os
from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

from Environment.gui_game import GUIGame
from Play.render import BoardGeometry, Coord
from Play.render.static_renderer import move_to_text


class BoardRenderer:
    def __init__(
        self,
        resource_root: str = "/workspace/Play/render/Resources",
        font_path: str = "/workspace/Play/render/Resources/arialbd.ttf"
    ):
        # Paths
        self.resource_root = resource_root
        self.images_root = os.path.join(self.resource_root, "images")
        self.font_path = font_path

        # Assets
        self.font = ImageFont.truetype(self.font_path, 120)

        # Runtime state
        self._canvas: Image.Image
        self.geom = BoardGeometry()
        self.draw: ImageDraw.ImageDraw
        self._clickmap: Dict[Tuple[int, int, int, int], tuple] = {}
        self.game: GUIGame

    # Public API
    def render(self, game, buf):
        self.game = game
        self._reset_canvas()

        # Draw board
        self._draw_background()
        self._draw_nobles(game.board.nobles)
        self._draw_board_cards(game.board)
        self._draw_board_gems(game.board)

        # Draw Players
        for player in game.players:
            self._draw_player(player)

        # Draw HUD
        self._draw_turn_indicator(game.half_turns)

        self._save(buf)
        return self._clickmap

    # Internals
    def _reset_canvas(self):
        self._canvas = Image.new("RGB", self.geom.canvas)
        self.draw = ImageDraw.Draw(self._canvas)
        self._clickmap: Dict[Tuple[int, int, int, int], tuple] = {}

    def _mark(self, rect: Tuple[int, int, int, int], payload):
        """Register a board region as clickable payload."""
        self._clickmap[rect] = payload

    def _player_origin(self, pos: int) -> Coord:
        return self.geom.player_origins[pos]

    def _reserve_origin(self, pos: int) -> Coord:
        return self.geom.reserve_origins[pos]

    def _draw_background(self):
        table_path = os.path.join(self.images_root, "table.jpg")
        table_image = Image.open(table_path).convert("RGB").resize(self.geom.canvas)  # SHOULD BE CONVERTED TO BYTESIO?
        self._canvas.paste(table_image, (0, 0))

    def _draw_nobles(self, nobles):
        # Nobles
        noble_x = self.geom.deck_origin.x + self.geom.card.x + 50
        noble_y = self.geom.deck_origin.y
        for noble in nobles:
            noble_x += 50
            if noble:
                noble_path = os.path.join(self.images_root, "nobles", f"{noble.id}.jpg")
                noble_image = Image.open(noble_path)
                self._canvas.paste(noble_image, (noble_x, noble_y))
            noble_x += self.geom.card.x + 50  # Cosmetic only

    def _draw_board_cards(self, board):
        # Tier covers + face‑up cards
        tier_x, tier_y = self.geom.deck_origin.x, self.geom.deck_origin.y + self.geom.card.y + 50

        for reversed_tier, tier_cards in enumerate(reversed(board.cards)):
            tier = 2 - reversed_tier

            # Cover
            cover_path = os.path.join(
                self.images_root,
                str(tier),
                "cover.jpg",
            )
            cover_image = Image.open(cover_path)
            self._canvas.paste(cover_image, (tier_x, tier_y))

            # Face‑up cards
            card_x = tier_x + self.geom.card.x + 50
            for position, card in enumerate(tier_cards):
                if card:
                    card_path = os.path.join(
                        self.images_root,
                        str(tier),
                        f"{card.id}.jpg",
                    )
                    card_image = Image.open(card_path)
                    self._canvas.paste(card_image, (card_x, tier_y))
                    self._mark((
                            card_x,
                            tier_y,
                            card_x + self.geom.card.x,
                            tier_y + self.geom.card.y,
                        ), ("card", tier, position),
                    )
                card_x += self.geom.card.x + 10

            # Deck top (reservable)
            self._mark((
                    tier_x,
                    tier_y,
                    tier_x + self.geom.card.x,
                    tier_y + self.geom.card.y,
                ), ("card", tier, 4),
            )

            tier_y += self.geom.card.y + 50

    def _draw_board_gems(self, board):
        # Board gems
        gem_x = self.geom.deck_origin.x + (self.geom.card.x*5 + 150)
        gem_y = self.geom.deck_origin.y + self.geom.card.y//2

        for gem_index, gem_count in enumerate(board.gems):
            gem_path = os.path.join(
                self.images_root,
                "gems",
                f"{gem_index}.png",
            )
            gem_image = Image.open(gem_path)

            # Gem sprite and count
            self._canvas.paste(
                gem_image,
                (gem_x-20, gem_y-15),
                gem_image.split()[3],
            )
            self.draw.text(
                (gem_x + self.geom.gem.x + 10, gem_y),
                str(gem_count),
                fill=(255, 255, 255),
                font=self.font,
            )

            # Clickable non‑gold tokens
            if gem_index != 5:
                self._mark((
                        gem_x-20,
                        gem_y-15,
                        gem_x + self.geom.gem.x,
                        gem_y + self.geom.gem.y,
                    ), ("gem", gem_index),
                )

            gem_y += self.geom.gem.y + 40

    def _draw_player(self, player):
        # Gems and owned cards
        start_x, start_y = self._player_origin(player.pos)
        current_x, current_y = start_x, start_y

        for gem_index, gem_count in enumerate(player.gems):
            gem_path = os.path.join(self.images_root, "gems", f"{gem_index}.png")
            gem_image = Image.open(gem_path)

            # Token pile
            for _ in range(gem_count):
                self._canvas.paste(
                    gem_image,
                    (current_x, current_y),
                    gem_image.split()[3],
                )
                current_y += int(self.geom.gem.y / 1.7)

            # Permanent bonus cards (skip gold)
            if gem_index != 5:
                current_y = start_y + int(self.geom.gem.y * 2.1)
                for tier, card_id in player.card_ids[gem_index]:
                    card_path = os.path.join(
                        self.images_root,
                        str(tier),
                        f"{card_id}.jpg",
                    )
                    card_image = Image.open(card_path)
                    self._canvas.paste(card_image, (current_x, current_y))
                    current_y += int(self.geom.card.y / 7)

            current_x += self.geom.card.x + 50
            current_y = start_y

        # Reserved cards
        # reserved_origin = (current_x - self.geom.card.x - 50, start_y + 600)  OLD LINE
        self._draw_reserved_cards(player)

        # Last move
        if player is not self.game.active_player:
            self._draw_last_move(player)

    def _draw_reserved_cards(self, player):
        # Reserved cards
        x, y = self._reserve_origin(player.pos)

        for reserve_idx, card in enumerate(player.reserved_cards):
            card_path = os.path.join(
                self.images_root,
                str(card.tier),
                f"{card.id}.jpg",
            )
            card_image = Image.open(card_path)
            self._canvas.paste(card_image, (x, y))

            # Click target for active player
            if player is self.game.active_player:
                move_idx = player.take_dim + 24 + reserve_idx * 2
                self._mark(
                    (x, y, x + self.geom.card.x, y + self.geom.card.y),
                    ("move", move_idx),
                )

            # Fan offset
            x += self.geom.card.x // 3
            y += self.geom.card.y // 3

    def _draw_last_move(self, player):
        """Annotate the board with the last move."""
        move_text = move_to_text(self.game.move_index, player)
        x, y = self._player_origin(player.pos)

        # Place text above the upper player, and below the lower
        text_y = y - 170 if y < self.geom.canvas.y // 2 else y + 1115
        self.draw.text((x, text_y), move_text, fill=(255, 255, 255), font=self.font)

    def _draw_turn_indicator(self, half_turns: int):
        turn_num = half_turns // 2 + 1
        self.draw.text((50, 50), f"Turn {turn_num}",
                       fill=(255, 255, 255), font=self.font)

    def _save(self, buf):
        """Persist composed canvas to the provided buffer."""
        self._canvas.save(buf, format="JPEG")
        buf.seek(0)

    @property
    def clickmap(self) -> Dict[Tuple[int, int, int, int], tuple]:
        return self._clickmap.copy()
