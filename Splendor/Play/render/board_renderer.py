# Splendor/Play/render/board_renderer.py 

import os
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from Environment.gui_game import GUIGame
from Play import ClickMap, ClickToken
from Play.render import BoardGeometry, Rect, Coord
from Play.render.static_renderer import move_to_text


class BoardRenderer:
    def __init__(
        self,
        resource_root: str = "/workspace/Play/render/Resources",
        font_path: str = "/workspace/Play/render/Resources/arialbd.ttf"
    ):
        # Assets
        self.resource_root = resource_root
        self.img_root = os.path.join(self.resource_root, "images")
        self._img_cache: dict[str, Image.Image] = {}
        self.font_path = font_path
        self.font = ImageFont.truetype(self.font_path, 60)

        # Runtime state
        self._canvas: Image.Image
        self.geom = BoardGeometry()
        self.draw: ImageDraw.ImageDraw
        self._clickmap: ClickMap = {}
        self.game: GUIGame

    # Public API
    def render(self, game: GUIGame, buf: BytesIO):
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
    def _load(self, path: str, target: Coord, alpha: bool = False) -> Image.Image:
        if path not in self._img_cache:
            img = Image.open(path)
            img = img.resize((target.x, target.y), Image.Resampling.LANCZOS)
            if alpha:
                img = img.convert("RGBA")
            self._img_cache[path] = img
        return self._img_cache[path]

    def _reset_canvas(self):
        self._canvas = Image.new("RGB", self.geom.canvas)
        self.draw = ImageDraw.Draw(self._canvas)
        self._clickmap = {}

    def _mark(self, rect: Rect, payload: ClickToken):
        """Register a board region as clickable payload."""
        self._clickmap[rect] = payload

    def _draw_background(self):
        canvas_path = os.path.join(self.img_root, "table.jpg")
        canvas_image = self._load(canvas_path, self.geom.canvas)
        self._canvas.paste(canvas_image, (0, 0))

    def _draw_nobles(self, nobles):
        g = self.geom
        x, y = g.nobles_origin
        for noble in nobles:
            if noble:
                noble_path = os.path.join(self.img_root, "nobles", f"{noble.id}.jpg")
                noble_image = self._load(noble_path, g.noble)
                self._canvas.paste(noble_image, (x, y))
            x += g.card.x + g.noble_offset.w

    def _draw_board_cards(self, board):
        # Tier covers + face‑up cards
        g = self.geom
        x, y = g.shop_origin
        deck_x = g.deck_origin.x
        card_width, card_height = g.card

        for reversed_tier, tier_cards in enumerate(reversed(board.cards)):
            tier = 2 - reversed_tier

            # Deck cover
            cover_path = os.path.join(self.img_root, str(tier), "cover.jpg",)
            cover_image = self._load(cover_path, g.card)
            self._canvas.paste(cover_image, (deck_x, y))
            self._mark(
                Rect.from_size(deck_x, y, *g.card), 
                ("board_card", tier, 4),
            )

            # Face‑up cards
            for position, card in enumerate(tier_cards):
                if card:
                    card_path = os.path.join(
                        self.img_root,
                        str(tier),
                        f"{card.id}.jpg",
                    )
                    card_image = self._load(card_path, g.card)
                    self._canvas.paste(card_image, (x, y))
                    self._mark(
                        Rect.from_size(x, y, *g.card), 
                        ("board_card", tier, position),
                    )
                x += card_width + g.card_offset.w

            x = g.shop_origin.x
            y += card_height + g.card_offset.h

    def _draw_reserved_cards(self, player):
        # Reserved cards
        g = self.geom
        x, y = g.reserve_origin(player.pos)

        for reserve_idx, card in enumerate(player.reserved_cards):
            card_path = os.path.join(self.img_root, str(card.tier), f"{card.id}.jpg")
            card_image = self._load(card_path, g.card)
            self._canvas.paste(card_image, (x, y))

            # Click target only for active player
            if player is self.game.active_player:
                move_idx = player.take_dim + 24 + reserve_idx * 2
                self._mark(
                    Rect.from_size(x, y, *g.card), 
                    ("reserved_card", reserve_idx, move_idx),
                )

            # Fan offset
            x += g.reserve_offset.w
            y += g.reserve_offset.h

    def _draw_board_gems(self, board):
        g = self.geom
        gem_x, gem_y = g.gem_origin

        for gem_index, gem_count in enumerate(board.gems):
            gem_path = os.path.join(self.img_root, "gems", f"{gem_index}.png")
            gem_image = self._load(gem_path, g.gem)

            # Gem sprite and count
            self._canvas.paste(
                gem_image,
                (gem_x-20, gem_y-15),
                gem_image.split()[3],
            )
            self.draw.text(
                (gem_x + g.gem.x + g.board_gem_text_offset.w, gem_y),
                str(gem_count),
                fill=(255, 255, 255),
                font=self.font,
            )

            # Clickable non‑gold gems
            if gem_index != 5:
                self._mark(
                    Rect.from_size(gem_x-20, gem_y-15, *g.gem), 
                    ("gem", gem_index),
                )

            gem_y += g.gem.y + g.board_gem_offset.h

    def _draw_player(self, player):
        # Gems and owned cards
        g = self.geom
        start_x, start_y = g.player_origin(player.pos)
        current_x, current_y = start_x, start_y

        for gem_index, gem_count in enumerate(player.gems):
            gem_path = os.path.join(self.img_root, "gems", f"{gem_index}.png")
            gem_image = self._load(gem_path, g.gem)

            # Gem pile
            for _ in range(gem_count):
                self._canvas.paste(
                    gem_image,
                    (current_x, current_y),
                    gem_image.split()[3],
                )
                current_y += int(g.gem.y / 1.7)

            # Permanent bought cards
            if gem_index != 5:
                current_y = start_y + int(g.gem.y * 2.1)
                for tier, card_id in player.card_ids[gem_index]:
                    card_path = os.path.join(self.img_root, str(tier), f"{card_id}.jpg")
                    card_image = self._load(card_path, g.card)
                    self._canvas.paste(card_image, (current_x, current_y))
                    current_y += int(g.card.y / 7)

            current_x += g.card.x + 50
            current_y = start_y

        # Reserved cards
        self._draw_reserved_cards(player)

        # Draw the other player's previous move for context
        if player is not self.game.active_player:
            self._draw_last_move(player)

    def _draw_last_move(self, player):
        """Annotate the board with the last move."""
        move_text = move_to_text(self.game.move_idx, player)
        x, y = self.geom.move_text_origin(player.pos)
        self.draw.text((x, y), move_text, fill=(255, 255, 255), font=self.font)

    def _draw_turn_indicator(self, half_turns: int):
        turn_num = half_turns // 2 + 1
        self.draw.text((50, 50), f"Turn {turn_num}",
                       fill=(255, 255, 255), font=self.font)

    def _save(self, buf):
        self._canvas.save(buf, format="JPEG")
        buf.seek(0)

    @property
    def clickmap(self) -> ClickMap:
        return self._clickmap.copy()
