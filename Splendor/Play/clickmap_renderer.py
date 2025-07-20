# Splendor/Play/clickmap_renderer.py

"""
Augments generate_images.render_game_state() to return a click-map.

Dict[(x0,y0,x1,y1)] -> move_index
Coordinates mapped to the 5000×3000 frame.
"""

import os
from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

# from meta.generate_images import gem_types, take_3_indices, take_2_diff_indices # Not needed?
from meta.generate_images import move_to_text


# Public API
__all__ = ["render_game_state", "draw_game_state"]

# Constants
font_path = "/workspace/meta/arialbd.ttf"
font = ImageFont.truetype(font_path, 120)

card_width, card_height = 300, 400
gem_width, gem_height = 200, 200
canvas_width, canvas_height = 5000, 3000


def render_game_state(game, image_save_path) -> Dict[Tuple[int, int, int, int], int]:
    """Draw game state and create clickmap."""
    clickmap = {}
    board = game.board
    turn = game.half_turns

    # Background
    base_path = "/workspace/meta/images"
    table_path = os.path.join(base_path, "table.jpg")
    canvas = Image.open(table_path).resize((canvas_width, canvas_height))
    draw = ImageDraw.Draw(canvas)

    board_x, board_y = 200, 680
    player1_x, player1_y = 2600, 200
    player2_x, player2_y = 2600, 1700
    start_positions = [(player1_x, player1_y), (player2_x, player2_y)]

    # Clickable area register
    def mark(rect, index):
        clickmap[rect] = index

    # Nobles
    noble_x = board_x + card_width + 50
    noble_y = board_y
    for noble in board.nobles:
        noble_x += 50
        if noble:
            path = os.path.join(base_path, "nobles", f"{noble.id}.jpg")
            image = Image.open(path)
            canvas.paste(image, (noble_x, noble_y))
            mark((noble_x, noble_y, noble_x+card_width, noble_y+card_height), noble.move_index)
        noble_x += card_width + 50

    # Board cards
    tier_x = board_x
    tier_y = noble_y + card_height + 50
    for reversed_tier, cards in enumerate(reversed(board.cards)):
        tier = 2 - reversed_tier
        cover = Image.open(os.path.join(base_path, str(tier), "cover.jpg"))
        canvas.paste(cover, (tier_x, tier_y))

        card_x = tier_x + card_width + 50
        for position, card in enumerate(cards):
            if card:
                card_path = os.path.join(base_path, str(tier), f"{card.id}.jpg")
                card_image = Image.open(card_path)
                canvas.paste(card_image, (card_x, tier_y))

                board_position = tier * 4 + position
                for gold in (0, 1):
                    move_index = game.active_player.take_dim + board_position * 2 + gold
                    mark((card_x, tier_y, card_x+card_width, tier_y+card_height), move_index)
            card_x += card_width + 10
        tier_y += card_height + 50

    # Board gems
    gem_x = board_x + (card_width*5 + 150)
    gem_y = board_y + card_height//2
    for gem_index, gem_count in enumerate(board.gems):
        gem_image = Image.open(os.path.join(base_path, "gems", f"{gem_index}.png"))
        canvas.paste(gem_image, (gem_x-20, gem_y-15), gem_image.split()[3])
        draw.text((gem_x+gem_width+10, gem_y), str(gem_count), fill=(255, 255, 255), font=font)

        base_index = gem_index * 2 + 85
        for discard in (0, 1):
            index = base_index + discard
            mark((gem_x-20, gem_y-15, gem_x+gem_width, gem_y+gem_height), index)
        gem_y += gem_height + 40

    # Players
    for player_idx, (player, (player_x, player_y)) in enumerate(
        zip(game.players, start_positions)
    ):
        current_x, current_y = player_x, player_y

        # Gems and owned cards
        for gem_index, gem_count in enumerate(player.gems):
            gem_path = os.path.join(base_path, "gems", f"{gem_index}.png")
            gem_image = Image.open(gem_path)

            # Pile gems
            for _ in range(gem_count):
                canvas.paste(gem_image, (current_x, current_y), gem_image.split()[3])
                current_y += int(gem_height/1.7)

            # Owned cards (skip gold)
            if gem_index != 5:
                current_y = player_y + int(gem_height*2.1)
                for tier, card_id in player.card_ids[gem_index]:
                    card_path = os.path.join(base_path, str(tier), f"{card_id}.jpg")
                    card_image = Image.open(card_path)
                    canvas.paste(card_image, (current_x, current_y))
                    current_y += int(card_height/7)

            current_x += card_width + 50
            current_y = player_y

        # Reserved cards
        current_x -= card_width + 50
        current_y += 600
        for reserve_index, card in enumerate(player.reserved_cards):
            reserved_path = os.path.join(base_path, str(card.tier), f"{card.id}.jpg")
            reserved_image = Image.open(reserved_path)
            canvas.paste(reserved_image, (current_x, current_y))

            # Add BUY_RESERVED click target
            if player is game.active_player:
                move_index = player.take_dim + 24 + reserve_index*2
                mark(
                    (current_x, current_y,
                    current_x+card_width, current_y+card_height),
                    move_index
                )

            current_x += int(card_width/3)
            current_y += int(card_height/3)

        # Last move annotation
        if player is not game.active_player:
            move_text = move_to_text(game.move_index, player)
            text_y = player_y - 170 if player_idx == 0 else player_y + 1115
            draw.text(
                (player_x, text_y), move_text,
                fill=(255, 255, 255), font=font
            )

    # Turn indicator
    draw.text((50, 50), f"Turn {turn//2+1}", fill=(255, 255, 255), font=font)

    canvas.save(image_save_path)

    return clickmap


def draw_game_state(episode, game):
    """Wrapper for training."""
    output_directory = os.path.join(game.model.paths['images_dir'], f'episode {episode}')
    os.makedirs(output_directory, exist_ok=True)

    image_path = os.path.join(output_directory, f"turn_{game.half_turns}.jpg")
    render_game_state(game, image_path)
