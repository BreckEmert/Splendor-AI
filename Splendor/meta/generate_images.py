# Splendor/meta/generate_images.py

import os
import itertools as it
from PIL import Image, ImageDraw, ImageFont

# Global references
gem_types = ['white', 'blue', 'green', 'red', 'black', 'gold']
font_path = "/workspace/meta/arialbd.ttf"
font = ImageFont.truetype(font_path, 120)
take_3_indices = list(it.combinations(range(5), 3))
take_2_diff_indices = list(it.combinations(range(5), 2))

# Convert a move index into text
def move_to_text(move_index):
    if move_index < 95:
        if move_index < 40:
            local_index = move_index
            combo_index = local_index // 4
            discard_idx = local_index % 4
            combo = take_3_indices[combo_index]
            combo_str = ", ".join(gem_types[color] for color in combo)
            return (f"Take 3 different: {combo_str}" if discard_idx == 0
                    else f"Take 3: {combo_str} (discard {discard_idx})")
        elif move_index < 55:
            local_index = move_index - 40
            gem_index = local_index // 3
            discard_idx = local_index % 3
            color = gem_types[gem_index]
            return (f"Take 2 same: {color}" if discard_idx == 0
                    else f"Take 2: {color} (discard {discard_idx})")
        elif move_index < 85:
            local_index = move_index - 55
            combo_index = local_index // 3
            discard_idx = local_index % 3
            combo = take_2_diff_indices[combo_index]
            combo_str = " & ".join(gem_types[color] for color in combo)
            return (f"Take 2 different: {combo_str}" if discard_idx == 0
                    else f"Take 2: {combo_str} (discard {discard_idx})")
        else:
            local_index = move_index - 85
            gem_index = local_index // 2
            discard_idx = local_index % 2
            color = gem_types[gem_index]
            return (f"Take 1 {color}" if discard_idx == 0
                    else f"Take 1 {color} (discard {discard_idx})")
    elif move_index < 125:
        buy_index = move_index - 95
        if buy_index < 24:
            board_card_idx = buy_index // 2
            with_gold_flag = buy_index % 2
            tier = board_card_idx // 4
            position = board_card_idx % 4
            txt = f"Buy tier {tier+1}, pos {position+1}"
            return txt if with_gold_flag == 0 else txt + " [gold]"
        else:
            reserved_idx = (buy_index - 24) // 2
            with_gold_flag = (buy_index - 24) % 2
            txt = f"Buy reserved slot {reserved_idx+1}"
            return txt if with_gold_flag == 0 else txt + " [gold]"
    else:
        reserve_index = move_index - 125
        tier = reserve_index // 5
        card_pos = reserve_index % 5
        return (f"Reserve from tier {tier+1}, pos {card_pos+1}"
                if card_pos < 4
                else f"Reserve top card from tier {tier+1}")

# Draw the game
def render_game_state(game, image_save_path):
    board = game.board
    turn = game.half_turns

    # Blank canvas, paths, and start indices
    base_path = "/workspace/meta/images"
    table_path = os.path.join(base_path, "table.jpg")

    width, height = 5000, 3000
    canvas = Image.open(table_path).resize((width, height))

    card_width, card_height = 300, 400
    gem_width, gem_height   = 200, 200

    board_start_x, board_start_y = 200, 680
    p1_start_x, p1_start_y       = 2600, 200
    p2_start_x, p2_start_y       = 2600, 1700

    # Draw the move text
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 50), f"Turn number: {turn//2 + 1}", fill=(255, 255, 255), font=font)

    # Draw nobles
    x_offset = board_start_x + card_width + 50
    y_offset = board_start_y
    for noble in board.nobles:
        x_offset += 50
        if noble is not None:
            noble_path = os.path.join(base_path, "nobles", f"{noble.id}.jpg")
            noble_img  = Image.open(noble_path)
            canvas.paste(noble_img, (x_offset, y_offset))
        x_offset += card_width + 50

    # Draw board cards
    board_x_offset = board_start_x
    y_offset += noble_img.height + 50

    for i, tier in enumerate(reversed(board.cards)):
        i = 2 - i
        cover_path = os.path.join(base_path, str(i), "cover.jpg")
        cover_img = Image.open(cover_path)
        canvas.paste(cover_img, (board_x_offset, y_offset))
        x_offset = board_x_offset + card_width + 50

        # Loop through cards in the tier
        for card in tier:
            if card is not None:
                card_image_path = os.path.join(base_path, str(i), f"{card.id}.jpg")
                card_image = Image.open(card_image_path)
                canvas.paste(card_image, (x_offset, y_offset))

            x_offset += card_width + 10

        y_offset += card_height + 50
        board_x_offset = board_start_x

    # Draw board gems
    gem_x_offset = board_start_x + (card_width*5 + 150)
    gem_y_offset = board_start_y + int(card_height/2)

    for i, gems in enumerate(board.gems):
        gem_img_path = os.path.join(base_path, "gems", f"{i}.png")
        gem_image = Image.open(gem_img_path)
        canvas.paste(gem_image, (gem_x_offset - 20, gem_y_offset - 15), gem_image.split()[3])
        draw.text((gem_x_offset + gem_width + 10, gem_y_offset),
                  str(gems), fill=(255, 255, 255), font=font)
        gem_y_offset += (gem_height + 40)

    # Draw players
    start_positions = [(p1_start_x, p1_start_y), (p2_start_x, p2_start_y)]
    for (player, (p_start_x, p_start_y)) in zip(game.players, start_positions):
        current_x = p_start_x
        current_y = p_start_y

        # Gems and owned cards
        for i, gems in enumerate(player.gems):
            gem_image_path = os.path.join(base_path, "gems", f"{i}.png")
            gem_image = Image.open(gem_image_path)
            for _ in range(gems):
                canvas.paste(gem_image, (current_x, current_y),
                             gem_image.split()[3])
                current_y += int(gem_height / 1.7)

            # Owned cards for each tier (except gold)
            if i != 5:
                current_y = p_start_y + int(gem_height * 2.1)
                for (tier, id) in player.card_ids[i]:
                    card_img_path = os.path.join(base_path, str(tier), f"{id}.jpg")
                    card_image = Image.open(card_img_path)
                    canvas.paste(card_image, (current_x, current_y))

                    current_y += int(card_height / 7)

            current_x += (card_width + 50)
            current_y = p_start_y

        # Reserved cards
        current_x -= (card_width + 50)
        current_y += 600
        for card in player.reserved_cards:
            img_path = os.path.join(base_path, str(card.tier), f"{card.id}.jpg")
            card_image = Image.open(img_path)
            canvas.paste(card_image, (current_x, current_y))

            current_x += int(card_width / 3)
            current_y += int(card_height / 3)
        
        # Player move (turn count was incremented so we have to do 'not')
        if player is not game.active_player:
            move_str = move_to_text(game.move_index)
            y_offset = -170 if player is game.players[0] else 1115
            draw.text((p_start_x+200, p_start_y + y_offset),
                        move_str, fill=(255, 255, 255), font=font)
            
    canvas.save(image_save_path)

# function called from training.py
def draw_game_state(episode, game):
    output_dir = game.model.paths['images_dir']
    output_dir = os.path.join(output_dir, f'episode {episode}')
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(output_dir, f"turn_{game.half_turns}.jpg")
    render_game_state(game, image_path)
