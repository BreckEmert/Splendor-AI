# Splendor/meta/generate_images.py

import json
import os
import shutil
from PIL import Image, ImageDraw, ImageFont


def draw_game_state(game_state, index, image_save_path):
    image_base_path = "/workspace/meta/images"

    # Canvas
    width, height = 5000, 3000
    path = "/workspace/meta/images/table.jpg"
    canvas = Image.open(path).resize((width, height))
    draw = ImageDraw.Draw(canvas)

    draw.text((50, 50), f"Turn order: {index//2 + 1}", fill=(255, 255, 255), font=font)

    # Card and gem sizes
    card_width, card_height = 300, 400
    gem_width, gem_height = 200, 200

    # Board and player offsets
    board_start_x, board_start_y = 200, 680
    p1_start_x, p1_start_y = 2600, 200
    p2_start_x, p2_start_y = 2600, 1700

    # Draw Board Cards
    board_x_offset = board_start_x
    board_y_offset = board_start_y
    noble_spacing = 90
    noble_space = 0
    for tier, card_ids in reversed(game_state['board']['cards'].items()):
        if tier == 'nobles':
            noble_space = noble_spacing
        card_image = Image.open(image_base_path + f"/{tier}/cover.jpg")
        canvas.paste(card_image, (board_x_offset, board_y_offset))
        board_x_offset += card_width + 50
        for card_id in card_ids:
            if card_id is not None:
                card_image_path = image_base_path + f"/{tier}/{card_id}.jpg"
                card_image = Image.open(card_image_path)
                canvas.paste(card_image, (board_x_offset, board_y_offset))
                board_x_offset += card_width + noble_space + 10
        noble_space = 0
        board_y_offset += card_height + 50
        board_x_offset = board_start_x

    # Draw Gems
    gem_x_offset = board_x_offset + card_width * 5 + 150
    gem_y_offset = board_start_y + int(card_height/2)
    for gem in gem_types:
        amount = game_state['board']['gems'][gem_types.index(gem)]
        gem_image_path = image_base_path + f"/gems/{gem}.png"
        gem_image = Image.open(gem_image_path)
        canvas.paste(gem_image, (gem_x_offset-20, gem_y_offset-15), gem_image.split()[3])
        draw.text((gem_x_offset + gem_width + 10, gem_y_offset), 
                  str(amount), fill=(255, 255, 255), font=font)
        gem_y_offset += gem_height + 40

    # Draw players
    for i, (player_name, player_state) in enumerate(game_state['players'].items()):
        if i == 0:
            player_x_offset = p1_start_x
            player_y_offset = p1_start_y
        else:
            player_x_offset = p2_start_x
            player_y_offset = p2_start_y

        # Draw gems
        current_x_offset = player_x_offset
        current_y_offset = player_y_offset
        for gem_index, gem in enumerate(gem_types):
            amount = player_state['gems'][gem_index]
            gem_image_path = image_base_path + f"/gems/{gem}.png"
            gem_image = Image.open(gem_image_path)
            for _ in range(amount):
                canvas.paste(gem_image, (current_x_offset, current_y_offset), gem_image.split()[3]) # trying alpha
                current_y_offset += int(gem_height/1.7)

            # Draw cards for the same type
            if gem != 'gold':
                current_y_offset = player_y_offset + int(gem_height*2.1)
                for tier_index in range(3):
                    tier = f"tier{tier_index+1}"
                    for card_id in player_state['cards'][tier_index][gem_index]:
                        card_image_path = image_base_path + f"/{tier}/{card_id}.jpg"
                        card_image = Image.open(card_image_path)
                        canvas.paste(card_image, (current_x_offset, current_y_offset))
                        current_y_offset += int(card_height/7)
            current_x_offset += card_width + 50
            current_y_offset = player_y_offset

        # Draw Reserved Cards
        current_x_offset -= card_width + 50
        current_y_offset = player_y_offset + 600
        for tier_index, card_id in player_state['reserved_cards']:
            tier = f"tier{tier_index+1}"
            card_image_path = image_base_path + f"/{tier}/{card_id}.jpg"
            card_image = Image.open(card_image_path)
            canvas.paste(card_image, (current_x_offset, current_y_offset))
            current_x_offset += int(card_width/3)
            current_y_offset += int(card_height/3)

        # Draw move and reward
        move = move_to_text(player_state['chosen_move'])
        if move:
            draw.text((int((p1_start_x+width/2)/2), 30+i*2800), move, fill=(255, 255, 255), font=font)

    # Save the image
    canvas.save(image_save_path)

def move_to_text(move_index):
    mapping = {
        0: 'white', 
        1: 'blue', 
        2: 'green', 
        3: 'red', 
        4: 'brown', 
        5: 'gold'
    }

    tier = move_index % 15 // 4
    card_index = move_index % 15 % 4

    if move_index == 9999:
        return ""
    
    if move_index < 15:
        gem_index = move_index % 5
        if move_index < 5:
            return f"Take 1 {mapping[gem_index]}"
        elif move_index >= 10:
            return f"Discard 1 {mapping[gem_index]}"
        else:
            return f"Take 2 {mapping[gem_index]}"
    
    elif move_index < 27:
        return f"Buy tier {tier + 1}, position {card_index + 1}"
    
    elif move_index < 30:
        return f"Buy reserved {card_index + 1}"
    
    elif move_index < 42:
        return f"Buy tier {tier + 1}, position {card_index + 1} with gold"
    
    elif move_index < 45:
        return f"Buy reserved {card_index + 1} with gold"
    
    elif move_index < 57:
        return f"Reserve tier {tier + 1}, position {card_index + 1}"
    
    else:
        return f"Reserve top card from tier {move_index - 57 + 1}"

def main(log_path, output_image_path):
    for game_states in os.listdir(log_path):
        if game_states.endswith('.json'):
            game_folder = f"game {os.path.splitext(game_states)[0][-3:]}"
            game_output_path = os.path.join(output_image_path, game_folder)
            os.makedirs(game_output_path, exist_ok=True)
            with open(os.path.join(log_path, game_states), 'r') as f:
                for index, line in enumerate(f):
                    game_state = json.loads(line)
                    output_file = os.path.join(game_output_path, f"state_{index}.jpg")
                    print(f"drawing state_{index}.jpg...")
                    draw_game_state(game_state, index, output_file)

if __name__ == '__main__':
    model_type = "random"  # ddqn, random

    base_path = "/workspace/RL/saved_files"
    log_path = os.path.join(base_path, "game_states", "random")
    image_save_path = os.path.join(base_path, "rendered_games", model_type)
    shutil.rmtree(image_save_path, ignore_errors=True)
    os.makedirs(image_save_path, exist_ok=True)

    gem_types = ['white', 'blue', 'green', 'red', 'black', 'gold']
    font_path = "/workspace/meta/arialbd.ttf"
    font = ImageFont.truetype(font_path, 120)

    main(log_path, image_save_path)
