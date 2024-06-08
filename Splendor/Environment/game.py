# Splendor/Environment/game.py


from Environment.Splendor_components import Board # type: ignore
from Environment.Splendor_components import Player # type: ignore

class Game:
    def __init__(self, players):
        self.num_players = len(players)

        self.board = Board(self.num_players)
        self.players: list = [Player(name, strategy, strategy_strength, layer_sizes, model_path) 
                              for name, strategy, strategy_strength, layer_sizes, model_path in players]

        self.reward = 0
        self.active_player = 0
        self.half_turns: int = 0
        self.is_final_turn: bool = False
        self.victor = 0
    
    def turn(self):
        if self.is_final_turn:
            self.victor = self.get_victor()
            self.active_player.victor = True

        self.reward = 0
        self.active_player = self.players[(self.half_turns + 1) % self.num_players]
        prev_state = self.to_vector()

        chosen_move = self.active_player.choose_move(self.board, prev_state)
        print(self.half_turns, chosen_move)
        self.apply_move(chosen_move)

        self.check_noble_visit()
        if self.active_player.points >= 15:
            self.is_final_turn = True

        self.half_turns += 1

    def apply_move(self, move):
        action, (tier, position) = move
        match action:
            case 'go for':
                position, interfere_percentage = position
                # NEED TO REIMPLMENT TAKING GEMS DETAILS BUT FOR MAKING THE MOVE
                # 0 interfere means go 100% for your own and 1 means go as much competing that still aligns with your own
                # Need a 2 which means go 100% for their card
                card = self.board.cards(tier, position)
                self.active_player.go_for_card(interfere_percentage)
                # IMPLEMENT
                self.board.change_gems({gem: amount for gem, amount in details.items()})
                self.active_player.change_gems({gem: amount for gem, amount in details.items()})
                self.reward -= 1


            case 'buy':
                bought_card = self.board.take_card(tier, position)
                self.active_player.get_bought_card(bought_card)

                self.board.change_gems(bought_card.cost)
                self.active_player.change_gems(bought_card.cost)
                self.reward += bought_card.points
            case 'buy_with_gold':
                position, spent_gems = position
                bought_card = self.board.take_card(tier, position)
                self.active_player.get_bought_card(bought_card)

                self.board.change_gems(spent_gems)
                self.active_player.change_gems(spent_gems)
            case 'buy_reserved':
                bought_card = self.active_player.reserved_cards.pop(position)
                self.active_player.get_bought_card(bought_card)

                self.board.change_gems(bought_card.cost)
                self.active_player.change_gems(bought_card.cost)
                
            case 'buy_reserved_with_gold':
                position, spent_gems = position
                bought_card = self.active_player.reserved_cards.pop(position)
                self.active_player.get_bought_card(bought_card)

                self.board.change_gems(spent_gems)
                self.active_player.change_gems(spent_gems)
            case 'reserve':
                tier, position = details
                reserved_card, gold = self.board.reserve(tier, position)
                self.active_player.reserve_card(reserved_card)

                self.active_player.gems['gold'] += gold
            case 'reserve_top':
                reserved_card, gold = self.board.reserve_from_deck(tier)
                self.active_player.reserve_card(reserved_card)

                self.active_player.gems['gold'] += gold

    def handle_gem_selection(self, self_tier, self_pos, enemy_tier, enemy_pos, go_for_enemy_value):
        from itertools import combinations
        #region Calculate the combinations of gems that can be taken
        non_zero_board_gems = [gem for gem in self.board.gems if gem != 'gold' and self.board.gems[gem] > 0]
        non_zero_player_gems = [gem for gem in self.gems if gem != 'gold' and self.gems[gem] > 0]
        take_1 = take_2 = take_2_diff = take_3 = []
        take_moves = []
        
        # Precompute combinations
        combinations_3 = list(combinations(non_zero_board_gems, 3))
        combinations_2 = list(combinations(non_zero_board_gems, 2))

        # Take 3 different gems
        num_discards = min(3, (sum(self.gems.values()) + 3) - 10)
        for combo in combinations_3:
            take_3.append(('take', {combo[0]: -1, combo[1]: -1, combo[2]: -1}))
        if num_discards > 0:
            take_moves += self.handle_discards(take_3, non_zero_player_gems, num_discards)
        else:
            take_moves += take_3

        # Take 2 of the same gem if at least 4 available
        num_discards -= 1
        for gem, count in self.board.gems.items():
            if gem != 'gold' and count >= 4:
                take_2.append(('take', {gem: -2}))
        if num_discards > 0:
            take_moves += self.handle_discards(take_2, non_zero_player_gems, num_discards)
        else:
            take_moves += take_2

        # Take 2 different gems if no legal take 3s
        if not take_3:
            for combo in combinations_2:
                take_2_diff.append(('take', {combo[0]: -1, combo[1]: -1}))
        if num_discards > 0:
            take_moves += self.handle_discards(take_2_diff, non_zero_player_gems, num_discards)
        else:
            take_moves += take_2_diff

        # Take 1 gem if no legal takes
        num_discards -= 1
        if not take_2_diff:
            for gem, count in self.board.gems.items():
                if gem != 'gold' and count > 0:
                    take_1.append(('take', {gem: -1}))
        if num_discards > 0:
            take_moves += self.handle_discards(take_1, non_zero_player_gems, num_discards)
        else:
            take_moves += take_1
        #endregion
        
        # Calculate combinations of own card gems to go for and combinations of enemy card gems to go for
        # We will have go_for_enemy_value 0=entirely own card gems, 3=50% yours 50% enemies, 5=entirely enemy card gems
        # Now, do union(take_moves, go_for_enemy_value moves we just calculated)
        # Implement tie-breaking logic (just random chance probably unless we think of something better)

        

        # Change gems
        self.change_gems(selected_gems)
        self.active_player.change_gems(selected_gems)

    def check_noble_visit(self):
        for noble in self.board.cards['nobles']:
            if all(self.active_player.cards[gem] >= amount for gem, amount in noble.cost.items()):
                self.reward += noble.points
                self.active_player.points += noble.points
                self.board.cards['nobles'].remove(noble)
                break # Implement logic to choose the noble if tied

    def get_victor(self):
        victor = max(self.players, key=lambda p: p.points)
        return victor
   
    def get_state(self):
        return {
            'board': self.board.get_state(),
            'players': {player.name: player.get_state() for player in self.players},
            'current_half_turn': self.half_turns
        }

    def to_vector(self):
        state_vector = self.board.to_vector()
        for player in self.players:
            state_vector.extend(player.to_vector())
        state_vector.append(int(self.is_final_turn))
        return state_vector