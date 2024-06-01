# Splendor/Environment/game.py

from Environment.Splendor_components import Board # type: ignore
from Environment.Splendor_components import Player # type: ignore

class Game:
    def __init__(self, players):
        self.num_players = len(players)

        self.board = Board(self.num_players)
        self.players: list = [Player(name, strategy, strategy_strength) 
                              for name, strategy, strategy_strength in players]

        self.active_player = 0
        self.turn_order: int = 0
        self.is_final_turn: bool = False
    
    def turn(self):
        self.active_player = self.players[self.turn_order]
        prev_state = self.get_state()

        chosen_move = self.active_player.choose_move(self.board, prev_state)
        self.apply_move(chosen_move)

        self.check_noble_visit()
        if self.active_player.points >= 15:
            self.is_final_turn = True

        self.turn_order = (self.turn_order + 1) % self.num_players

        print('Active player points:', self.active_player.points)

    def apply_move(self, move):
        print(move)
        action, details = move
        match action:
            case 'take':
                gem, amount = list(details.items())[0]
                self.board.take_gems(gem, amount)
                self.active_player.take_gems(details)
            case 'buy':
                bought_card = self.board.buy_card(card_id = details)
                self.active_player.buy_card(bought_card)
            case 'buy_reserved':
                bought_card = next(card for card in self.active_player.reserved_cards if card.id==details)
                self.active_player.reserved_cards.remove(bought_card)
                self.active_player.buy_card(bought_card)
            case 'reserve':
                reserved_card = self.board.reserve(card_id = details)
                self.active_player.reserve_card(reserved_card)
            case 'reserve_top':
                reserved_card = self.board.reserve_from_deck(tier = details)
                self.active_player.reserve_card(reserved_card)

    def check_noble_visit(self):
        for noble in self.board.cards['nobles']:
            if all(self.active_player.cards[gem] >= amount for gem, amount in noble.cost.items()):
                self.active_player.points += noble.points
                self.Board.deck.remove(noble)
                break # Implement logic to choose the noble if tied

    def get_victor(self):
        victor = max(self.players, key=lambda p: p.points)
        return victor
   
    def get_state(self):
        return {
            'board': self.board.get_state(),
            'players': {player.name: player.get_state() for player in self.players}, 
            'current_turn': self.turn, 
            'is_final_turn': self.is_final_turn
        }

    def to_vector(self):
        state_vector = self.board.to_vector()
        for player in self.players:
            state_vector.extend(player.to_vector())
        state_vector.append(self.turn_order)
        state_vector.append(int(self.is_final_turn))
        return state_vector

if __name__ == "__main__":
    import sys

    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")

    from Environment.Splendor_components import Board # type: ignore
    from Environment.Splendor_components import Player # type: ignore
    from Environment.Splendor_components.Player_components.strategy import BestStrategy # type: ignore

    players = [('Player1', BestStrategy(), 1), ('Player2', BestStrategy(), 1)]
    game = Game(players)
    done = False
    while not done:
        game.turn()
        if game.is_final_turn:
            done = True