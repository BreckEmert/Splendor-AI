import random
import pandas as pd


class Player:
    def __init__(self, name, strategy):
        self.name = name
        self.gems = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0, 'gold': 0}
        self.gem_cards = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0}
        self.reserved_cards = []
        self.points = 0
        self.strategy = strategy

    def take_gems(self, gems_to_take):
        for color, amount in gems_to_take.items():
            self.gems[color] += amount

    def buy_card(self, card_id, tier, gameboard):
        card = gameboard.development_cards[tier][card_id]
        for gem in card.cost:
            required_gems = max(0, card.cost[gem] - self.gem_cards[gem])
            self.gems[gem] -= required_gems
        self.gem_cards[card.color] += 1
        self.points += card.points
        gameboard.remove_card(tier, card_id)

    def reserve_card(self, card_id, tier, gameboard):
        card = gameboard.development_cards[tier][card_id]
        self.reserved_cards.append(card)
        gameboard.remove_card(tier, card_id)

    def total_gems(self):
        return sum(self.gems.values())

    def possible_moves(self, game_state):
        max_take = 10 - self.total_gems()

        # Locate purchasable cards
        purchasable_cards = []
        for tier in gameboard.development_cards:
            for index, card in enumerate(gameboard.development_cards[tier]):
                total_cost = {gem: max(0, card.cost[gem] - self.gem_cards[gem]) for gem in card.cost}
                if sum(total_cost.values()) <= sum(self.gems.values()):
                    purchasable_cards.append((tier, index))

        return purchasable_cards

    def choose_move(self):
        best_payout = 0

        # Analyze taking gems

        # Analyze reserving

        # Analyze buying


class Board:
    def __init__(self):
        self.available_gems = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0, 'gold': 5}
        self.development_cards = {'tier1': [], 'tier2': [], 'tier3': []}
        self.noble_tiles = []
        self.players = []
        self.deck = None  # This will be initialized with a Deck object

    def setup_game(self, players, deck):
        self.players = players
        self.deck = deck

        # Initialize gems
        num_players = len(players)
        gem_counts = {2: 4, 3: 5, 4: 7}
        for gem in ['white', 'blue', 'green', 'red', 'black']:
            self.available_gems[gem] = gem_counts.get(num_players, 4)

        # Draw initial development cards
        for tier in ['tier1', 'tier2', 'tier3']:
            self.development_cards[tier] = [self.deck.draw_card(int(tier[-1])) for _ in range(4)]

        # Draw noble tiles
        self.noble_tiles = [self.deck.draw_noble() for _ in range(num_players + 1)]

    def display_board(self):
        pass

    def add_player(self, player):
        self.players.append(player)

    def take_gems(self, player, gems_to_take):
        for color, amount in gems_to_take.items():
            self.available_gems[color] -= amount
            player.gems[color] += amount  # Update player's gems

    def remove_card(self, tier, id):
        removed_card = self.development_cards[tier].pop(card_id)
        tier.gems[color] += removed_card[color] for color in removed_card.cost

    def check_noble_visit(self, player):
        visited_nobles = []
        for noble in self.noble_tiles:
            if self.can_noble_visit(player, noble):
                visited_nobles.append(noble)
                player.points += noble.points  # Assuming noble has points attribute

        return visited_nobles

    def can_noble_visit(self, player, noble):
        # Check if the player meets the noble's requirements
        for gem, required_amount in noble.requirements.items():
            if player.gem_cards[gem] < required_amount:
                return False
        return True


class Card:
    def __init__(self, card_id, gem_type, points, cost):
        self.card_id = card_id
        self.gem_type = gem_type
        self.points = points
        self.cost = cost  # Dictionary of cost in gems

    def __repr__(self):
        return f"Card(ID: {self.card_id}, Gem: {self.gem_type}, Points: {self.points}, Cost: {self.cost})"


class Noble:
    def __init__(self, requirements, points):
        self.requirements = requirements  # Dictionary of gem type to required amount
        self.points = points

    def __repr__(self):
        return f"Noble(Requirements: {self.requirements}, Points: {self.points})"

    
class Deck:
    def __init__(self, tier1_cards, tier2_cards, tier3_cards, noble_cards):
        self.tier1, self.tier2, self.tier3, self.nobles = self.load_deck(excel_file_path)
        self.shuffle(self)

    def create_cost_dict(row):
        return {gem.split('_')[1]: row[gem] for gem in cost_columns if row[gem] > 0}

    def create_card(row):
        cost = create_cost_dict(row)
        return Card(card_id=row['id'], gem_type=row['gem_type'], points=row['points'], cost=cost)

    def load_deck(self, deck_path):
        df = pd.read_excel(deck_path, sheet_name=None)

        tier1 = df[Tier1].to_dict('records')
        tier2 = df[Tier2].to_dict('records')
        tier2 = df[Tier3].to_dict('records')
        nobles = df[Nobles].to_dict('records')

        return tier1_cards, tier2_cards, tier3_cards, noble_cards

    def shuffle(self):
        random.shuffle(self.tier1)
        random.shuffle(self.tier2)
        random.shuffle(self.tier3)
        random.shuffle(self.nobles)

    def draw_card(self, tier):
        if tier == 1 and self.tier1:
            return self.tier1.pop()
        elif tier == 2 and self.tier2:
            return self.tier2.pop()
        elif tier == 3 and self.tier3:
            return self.tier3.pop()
        else:
            return None

    def draw_noble(self):
        if self.nobles:
            return self.nobles.pop()
        else:
            return None


def turn():


# Example of creating a board and adding players
board = Board()
Alice = SplendorPlayer("Alice", "Aggressive")
Dan = SplendorPlayer("Dan", "Defensive")
Elly = SplendorPlayer("Elly", "Efficient")

# Choose from Aggressive Alice, Defensive Dan, Efficient Elly
players = [Alice, Dan]
setup(players)