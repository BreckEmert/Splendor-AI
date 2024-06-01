# Splendor/Rl folder - strategy.py
import numpy as np


class BestStrategy():
    """Return the raw moves, as that's what the model chose
    """
    def strategize(moves):
        return moves

class RandomStrategy():
    """Adjusts the logits randomly
    """
    def strategize(moves):
        noise = np.random.rand(*moves.shape) - 0.5
        return moves + noise

class OffensiveStrategy(Strategy):
    def __init__(self):
        pass

    def choosemove():
        # 1: Sum tier n gems * round weight for tier n
        # 2: Analyze reserving
        # 3: Analyze buying
            # Never buy if the card can't be bought or reserved, and we have good choices of gems left.
        pass

class ResourceHog(Strategy):
    def __init__(self):
        pass

class ObliviousStrategy(Strategy):
    def __init__(self):
        pass