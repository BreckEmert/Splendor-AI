# Splendor/Splendor_components/Player_components/human_agent.py

import numpy as np
import queue


class HumanAgent:
    def __init__(self):
        self._move_queue = queue.Queue(maxsize=1)
        self.pending_spend = None

    def feed_move(self, move_index: int):
        self._move_queue.put(move_index, block=False)

    def feed_spend(self, spent_gems: np.ndarray):
        self.pending_spend = spent_gems.copy()

    def await_move(self, legal_mask) -> int:
        """Blocks until GUI pushes a legal index."""
        # Expose the legal_mask to the GUI thread
        self.legal_mask = legal_mask.copy()

        # Wait for a move
        while True:
            move = self._move_queue.get()
            if legal_mask[move]:
                return move
            else:
                raise ValueError("await_move received an illegal move")
