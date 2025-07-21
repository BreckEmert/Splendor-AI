# Splendor/Play/human_agent.py
"""
A blocking Agent that waits for the GUI to supply a legal move.
"""

import queue


class HumanAgent:
    """Same contract as RLAgent."""

    def __init__(self):
        # GUI → agent
        self._move_queue = queue.Queue(maxsize=1)

    # Public GUI API
    def feed_move(self, move_index: int):
        """Called after click detection."""
        try:
            self._move_queue.put(move_index, block=False)
        except queue.Full:
            pass  # should never happen

    # Game engine callback
    def await_move(self, legal_mask) -> int:  # noqa: N802
        """Blocks until GUI pushes a legal index."""
        while True:
            # blocks until click :contentReference[oaicite:3]{index=3}:
            move = self._move_queue.get()
            if legal_mask[move]:
                return move
