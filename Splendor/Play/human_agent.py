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
    def choose_move(self, board, state) -> int:  # noqa: N802
        """Blocks until GUI pushes a move index."""
        return self._move_queue.get()
