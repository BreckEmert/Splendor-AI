# Splendor/Splendor_components/Player_components/human_agent.py

import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Play.common_types import GUIMove


class HumanAgent:
    def __init__(self):
        self._move_queue = queue.Queue(maxsize=1)
        self.pending_take = None
        self.pending_spend = None
        self.awaiting_move = False

    def feed_move(self, move: "GUIMove"):
        try:
            self._move_queue.put(move, block=False)
        except:
            print("Debug: _move_queue is full.")

    def await_move(self) -> "GUIMove":
        """Blocks until GUI pushes a GUIMove object.
        Read by GUIGame.turn and sent to GUIGame.apply_human_move.
        """

        # Wait for a move
        self.awaiting_move = True
        while True:
            move = self._move_queue.get()
            if move:
                self.awaiting_move = False
                return move
