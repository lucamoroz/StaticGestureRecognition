import collections
import os


class ActionTaker:

    def __init__(self, commands={}, history_len=1, min_confidence=0.98):
        self.commands = commands
        self.min_confidence = min_confidence
        self.history_len = history_len
        self.history = {}
        for key in commands.keys():
            self.history[key] = collections.deque(history_len*[0.], history_len)

    def on_new_state(self, state: str, confidence: float):
        if state not in self.commands.keys():
            raise RuntimeError("State ", state, " not in commands.")

        self.history[state].appendleft(confidence)
        if all(conf > self.min_confidence for conf in self.history[state]):
            self.history[state] = collections.deque(self.history_len*[0.], self.history_len)
            print("Executing ", self.commands[state])
            self.execute_action(state)

    def execute_action(self, state: str):
        if state not in self.commands.keys():
            raise RuntimeError("State ", state, " not in commands.")
        os.system(self.commands[state])