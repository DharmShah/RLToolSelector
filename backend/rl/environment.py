import random


TOOLS = [
    "search",
    "calculator",
    "llm",
    "ask_user",
]


class ToolSelectionEnvironment:

    def __init__(self, dataset):
        self.dataset = dataset
        self.current_sample = None

    def reset(self):
        """
        Start a new episode.
        """

        self.current_sample = random.choice(self.dataset)

        return self.current_sample["query"]

    def step(self, selected_tool):
        """
        Execute the selected action and return reward.
        """

        correct_tool = self.current_sample["tool"]

        if selected_tool == correct_tool:
            reward = 1.0
        else:
            reward = -1.0

        done = True

        return reward, done