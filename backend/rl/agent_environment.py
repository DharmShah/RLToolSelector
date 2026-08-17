class AgentEnvironment:

    def __init__(self):

        self.steps = 0

        self.max_steps = 5

        self.history = []

        self.query = None


    def reset(self, query):

        self.query = query

        self.steps = 0

        self.history = []

        return self._build_state()


    def step(
        self,
        action,
        tool_result=None,
    ):

        self.steps += 1

        self.history.append({
            "action": action,
            "result": tool_result,
        })


        # -------------------------------------------------
        # Finish
        # -------------------------------------------------

        if action == "finish":

            reward = self.calculate_reward()

            return (
                self._build_state(),
                reward,
                True
            )


        # -------------------------------------------------
        # Maximum steps
        # -------------------------------------------------

        if self.steps >= self.max_steps:

            reward = -1.0

            return (
                self._build_state(),
                reward,
                True
            )


        # -------------------------------------------------
        # Continue
        # -------------------------------------------------

        reward = self.intermediate_reward(
            action
        )

        return (
            self._build_state(),
            reward,
            False
        )


    def intermediate_reward(
        self,
        action
    ):

        """
        Temporary reward.

        We'll replace this with a real
        task-success/cost/latency reward.
        """

        if action in [
            "search",
            "calculator",
        ]:

            return 0.1


        if action == "ask_user":

            return 0.0


        if action == "llm":

            return 0.05


        return 0.0


    def calculate_reward(self):

        tool_count = len(
            self.history
        )

        # Small penalty for unnecessary steps
        penalty = tool_count * 0.05

        return 1.0 - penalty


    def _build_state(self):

        return {
            "query": self.query,
            "history": self.history,
            "steps": self.steps,
        }