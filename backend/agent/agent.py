from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from agent.policy import ToolPolicy
from tools.executor import execute_tool
from tools.llm import generate_response


class Agent:

    def __init__(self):

        # =================================================
        # PATHS
        # =================================================

        self.base_dir = (
            Path(__file__).resolve().parent.parent
        )

        self.model_path = (
            self.base_dir
            / "models"
            / "agent_policy_v2.pt"
        )

        # =================================================
        # EMBEDDING MODEL
        # =================================================

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # =================================================
        # LOAD RL POLICY
        # =================================================

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False,
        )

        self.tools = checkpoint["tools"]

        self.policy = ToolPolicy(
            input_dim=checkpoint["embedding_dim"],
            num_actions=len(self.tools),
        )

        self.policy.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.policy.eval()

    # =====================================================
    # TOOL SELECTION
    # =====================================================

    def select_tool(
        self,
        query: str,
    ) -> dict:
        """
        Use the trained RL policy to select
        the most appropriate tool.
        """

        embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
        )

        embedding = (
            embedding
            .clone()
            .detach()
            .float()
            .unsqueeze(0)
        )

        with torch.no_grad():

            logits = self.policy(
                embedding
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            tool_id = probabilities.argmax(
                dim=-1
            ).item()

            confidence = probabilities[
                0,
                tool_id
            ].item()

        return {
            "tool": self.tools[tool_id],
            "confidence": round(
                confidence,
                4,
            ),
        }

    # =====================================================
    # REWARD CALCULATION
    # =====================================================

    def calculate_reward(
        self,
        tool: str,
        confidence: float,
        result,
    ) -> float:
        """
        Calculate runtime reward.

        Reward is based on whether the selected
        tool successfully completed execution.

        Successful execution:
            positive reward

        Failed execution:
            zero reward
        """

        # -------------------------------------------------
        # Tool execution failed
        # -------------------------------------------------

        if result is None:

            return 0.0

        # -------------------------------------------------
        # Empty result
        # -------------------------------------------------

        if isinstance(result, str):

            if not result.strip():

                return 0.0

        # -------------------------------------------------
        # Successful execution
        # -------------------------------------------------

        return round(
            confidence,
            4,
        )

    # =====================================================
    # ASK USER
    # =====================================================

    def handle_ask_user(
        self,
        query: str,
        confidence: float,
    ) -> dict:

        return {
            "query": query,
            "tool": "ask_user",
            "confidence": confidence,
            "reward": round(
                confidence,
                4,
            ),
            "result": None,
            "response": (
                "I need a little more information "
                "to answer that. Could you clarify?"
            ),
        }

    # =====================================================
    # CALCULATOR
    # =====================================================

    def handle_calculator(
        self,
        query: str,
        confidence: float,
    ) -> dict:
        """
        Execute calculator safely.

        If the RL policy incorrectly chooses
        calculator for a non-mathematical query,
        fall back to the LLM instead of crashing.
        """

        try:

            execution = execute_tool(
                "calculator",
                query,
            )

            result = execution["result"]

            reward = self.calculate_reward(
                tool="calculator",
                confidence=confidence,
                result=result,
            )

            return {
                "query": query,
                "tool": "calculator",
                "confidence": confidence,
                "reward": reward,
                "result": result,
                "response": (
                    f"The answer is {result}."
                ),
            }

        except Exception as error:

            # -------------------------------------------------
            # Wrong tool selected by RL
            # -------------------------------------------------

            print(
                f"[Calculator Error] {error}"
            )

            # -------------------------------------------------
            # Fallback to LLM
            # -------------------------------------------------

            response = generate_response(
                user_query=query,
                tool="llm",
                tool_result=(
                    "The calculator could not "
                    "process this request. "
                    "Answer the user's question "
                    "using general knowledge."
                ),
            )

            return {
                "query": query,

                # Actual fallback tool
                "tool": "llm",

                # Keep original policy confidence
                "confidence": confidence,

                # Wrong calculator selection
                "reward": 0.0,

                "result": None,

                "response": response,
            }

    # =====================================================
    # SEARCH
    # =====================================================

    def handle_search(
        self,
        query: str,
        confidence: float,
    ) -> dict:
        """
        Execute search and generate a natural
        language response.
        """

        try:

            execution = execute_tool(
                "search",
                query,
            )

            result = execution["result"]

            reward = self.calculate_reward(
                tool="search",
                confidence=confidence,
                result=result,
            )

            response = generate_response(
                user_query=query,
                tool="search",
                tool_result=str(result),
            )

            return {
                "query": query,
                "tool": "search",
                "confidence": confidence,
                "reward": reward,
                "result": result,
                "response": response,
            }

        except Exception as error:

            print(
                f"[Search Error] {error}"
            )

            return {
                "query": query,
                "tool": "search",
                "confidence": confidence,
                "reward": 0.0,
                "result": None,
                "response": (
                    "I couldn't retrieve the "
                    "information right now."
                ),
            }

    # =====================================================
    # LLM
    # =====================================================

    def handle_llm(
        self,
        query: str,
        confidence: float,
    ) -> dict:
        """
        Generate a response using the LLM.
        """

        try:

            response = generate_response(
                user_query=query,
                tool="llm",
                tool_result=(
                    "No external tool was required. "
                    "Answer using general knowledge."
                ),
            )

            reward = self.calculate_reward(
                tool="llm",
                confidence=confidence,
                result=response,
            )

            return {
                "query": query,
                "tool": "llm",
                "confidence": confidence,
                "reward": reward,
                "result": None,
                "response": response,
            }

        except Exception as error:

            print(
                f"[LLM Error] {error}"
            )

            return {
                "query": query,
                "tool": "llm",
                "confidence": confidence,
                "reward": 0.0,
                "result": None,
                "response": (
                    "I couldn't generate a response "
                    "right now."
                ),
            }

    # =====================================================
    # MAIN AGENT EXECUTION
    # =====================================================

    def run(
        self,
        query: str,
    ) -> dict:
        """
        Complete NEXUS execution pipeline.

        User Query
              ↓
        RL Tool Selection
              ↓
        ┌───────────────┐
        │ Selected Tool│
        └───────┬───────┘
                ↓
        Tool Execution
                ↓
        Response Generation
                ↓
        Reward
                ↓
        Final Response
        """

        # -------------------------------------------------
        # Validate query
        # -------------------------------------------------

        query = query.strip()

        if not query:

            return {
                "query": query,
                "tool": "ask_user",
                "confidence": 1.0,
                "reward": 1.0,
                "result": None,
                "response": (
                    "Please provide a question "
                    "or task."
                ),
            }

        # -------------------------------------------------
        # 1. RL TOOL SELECTION
        # -------------------------------------------------

        decision = self.select_tool(
            query
        )

        tool = decision["tool"]
        confidence = decision["confidence"]

        print(
            f"[NEXUS] Tool: {tool} | "
            f"Confidence: {confidence}"
        )

        # -------------------------------------------------
        # 2. ASK USER
        # -------------------------------------------------

        if tool == "ask_user":

            return self.handle_ask_user(
                query=query,
                confidence=confidence,
            )

        # -------------------------------------------------
        # 3. CALCULATOR
        # -------------------------------------------------

        if tool == "calculator":

            return self.handle_calculator(
                query=query,
                confidence=confidence,
            )

        # -------------------------------------------------
        # 4. SEARCH
        # -------------------------------------------------

        if tool == "search":

            return self.handle_search(
                query=query,
                confidence=confidence,
            )

        # -------------------------------------------------
        # 5. LLM
        # -------------------------------------------------

        if tool == "llm":

            return self.handle_llm(
                query=query,
                confidence=confidence,
            )

        # -------------------------------------------------
        # 6. FINISH / UNKNOWN TOOL
        # -------------------------------------------------

        print(
            f"[NEXUS] Unknown tool selected: {tool}"
        )

        return {
            "query": query,
            "tool": tool,
            "confidence": confidence,
            "reward": 0.0,
            "result": None,
            "response": (
                "I couldn't determine the "
                "appropriate action for this request."
            ),
        }