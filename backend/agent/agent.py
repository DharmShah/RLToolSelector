from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from agent.policy import ToolPolicy
from tools.executor import execute_tool
from tools.llm import generate_response


class Agent:

    def __init__(self):

        self.base_dir = Path(__file__).resolve().parent.parent

        self.model_path = (
            self.base_dir
            / "models"
            / "agent_policy_v2.pt"
        )

        # -------------------------------------------------
        # Embedding model
        # -------------------------------------------------

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # -------------------------------------------------
        # Load trained RL policy
        # -------------------------------------------------

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
    # RL TOOL SELECTION
    # =====================================================

    def select_tool(self, query: str) -> dict:

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
            "confidence": confidence,
        }

    # =====================================================
    # AGENT EXECUTION
    # =====================================================

    def run(self, query: str) -> dict:

        # -------------------------------------------------
        # 1. RL selects tool
        # -------------------------------------------------

        decision = self.select_tool(query)

        tool = decision["tool"]
        confidence = decision["confidence"]

        # -------------------------------------------------
        # 2. Ask user
        # -------------------------------------------------

        if tool == "ask_user":

            return {
                "query": query,
                "tool": tool,
                "confidence": confidence,
                "result": None,
                "response": (
                    "I need a little more information "
                    "to answer that. Could you clarify?"
                ),
            }

        # -------------------------------------------------
        # 3. Calculator
        # -------------------------------------------------

        if tool == "calculator":

            execution = execute_tool(
                tool,
                query,
            )

            result = execution["result"]

            # Deterministic response.
            # Don't send a simple calculation through the LLM.
            response = (
                f"The answer is {result}."
            )

            return {
                "query": query,
                "tool": tool,
                "confidence": confidence,
                "result": result,
                "response": response,
            }

        # -------------------------------------------------
        # 4. Search
        # -------------------------------------------------

        if tool == "search":

            execution = execute_tool(
                tool,
                query,
            )

            result = execution["result"]

            response = generate_response(
                user_query=query,
                tool=tool,
                tool_result=str(result),
            )

            return {
                "query": query,
                "tool": tool,
                "confidence": confidence,
                "result": result,
                "response": response,
            }

        # -------------------------------------------------
        # 5. LLM
        # -------------------------------------------------

        if tool == "llm":

            response = generate_response(
                user_query=query,
                tool="llm",
                tool_result=(
                    "No external tool was required. "
                    "Answer using your general knowledge."
                ),
            )

            return {
                "query": query,
                "tool": tool,
                "confidence": confidence,
                "result": None,
                "response": response,
            }

        # -------------------------------------------------
        # 6. Finish
        # -------------------------------------------------

        return {
            "query": query,
            "tool": tool,
            "confidence": confidence,
            "result": None,
            "response": (
                "I couldn't determine the appropriate "
                "action for this request."
            ),
        }