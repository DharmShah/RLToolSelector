import torch
from sentence_transformers import SentenceTransformer

from policy import ToolSelectionPolicy


TOOLS = [
    "search",
    "calculator",
    "llm",
    "ask_user",
]


MODEL_PATH = "models/tool_policy.pt"


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# CREATE POLICY
# =========================================================

policy = ToolSelectionPolicy(
    input_size=384,
    num_actions=len(TOOLS)
)


# =========================================================
# LOAD TRAINED WEIGHTS
# =========================================================

policy.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)


policy.eval()


# =========================================================
# PREDICT TOOL
# =========================================================

def select_tool(query: str):

    embedding = encoder.encode(
        query,
        convert_to_tensor=True
    )

    embedding = embedding.clone().detach().float()


    with torch.no_grad():

        logits = policy(
            embedding
        )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )


    action = torch.argmax(
        probabilities
    ).item()


    selected_tool = TOOLS[action]


    confidence = probabilities[action].item()


    return {
        "tool": selected_tool,
        "confidence": confidence,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    questions = [
    "What are the latest developments in AI agents?",
    "What is 18 percent of 7250?",
    "Explain why transformers work well for language tasks.",
    "Which company are you referring to?",
    "What is today's USD to INR exchange rate?",
    "Calculate 1275 multiplied by 84.",
    "What is the latest NVIDIA GPU?",
    "Why do LLMs hallucinate?",
    "Can you tell me more about that?",
    "What is the current price of gold?",
    "Calculate 35 percent of 18400.",
    "Explain the difference between RAG and fine-tuning.",
    "Which product do you mean?",
    "What happened recently in the AI industry?",
    "Calculate 9876 divided by 24.",
    "How does reinforcement learning work?",
    "What are the newest developments in robotics?",
    "Which version are you talking about?",
    "What is the current Bitcoin price?",
    "Why are embeddings useful?"
]


    for question in questions:

        result = select_tool(
            question
        )

        print("\nQuery:")
        print(question)

        print(
            f"Tool: {result['tool']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.2%}"
        )