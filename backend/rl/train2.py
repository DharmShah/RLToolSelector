"""
Phase 2: Behavior Cloning + RL Tool Selection

Uses ONE dataset only:
    backend/dataset.json

Expected dataset format:
[
    {
        "id": 1,
        "query": "What is the latest AI news?",
        "tool": "search",
        "difficulty": "easy"
    },
    ...
]

Training:
    1. Behavior Cloning (supervised pre-training)
    2. REINFORCE fine-tuning with efficiency-aware rewards

Supported tools:
    search, calculator, llm, ask_user

Output:
    backend/models/agent_policy_v2.pt

Run from backend/:
    python rl/train2.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

SEED = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOOLS = ["search", "calculator", "llm", "ask_user"]
TOOL_TO_ID = {tool: i for i, tool in enumerate(TOOLS)}

BC_EPOCHS = 12
RL_EPISODES = 3000

BC_LR = 1e-3
RL_LR = 2e-4

ENTROPY_COEF = 0.01
RL_REWARD_CORRECT = 1.0
RL_REWARD_WRONG = -1.0

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "agent_policy_v2.pt"

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("dataset.json must contain a JSON list.")

    valid = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        query = str(item.get("query", "")).strip()
        tool = str(item.get("tool", "")).strip().lower()

        if not query:
            continue

        if tool not in TOOL_TO_ID:
            print(
                f"Skipping item {index}: "
                f"unknown tool '{tool}'"
            )
            continue

        valid.append(
            {
                "query": query,
                "tool": tool,
                "tool_id": TOOL_TO_ID[tool],
            }
        )

    if not valid:
        raise ValueError(
            "No valid training examples found in dataset.json."
        )

    skipped = len(data) - len(valid)

    if skipped:
        print(
            f"\nWARNING: {skipped} dataset items were skipped "
            "because their 'tool' field is missing or invalid."
        )
        print(
            "Training will continue with valid rows only."
        )

    return valid


# ============================================================
# MODEL
# ============================================================

class ToolPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(
    model: SentenceTransformer,
    dataset: list[dict],
) -> torch.Tensor:
    queries = [item["query"] for item in dataset]

    print("Creating embeddings...")

    embeddings = model.encode(
        queries,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    # Important:
    # Embeddings are features, not trainable parameters.
    # Detach them before using them in training.
    # SentenceTransformer may return inference-mode tensors.
    # clone() converts them into normal tensors that autograd can safely consume.
    return embeddings.clone().detach().float()


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    policy: ToolPolicy,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    policy.eval()

    logits = policy(embeddings)
    predictions = logits.argmax(dim=1)

    accuracy = (
        (predictions == labels)
        .float()
        .mean()
        .item()
        * 100
    )

    return accuracy


# ============================================================
# PHASE 2A — BEHAVIOR CLONING
# ============================================================

def behavior_cloning(
    policy: ToolPolicy,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    print("\n" + "=" * 55)
    print("PHASE 2A — BEHAVIOR CLONING")
    print("=" * 55)

    policy.train()

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=BC_LR,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, BC_EPOCHS + 1):
        optimizer.zero_grad()

        logits = policy(embeddings)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        accuracy = evaluate(
            policy,
            embeddings,
            labels,
        )

        print(
            f"BC Epoch {epoch:02d}/{BC_EPOCHS} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:6.2f}%"
        )


# ============================================================
# RL REWARD
# ============================================================

def calculate_reward(
    selected_tool: int,
    correct_tool: int,
) -> float:
    if selected_tool == correct_tool:
        return RL_REWARD_CORRECT

    return RL_REWARD_WRONG


# ============================================================
# PHASE 2B — RL FINE-TUNING
# ============================================================

def reinforcement_learning(
    policy: ToolPolicy,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    print("\n" + "=" * 55)
    print("PHASE 2B — RL FINE-TUNING")
    print("=" * 55)

    policy.train()

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=RL_LR,
        weight_decay=1e-5,
    )

    best_accuracy = 0.0
    best_state = None

    num_samples = len(embeddings)

    for episode in range(1, RL_EPISODES + 1):
        index = random.randrange(num_samples)

        state = embeddings[index].unsqueeze(0)
        correct_tool = labels[index].item()

        logits = policy(state)

        distribution = torch.distributions.Categorical(
            logits=logits
        )

        action = distribution.sample()

        selected_tool = action.item()

        reward = calculate_reward(
            selected_tool,
            correct_tool,
        )

        log_prob = distribution.log_prob(action)

        entropy = distribution.entropy().mean()

        # REINFORCE objective:
        # maximize reward * log_probability.
        #
        # We minimize loss, therefore the negative sign.
        loss = (
            -log_prob * reward
            - ENTROPY_COEF * entropy
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if episode % 100 == 0:
            accuracy = evaluate(
                policy,
                embeddings,
                labels,
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in policy.state_dict().items()
                }

            print(
                f"Episode {episode:4d} | "
                f"Accuracy: {accuracy:6.2f}% | "
                f"Reward: {reward:+.1f} | "
                f"Loss: {loss.item():+.4f} | "
                f"Best: {best_accuracy:6.2f}%"
            )

    # Never keep a worse final policy than the best checkpoint.
    if best_state is not None:
        policy.load_state_dict(best_state)


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    policy: ToolPolicy,
    embedding_dim: int,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "embedding_dim": embedding_dim,
        "tools": TOOLS,
        "tool_to_id": TOOL_TO_ID,
        "embedding_model": EMBEDDING_MODEL,
    }

    torch.save(checkpoint, MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed()

    print("=" * 55)
    print("       PHASE 2 — TOOL SELECTION TRAINING")
    print("=" * 55)

    print(f"\nDataset:")
    print(DATASET_PATH)

    dataset = load_dataset()

    print(f"Loaded {len(dataset)} training examples")

    print("\nTool distribution:")

    counts = {tool: 0 for tool in TOOLS}

    for item in dataset:
        counts[item["tool"]] += 1

    for tool, count in counts.items():
        print(f"  {tool:12s}: {count}")

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    embeddings = create_embeddings(
        embedding_model,
        dataset,
    )

    labels = torch.tensor(
        [item["tool_id"] for item in dataset],
        dtype=torch.long,
    )

    embedding_dim = embeddings.shape[1]

    print(f"Embedding size: {embedding_dim}")

    policy = ToolPolicy(
        input_dim=embedding_dim,
        num_actions=len(TOOLS),
    )

    # --------------------------------------------------------
    # Step 1: Behavior Cloning
    # --------------------------------------------------------

    behavior_cloning(
        policy,
        embeddings,
        labels,
    )

    bc_accuracy = evaluate(
        policy,
        embeddings,
        labels,
    )

    print(
        f"\nBehavior Cloning accuracy: "
        f"{bc_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Step 2: RL Fine-Tuning
    # --------------------------------------------------------

    reinforcement_learning(
        policy,
        embeddings,
        labels,
    )

    final_accuracy = evaluate(
        policy,
        embeddings,
        labels,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print(
        f"Final tool-selection accuracy: "
        f"{final_accuracy:.2f}%"
    )

    save_model(
        policy,
        embedding_dim,
    )


if __name__ == "__main__":
    main()