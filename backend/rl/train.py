import json
import os

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from environment import ToolSelectionEnvironment
from policy import ToolSelectionPolicy


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_PATH = "rl/dataset.json"
MODEL_DIR = "models"

TOOLS = [
    "search",
    "calculator",
    "llm",
    "ask_user",
]

TOOL_TO_ID = {
    tool: index
    for index, tool in enumerate(TOOLS)
}


EPISODES = 3000

LEARNING_RATE = 0.001

GAMMA = 0.99


# =========================================================
# LOAD DATASET
# =========================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as file:

    dataset = json.load(file)


print(
    f"Loaded {len(dataset)} training examples"
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# =========================================================
# DETERMINE INPUT SIZE
# =========================================================

test_embedding = encoder.encode(
    "test query",
    convert_to_tensor=True
)

input_size = test_embedding.shape[0]

print(
    f"Embedding size: {input_size}"
)


# =========================================================
# POLICY
# =========================================================

policy = ToolSelectionPolicy(
    input_size=input_size,
    num_actions=len(TOOLS)
)


optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# ENVIRONMENT
# =========================================================

environment = ToolSelectionEnvironment(
    dataset
)


# =========================================================
# TRAINING
# =========================================================

print("\nStarting RL training...\n")


reward_history = []


for episode in range(EPISODES):

    # -----------------------------------------------------
    # Get state
    # -----------------------------------------------------

    query = environment.reset()


    # -----------------------------------------------------
    # Convert query → embedding
    # -----------------------------------------------------

    state = encoder.encode(
        query,
        convert_to_tensor=True
    )


    state = state.clone().detach().float()


    # -----------------------------------------------------
    # Policy prediction
    # -----------------------------------------------------

    logits = policy(
        state
    )


    probabilities = F.softmax(
        logits,
        dim=-1
    )


    # -----------------------------------------------------
    # Sample an action
    # -----------------------------------------------------

    distribution = torch.distributions.Categorical(
        probabilities
    )

    action = distribution.sample()


    selected_tool = TOOLS[
        action.item()
    ]


    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    reward, done = environment.step(
        selected_tool
    )


    # -----------------------------------------------------
    # REINFORCE loss
    # -----------------------------------------------------

    log_probability = distribution.log_prob(
        action
    )


    loss = -log_probability * reward


    # -----------------------------------------------------
    # Update policy
    # -----------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    # -----------------------------------------------------
    # Store reward
    # -----------------------------------------------------

    reward_history.append(
        reward
    )


    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    if (episode + 1) % 100 == 0:

        recent_rewards = reward_history[-100:]

        accuracy = (
            sum(
                reward == 1
                for reward in recent_rewards
            )
            / len(recent_rewards)
        )

        print(
            f"Episode {episode + 1:4d} | "
            f"Accuracy: {accuracy:.2%} | "
            f"Last tool: {selected_tool}"
        )


# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


model_path = os.path.join(
    MODEL_DIR,
    "tool_policy.pt"
)


torch.save(
    policy.state_dict(),
    model_path
)


print("\nTraining completed.")

print(
    f"Model saved to: {model_path}"
)