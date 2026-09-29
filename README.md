<div align="center">

# 🧭 RLToolSelector · NEXUS

### A tiny, locally-run reinforcement-learning policy that decides **which tool an AI agent should use**.

`search` · `calculator` · `llm` · `ask_user`

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-policy-EE4C2C?logo=pytorch&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)](#)
[![React](https://img.shields.io/badge/React_19-Vite-61DAFB?logo=react&logoColor=black)](#)

[Why](#-why-this-exists) · [How it works](#-how-it-works) · [Quick start](#-quick-start) · [API](#-api) · [Training](#-training) · [vs Jev](#-how-it-compares-to-jev-by-typesafe-ai) · [Limitations](#-honest-limitations) · [Roadmap](#-roadmap)

<!-- TODO: add a screenshot/GIF of the chat UI: docs/demo.gif -->

</div>

---

## 💡 Why this exists

Most agents pick a tool by asking a full LLM *"which tool should I call?"* and parsing the answer. That costs an extra LLM round-trip on **every** step, adds latency, and can return something that isn't a tool at all.

Tool choice is really a **classification decision**. RLToolSelector treats it that way:

- **Closed action space.** The output is an `argmax` over a fixed tool list. It structurally cannot invent a tool.
- **Tiny and local.** A sentence embedding plus a ~132k-parameter MLP, running on CPU. No API call is needed just to *decide*.
- **Confidence built in.** Every decision returns a softmax probability your code can threshold on.
- **Trained with RL.** Behavior cloning to warm-start, then REINFORCE fine-tuning with a reward signal.

The selected tool is then executed, and an LLM (via Groq) writes the final answer from the tool's result.

---

## 🧠 How it works

```
                        ┌───────────────────────────────┐
  User query ─────────▶ │ all-MiniLM-L6-v2  (384-dim)   │
                        └───────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │ RL policy MLP  384→256→128→4  │
                        └───────────────┬───────────────┘
                                        ▼
                     tool + confidence (softmax probability)
                                        │
        ┌───────────────┬───────────────┼───────────────┐
        ▼               ▼               ▼               ▼
     search         calculator         llm          ask_user
  (Tavily web)   (safe AST math)   (Groq LLM)   (request clarification)
        │               │               │               │
        └───────────────┴───────┬───────┴───────────────┘
                                ▼
                 Groq LLM writes the final response
                                ▼
              { response, tool, confidence, reward }
```

| Tool | When the policy picks it | What runs |
|---|---|---|
| `search` | Current / live / recent information (news, prices, "latest…") | Tavily web search, then Groq summarizes the results |
| `calculator` | Arithmetic and percentages | Regex extraction plus a **safe AST evaluator** (no `eval`) |
| `llm` | Explanations and general knowledge | Groq LLM answers directly |
| `ask_user` | Ambiguous or context-free queries ("Which product do you mean?") | Asks the user to clarify |

**Fail-safe design.** If the policy picks `calculator` for a non-math query, the executor fails gracefully and falls back to the LLM instead of crashing, and that turn gets a reward of `0.0`. Search and LLM failures also degrade to a friendly message.

**Runtime reward.** After each execution, the agent computes a reward: the policy's confidence if the tool succeeded, `0.0` if it failed or returned nothing. It's surfaced in the UI so you can watch the agent's behavior.

---

## 🧪 The model

| | |
|---|---|
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` → 384-d embedding |
| Policy | MLP `384 → 256 → 128 → 4` (ReLU, dropout 0.1), about **132k parameters** (~530 KB checkpoint) |
| Action space | `search`, `calculator`, `llm`, `ask_user` |
| Training | Behavior cloning (12 epochs, AdamW) → REINFORCE (3,000 episodes, entropy bonus 0.01, gradient clipping) |
| Reward (training) | `+1.0` correct tool, `-1.0` wrong tool |
| Checkpoint | `backend/models/agent_policy_v2.pt` (weights + tool list + embedding config) |

---

## 📁 Project structure

```
RLToolSelector/
├── backend/
│   ├── app.py                 # FastAPI app (/, /health, /api/chat)
│   ├── agent/
│   │   ├── agent.py           # Selection → execution → reward pipeline
│   │   └── policy.py          # ToolPolicy network
│   ├── rl/
│   │   ├── train2.py          # ✅ Current trainer: behavior cloning + REINFORCE
│   │   ├── agent_environment.py  # Multi-step environment (scaffold, not yet wired in)
│   │   ├── environment.py, train.py, predict.py  # v1 (REINFORCE-only) prototype
│   │   └── actions.py, policy.py
│   ├── tools/
│   │   ├── executor.py        # Routes selected tool → implementation
│   │   ├── search.py          # Tavily search
│   │   ├── calculator.py      # Safe arithmetic evaluator
│   │   └── llm.py             # Groq client + prompts
│   ├── models/                # Trained checkpoints (.pt)
│   ├── dataset.json           # Training data (query → tool)
│   └── requirements.txt
└── frontend/                  # React 19 + Vite + lucide-react
    └── src/
        ├── LandingPage.jsx
        └── Chatbot.jsx        # Chat UI showing chosen tool + reward live
```

---

## 🚀 Quick start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq](https://console.groq.com) API key (LLM responses)
- A [Tavily](https://tavily.com) API key (web search)

### 1 · Backend

```bash
git clone https://github.com/DharmShah/RLToolSelector.git
cd RLToolSelector/backend

python -m venv venv
# Windows:      .\venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

Create `backend/.env`:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
# optional (default: openai/gpt-oss-20b)
GROQ_MODEL=openai/gpt-oss-20b
```

Run:

```bash
uvicorn app:app --reload
```

The API is now live at **http://localhost:8000** (interactive docs at `/docs`).

### 2 · Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** and click **Launch NEXUS**.

> CORS is configured for `http://localhost:5173`. Edit `allow_origins` in `backend/app.py` if you deploy elsewhere.

---

## 🔌 API

### `POST /api/chat`

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 18 percent of 7250?"}'
```

```json
{
  "response": "The answer is 1305.0.",
  "model": "openai/gpt-oss-20b",
  "tool": "calculator",
  "confidence": 0.9712,
  "reward": 0.9712
}
```
*(Illustrative response; the exact confidence depends on your trained checkpoint.)*

| Field | Type | Description |
|---|---|---|
| `response` | string | Final answer for the user |
| `model` | string | LLM used to write the response |
| `tool` | string | Tool the RL policy selected |
| `confidence` | float | Policy's softmax probability for that tool |
| `reward` | float | Runtime reward (confidence if the tool succeeded, `0.0` otherwise) |

Other endpoints: `GET /` (status) and `GET /health`.

### Use just the selector (no server)

```python
# from backend/
from agent.agent import Agent

agent = Agent()
print(agent.select_tool("What is the current price of gold?"))
# -> {'tool': 'search', 'confidence': 0.xx}
```

Because selection is separate from execution, you can use `select_tool` as a cheap routing layer and fall back to a bigger model when confidence is low:

```python
d = agent.select_tool(query)
if d["confidence"] < 0.6:
    escalate_to_frontier_llm(query)
```

---

## 🏋️ Training

Retrain the policy on your own labeled queries:

```bash
cd backend
python rl/train2.py
```

Dataset format (`backend/dataset.json`):

```json
[
  { "id": 2, "query": "What changed recently in renewable energy?", "tool": "search", "difficulty": "easy" }
]
```

`train2.py`:

1. Loads and validates the dataset (rows with a missing or unknown `tool` are skipped with a warning).
2. Embeds all queries with MiniLM.
3. **Phase A:** Behavior cloning (supervised warm start).
4. **Phase B:** REINFORCE fine-tuning with an entropy bonus, keeping the best checkpoint.
5. Saves `models/agent_policy_v2.pt`.

The current dataset holds **700 usable labeled examples**, 175 per tool and balanced across `easy`, `medium` and `hard`.

To add a new tool, add it to `TOOLS` in `train2.py`, label examples for it, retrain, and add a handler in `tools/executor.py` and `agent/agent.py`.

---

## 🆚 How it compares to Jev by TypeSafe AI

[Jev](https://typesafe.ai/) is TypeSafe's general-purpose "System One" decision model. It's hosted, and you send it a state plus typed questions (yes/no, choice, score) and get calibrated answers with confidence. Tool selection is one of many things it can do.

RLToolSelector is a narrower, self-hostable take on the same idea: **replace an LLM call with a fast typed decision.**

| | Jev (TypeSafe AI) | RLToolSelector |
|---|---|---|
| Scope | General decisions: Noul / Choice / Score questions | **Purpose-built for agent tool routing** |
| Hosting | Hosted API and gateways | **Self-hosted, runs on CPU, fully local** |
| Selection cost | Per-request API pricing | **No API call for the decision itself** |
| Output can't be a non-existent option | ✅ | ✅ (closed action space) |
| Trained on *your* tools | Prompted with criteria | **Retrain on your own labeled queries in minutes** |
| Data privacy for routing | Query sent to a third party | **Query never leaves your machine for the decision** |
| Probability calibration | Calibrated (RL for calibrated decisions) | Raw softmax, not yet calibrated |
| Breadth | Very broad | Narrow (4 tools) |

**Positioning:** if you need one general decision model, Jev is the broader product. If you need a transparent, tiny, retrainable router you fully own, that's RLToolSelector. Head-to-head accuracy and latency numbers haven't been measured yet (see [Roadmap](#-roadmap)), so this README makes no speed or accuracy claims against Jev.

---

## ⚠️ Honest limitations

- **No held-out evaluation yet.** `train2.py` reports accuracy on the same data it trains on, so those numbers measure fit, not generalization. A proper train/test split is the top roadmap item.
- **Small dataset.** 700 labeled examples (a further 500 rows in `dataset.json` have no `tool` label and are skipped).
- **Confidence is uncalibrated.** It's the raw softmax probability, and it can be over-confident on out-of-distribution queries.
- **Single-step selection.** `rl/agent_environment.py` sketches multi-step episodes with step penalties, but the deployed agent picks one tool per query.
- **Legacy v1 scripts** (`rl/train.py`, `rl/predict.py`) use different paths and a REINFORCE-only recipe. Use `train2.py`.
- The `history` field is accepted by `/api/chat` but isn't yet used for selection.

---

## 🗺️ Roadmap

- [ ] Train/validation/test split, with per-tool precision/recall and a confusion matrix
- [ ] Calibration (temperature scaling) and reliability diagrams
- [ ] Benchmark vs. an LLM-as-router baseline and vs. Jev: accuracy, p50/p95 latency, cost per 1k decisions
- [ ] Use conversation history in the state
- [ ] Multi-step episodes using `agent_environment.py` (cost/latency-aware rewards)
- [ ] Online learning from real tool outcomes
- [ ] Docker Compose one-command deploy
- [ ] Python / TypeScript client packages

---

## 🤝 Contributing

Issues and PRs are welcome. Fork, branch, commit, open a PR. Adding labeled training examples, especially hard or ambiguous ones, is one of the most useful contributions.

## 📄 License

No license file yet. Add one (e.g. MIT) to the repo root so others can use the project.

## 👤 Author

**Dharm Shah** · [@DharmShah](https://github.com/DharmShah)

<div align="center">⭐ Star the repo if you find it useful ⭐</div>
