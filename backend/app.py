from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.agent import Agent


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

MODEL = "openai/gpt-oss-20b"


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="NEXUS RL Agent API",
    description="Backend API for the NEXUS AI Agent",
    version="2.1.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# AGENT
# =========================================================

agent = Agent()


# =========================================================
# REQUEST SCHEMAS
# =========================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


# =========================================================
# RESPONSE SCHEMA
# =========================================================

class ChatResponse(BaseModel):
    response: str
    model: str
    tool: str | None = None
    confidence: float | None = None
    reward: float | None = None


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "NEXUS RL Agent API",
        "model": MODEL,
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    try:

        # -------------------------------------------------
        # Run RL Agent
        # -------------------------------------------------

        result = agent.run(
            request.message
        )

        # -------------------------------------------------
        # Extract response
        # -------------------------------------------------

        response = result.get(
            "response",
            result.get("result", ""),
        )

        # -------------------------------------------------
        # Return agent information
        # -------------------------------------------------

        return ChatResponse(
            response=response,
            model=MODEL,
            tool=result.get("tool"),
            confidence=result.get("confidence"),
            reward=result.get("reward"),
        )

    except Exception as error:

        print(
            "Agent Error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Agent execution failed",
        )