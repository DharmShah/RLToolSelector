import os

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
# FASTAPI
# =========================================================

app = FastAPI(
    title="NEXUS RL Agent API",
    description="Backend API for the NEXUS AI Agent",
    version="2.0.0",
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
# REQUEST SCHEMA
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


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "NEXUS RL Agent API",
        "model": "openai/gpt-oss-20b",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# =========================================================
# CHAT
# =========================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    try:

        result = agent.run(
            request.message
        )

        return ChatResponse(
            response=result.get(
                "response",
                result.get("result", ""),
            ),
            model="openai/gpt-oss-20b",
            tool=result.get("tool"),
            confidence=result.get("confidence"),
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