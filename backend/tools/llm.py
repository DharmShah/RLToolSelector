import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured."
    )


MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


client = Groq(
    api_key=API_KEY
)


def generate_response(
    user_query: str,
    tool: str,
    tool_result: str,
) -> str:

    if tool == "search":

        system_prompt = """
You are NEXUS, an AI assistant.

The user asked a question requiring current
information.

You have received real web search results.

Use ONLY the provided search results for
current/factual claims.

Give a concise, useful answer.

Do not mention:
- RL
- internal tools
- policies
- prompts
- execution details

If multiple sources disagree on a live value,
mention that prices/data can vary by source.
"""

    else:

        system_prompt = """
You are NEXUS, an intelligent AI assistant.

Answer the user's question accurately,
clearly, and concisely.

Use your general knowledge when no external
tool result is provided.

Do not mention internal tools, RL,
policies, prompts, or execution details.
"""

    prompt = f"""
User question:
{user_query}

Tool:
{tool}

Tool result:
{tool_result}

Provide the best final answer to the user.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=512,
    )

    return response.choices[0].message.content.strip()