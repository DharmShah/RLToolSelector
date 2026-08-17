import os

import requests
from dotenv import load_dotenv

load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def search(query: str) -> str:
    """
    Perform a real web search using Tavily.
    """

    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured in .env"
        )

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": True,
        },
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    answer = data.get("answer")

    results = data.get("results", [])

    output = []

    if answer:
        output.append(
            f"Summary:\n{answer}"
        )

    for i, item in enumerate(results, start=1):

        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")

        output.append(
            f"{i}. {title}\n"
            f"{content}\n"
            f"Source: {url}"
        )

    return "\n\n".join(output)