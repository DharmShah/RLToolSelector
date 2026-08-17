import re

from tools.calculator import calculate
from tools.search import search
from tools.llm import generate_response


def extract_calculation(query: str) -> str:

    text = query.lower().strip()

    # ---------------------------------------------
    # Percentage: "27% of 8450"
    # ---------------------------------------------

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*%?\s*percent\s*of\s*(\d+(?:\.\d+)?)",
        text,
    )

    if not match:

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
            text,
        )

    if match:

        percent = match.group(1)
        number = match.group(2)

        return f"({percent} / 100) * {number}"

    # ---------------------------------------------
    # Mathematical expression
    # ---------------------------------------------

    expression_match = re.search(
        r"[\d\s+\-*/().%]+",
        text,
    )

    if expression_match:
        return expression_match.group(0).strip()

    raise ValueError(
        f"Could not extract calculation from: {query}"
    )


def execute_tool(
    tool: str,
    query: str,
) -> dict:

    if tool == "calculator":

        expression = extract_calculation(query)

        result = calculate(expression)

        return {
            "tool": "calculator",
            "input": expression,
            "result": str(result),
        }

    if tool == "search":

        result = search(query)

        return {
            "tool": "search",
            "result": result,
        }

    if tool == "llm":

        result = generate_response(
            user_query=query,
            tool="llm",
            tool_result=(
                "No external tool was required. "
                "Answer using general knowledge."
            ),
        )

        return {
            "tool": "llm",
            "result": result,
        }

    if tool == "ask_user":

        return {
            "tool": "ask_user",
            "result": None,
        }

    raise ValueError(
        f"Unknown tool: {tool}"
    )