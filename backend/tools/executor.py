import re

from tools.calculator import calculate
from tools.search import search
from tools.llm import generate_response


def extract_calculation(query: str) -> str:
    """
    Convert a natural-language calculation query
    into a mathematical expression.

    Examples:
        "What is 27% of 8450?"
            -> "(27 / 100) * 8450"

        "What is 17.5% of 12,800?"
            -> "(17.5 / 100) * 12800"

        "25 * 48"
            -> "25 * 48"
    """

    text = query.lower().strip()

    # =====================================================
    # PERCENTAGE CALCULATIONS
    # =====================================================

    # Matches:
    # 27% of 8450
    # 17.5% of 12,800
    # 25 percent of 800
    # 12.5 percent of 1,000
    percentage_match = re.search(
        r"(\d+(?:\.\d+)?)"
        r"\s*(?:%|percent)"
        r"\s+of\s+"
        r"([\d,]+(?:\.\d+)?)",
        text,
    )

    if percentage_match:

        percent = percentage_match.group(1)

        number = (
            percentage_match
            .group(2)
            .replace(",", "")
        )

        return (
            f"({percent} / 100) * {number}"
        )

    # =====================================================
    # GENERAL MATHEMATICAL EXPRESSIONS
    # =====================================================

    expression_match = re.search(
        r"[\d\s+\-*/().%,]+",
        text,
    )

    if expression_match:

        expression = (
            expression_match
            .group(0)
            .replace(",", "")
            .strip()
        )

        if expression:
            return expression

    raise ValueError(
        f"Could not extract calculation from: {query}"
    )


def execute_tool(
    tool: str,
    query: str,
) -> dict:
    """
    Execute the tool selected by the RL policy.
    """

    # =====================================================
    # CALCULATOR
    # =====================================================

    if tool == "calculator":

        expression = extract_calculation(
            query
        )

        result = calculate(
            expression
        )

        return {
            "tool": "calculator",
            "input": expression,
            "result": str(result),
        }

    # =====================================================
    # SEARCH
    # =====================================================

    if tool == "search":

        result = search(
            query
        )

        return {
            "tool": "search",
            "result": result,
        }

    # =====================================================
    # LLM
    # =====================================================

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

    # =====================================================
    # ASK USER
    # =====================================================

    if tool == "ask_user":

        return {
            "tool": "ask_user",
            "result": None,
        }

    # =====================================================
    # UNKNOWN TOOL
    # =====================================================

    raise ValueError(
        f"Unknown tool: {tool}"
    )