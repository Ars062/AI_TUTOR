"""Milestone 10: tool/function-calling registry.

The tutor can call external tools (search, calculator, time, etc.) when the
LLM decides they're needed. Each tool is a Python function; the agent
dispatches LLM-generated tool calls to the right function.
"""
import math
import time
from datetime import datetime


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, callable] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, description: str, func: callable):
        self._tools[name] = func
        self._descriptions[name] = description

    def call(self, name: str, **kwargs) -> str:
        fn = self._tools.get(name)
        if not fn:
            return f"Unknown tool: {name}"
        try:
            return str(fn(**kwargs))
        except Exception as e:
            return f"Tool error: {e}"

    def schema(self) -> list[dict]:
        return [
            {"name": n, "description": d}
            for n, d in self._descriptions.items()
        ]


registry = ToolRegistry()


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "log": math.log,
        "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def search_knowledge_base(query: str) -> str:
    return f"[Knowledge base search for '{query}' — use the main tutor pipeline for factual answers]"


registry.register("get_current_time", "Get the current date and time.", get_current_time)
registry.register("calculate", "Evaluate a math expression. Example: calculate(expression='sqrt(144) + 3**2')", calculate)
registry.register("search_knowledge_base", "Search uploaded documents and knowledge graph for information.", search_knowledge_base)
