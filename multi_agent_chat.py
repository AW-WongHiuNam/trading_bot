"""Compatibility stub.

The implementation lives in scripts/multi_agent_chat.py.
Keep this file so existing imports keep working:
  - `python multi_agent_chat.py`
  - `import multi_agent_chat as mac`
"""

from scripts.multi_agent_chat import (  # noqa: F401
    DEFAULT_CHAT_MODEL,
    Agent,
    _cfg,
    _print_tool_result,
    _vector_store,
    extract_response_text,
    fetch_and_store_news,
    handle_tool_request,
    run_conversation,
    search_qdrant,
)

__all__ = [
    "DEFAULT_CHAT_MODEL",
    "Agent",
    "extract_response_text",
    "search_qdrant",
    "fetch_and_store_news",
    "handle_tool_request",
    "run_conversation",
    "_cfg",
    "_vector_store",
    "_print_tool_result",
]


if __name__ == "__main__":
    from scripts.multi_agent_chat import run_conversation as _run_conversation

    a = Agent(
        "Alice",
        system_prompt=(
            "You are Alice, a concise analyst of stock. please try to call tools and tell me what you got. "
            "When you need data, issue TOOL commands like \"TOOL: fetch_news_full=AAPL\" or "
            "\"TOOL: search_news=AI chips\". After calling, summarize what you got. "
        ),
    )
    b = Agent("Bob", system_prompt="You are Bob, a a concise analyst of stack. discuss the stock with alice")
    conv = _run_conversation(a, b, turns=6, pause=0.3)
    print("\n--- Final transcript ---")
    for s, t in conv:
        print(f"{s}: {t}")
