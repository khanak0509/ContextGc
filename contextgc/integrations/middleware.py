"""LangGraph / LangChain agent middleware for ContextGC.

Provides a simple callable middleware that trims messages before each model call.
"""

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage

from contextgc.integrations.langchain import ContextGCMemory


class ContextGCMiddleware:
    """Runs ContextGC on agent messages before each model call.

    Compatible with LangGraph's functional middleware pattern:
        middleware = ContextGCMiddleware(memory)
        trimmed_messages = middleware(state)
    """

    def __init__(self, memory: ContextGCMemory):
        self.memory = memory

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any] | None:
        """Trim messages in state before passing to the model."""
        messages: List[BaseMessage] = state.get("messages", [])
        if not messages:
            return None

        trimmed = self.memory.process_langchain_messages(messages)
        print(self.memory.log_status(trimmed))
        return {"messages": trimmed}

    def before_model(self, state: Dict[str, Any], runtime: Any = None) -> Dict[str, Any] | None:
        """Alternative hook-style interface for frameworks that use before_model."""
        return self.__call__(state)
