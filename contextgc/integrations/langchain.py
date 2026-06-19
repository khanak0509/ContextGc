import uuid
import time
import sqlite3

from pydantic import ConfigDict, Field, model_validator
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from contextgc.core.eviction import EvictionOrchestrator


class ContextGCMemory(BaseChatMemory):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    model: str = "llama3.1"
    vectorstore: object = Field(default=None, exclude=True)
    memory_key: str = "history"
    return_messages: bool = True

    gc_messages: list = Field(default_factory=list)
    gc_goal: str = "general assistance"
    gc_token_log: list = Field(default_factory=list)
    gc: object = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def init(self):
        if self.gc is None:
            self.gc = EvictionOrchestrator(model=self.model, vectorstore=self.vectorstore)
        return self

    @property
    def memory_variables(self):
        return [self.memory_key]

    @staticmethod
    def _msg_content(msg):
        content = str(msg.content or "")
        if isinstance(msg, AIMessage) and msg.tool_calls:
            content = f"{content} {msg.tool_calls}".strip()
        return content

    @staticmethod
    def _lc_to_raw(messages):
        raw = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = "system"
            meta = dict(getattr(msg, "additional_kwargs", {}) or {})
            if isinstance(msg, ToolMessage):
                meta["tool_call_id"] = msg.tool_call_id
                meta["tool_name"] = msg.name
            raw.append({
                "id": str(getattr(msg, "id", None) or uuid.uuid4()),
                "role": role,
                "content": ContextGCMemory._msg_content(msg),
                "timestamp": time.time(),
                "metadata": meta,
            })
        return raw

    @staticmethod
    def _raw_to_lc(msgs):
        out = []
        for m in msgs:
            role, content, mid = m["role"], m["content"], m.get("id")
            meta = m.get("metadata", {})
            if role == "user":
                out.append(HumanMessage(content=content, id=mid))
            elif role == "assistant":
                out.append(AIMessage(content=content, id=mid))
            elif role == "tool":
                out.append(ToolMessage(content=content, tool_call_id=meta.get("tool_call_id", ""), name=meta.get("tool_name"), id=mid))
            else:
                out.append(SystemMessage(content=content, id=mid))
        return out

    def _add(self, role, content, msg_id=None):
        self.gc_messages.append({
            "id": msg_id or str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": {},
        })

    def _run_gc(self):
        out = self.gc.process(self.gc_messages)
        self.gc_messages = out
        self.gc_goal = self.gc.current_goal
        self.gc_token_log.append(self.gc.count_tokens(out))
        return out

    def process_langchain_messages(self, messages):
        self.gc_messages = self._lc_to_raw(messages)
        return self._raw_to_lc(self._run_gc())

    def load_memory_variables(self, inputs):
        out = self._run_gc()
        if self.return_messages:
            return {self.memory_key: self._raw_to_lc(out)}
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in out]
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs, outputs):
        input_str, output_str = self._get_input_output(inputs, outputs)
        self._add("user", str(input_str))
        self._add("assistant", str(output_str))

    def add_message(self, message):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            role = "system"
        mid = message.additional_kwargs.get("message_id", str(uuid.uuid4()))
        self._add(role, self._msg_content(message), msg_id=mid)

    def clear(self):
        self.gc_messages.clear()
        self.gc_token_log.clear()
        self.gc.total_evictions = 0
        self.gc.total_recalls = 0
        if self.gc.vectorstore and hasattr(self.gc.vectorstore, "delete"):
            try:
                # Naive clearing, assuming all docs belong to this session.
                # Production users should handle their own vectorstore cleanup.
                pass 
            except Exception:
                pass

    def log_status(self, messages):
        tokens = self.gc.count_tokens(self._lc_to_raw(messages))
        return (
            f"[ContextGC] Tokens: {tokens} | "
            f"Msgs: {len(messages)} | "
            f"Evictions: {self.gc.total_evictions} | "
            f"Goal: {self.gc_goal[:60]}"
        )
