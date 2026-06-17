import time
import uuid

from contextgc.core.eviction import EvictionOrchestrator

# one GC instance per archive path so state persists across graph nodes
_instances = {}


def contextgc_node(state):
    archive_path = state.get("archive_path", "gc.db")
    model = state.get("model", "llama3.1")

    if archive_path not in _instances:
        _instances[archive_path] = EvictionOrchestrator(model=model, archive_path=archive_path)

    gc = _instances[archive_path]
    msgs = _normalize(state.get("messages", []))
    state["messages"] = gc.process(msgs)
    return state


def _normalize(messages):
    out = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append({
                "id": msg.get("id", str(uuid.uuid4())),
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", time.time()),
                "metadata": msg.get("metadata", {}),
            })
            continue
        msg_type = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        if "human" in msg_type:
            role = "user"
        elif "ai" in msg_type or "assistant" in msg_type:
            role = "assistant"
        elif "system" in msg_type:
            role = "system"
        else:
            role = "user"
        out.append({
            "id": getattr(msg, "id", None) or str(uuid.uuid4()),
            "role": role,
            "content": getattr(msg, "content", str(msg)),
            "timestamp": time.time(),
            "metadata": {},
        })
    return out
