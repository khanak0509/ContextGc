import json
import os
import time
import requests

from contextgc.core.scorer import MessageScorer
from contextgc.core.archive import MessageArchive
from contextgc.core.compressor import MessageCompressor
from contextgc.core.state import CoreState

_limits_path = os.path.join(os.path.dirname(__file__), "..", "model_limits.json")
try:
    with open(_limits_path) as f:
        MODEL_LIMITS = json.load(f)
except Exception:
    MODEL_LIMITS = {}


def get_token_limit(model):
    return MODEL_LIMITS.get(model, 32000)


def tok(text):
    return max(1, int(len(text.split()) * 1.3))


class EvictionOrchestrator:
    def __init__(self, model="llama3.2:3b", archive_path="gc.db", max_tokens=None, watermark=0.8, state_path=None):
        self.model = model
        self.archive_path = archive_path
        self.max_tokens = max_tokens
        self.watermark = watermark
        self.current_goal = "general assistance"
        self.total_evictions = 0
        self.total_recalls = 0
        self.scorer = MessageScorer()
        self.archive = MessageArchive(archive_path)
        self.compressor = MessageCompressor(model)

        # derive state path from archive path if not given
        sp = state_path or archive_path.replace(".db", "_state.json")
        self.core_state = CoreState(model, sp)

    def count_tokens(self, msgs):
        return sum(tok(m["content"]) for m in msgs)

    def _trigger_limit(self):
        ceiling = self.max_tokens or get_token_limit(self.model)
        return int(ceiling * self.watermark)

    def _recall(self, msgs):
        # find the most recent user message to use as the search query
        user_msgs = [m for m in msgs if m["role"] == "user"]
        if not user_msgs:
            return msgs
        
        query = user_msgs[-1]["content"]

        goal_vec = self.scorer.compute_embeddings([query])[0]
        hits = self.archive.recall_relevant_messages(goal_vec, threshold=0.50)
        if not hits:
            return msgs

        existing = {m["id"] for m in msgs}
        recalled_ids = []
        for item, _ in hits:
            if item["id"] not in existing:
                # inject after system messages
                insert_at = next((i + 1 for i, m in enumerate(msgs) if m["role"] == "system"), 0)
                msgs.insert(insert_at, item)
                recalled_ids.append(item["id"])
                self.total_recalls += 1

        if recalled_ids:
            self.archive.delete_messages(recalled_ids)

        return msgs

    def _strip_injected_state(self, msgs):
        return [m for m in msgs if not m.get("metadata", {}).get("_gc_core_state")]

    def _inject_core_state(self, msgs):
        text = self.core_state.as_system_message()
        if not text:
            return msgs

        msgs = self._strip_injected_state(msgs)
        state_msg = {
            "id": "_gc_core_state",
            "role": "system",
            "content": text,
            "timestamp": time.time(),
            "metadata": {"_gc_core_state": True},
        }
        # insert right after any original system prompts
        insert_at = 0
        for i, m in enumerate(msgs):
            if m["role"] == "system":
                insert_at = i + 1
            else:
                break
        msgs.insert(insert_at, state_msg)
        return msgs

    def _block_evict(self, msgs):
        sys_msgs = [m for m in msgs if m["role"] == "system"]
        chat_msgs = [m for m in msgs if m["role"] != "system"]

        # always protect the last 6 chat messages (3 full turns)
        protect = 6
        if len(chat_msgs) <= protect:
            return msgs  # nothing safe to evict

        to_evict = chat_msgs[:-protect]
        keep = chat_msgs[-protect:]

        # extract facts from the evicted block — this is the SOTA upgrade
        # only calls Ollama here, not on every single turn
        self.core_state.extract_from(to_evict)

        # archive all evicted messages with their embeddings for future recall
        for msg in to_evict:
            vec = self.scorer.compute_embeddings([msg["content"]])[0]
            self.archive.archive_message(
                msg["id"], msg["role"], msg["content"],
                vec, msg.get("timestamp", time.time()), msg.get("metadata", {}),
            )

        self.total_evictions += len(to_evict)

        topic = self.core_state.data.current_topic or f"{len(to_evict)} turns"
        breadcrumb = {
            "id": f"_gc_crumb_{int(time.time())}",
            "role": "system",
            "content": f"[Archived {len(to_evict)} messages — Topic: {topic}. Key facts preserved in memory above.]",
            "timestamp": time.time(),
            "metadata": {"_gc_breadcrumb": True},
        }

        return sys_msgs + [breadcrumb] + keep

    def process(self, msgs):
        msgs = list(msgs)

        # 1. pull relevant archived messages back in
        msgs = self._recall(msgs)

        # 2. inject current core state so the agent always sees persistent facts
        msgs = self._inject_core_state(msgs)

        # 3. check if we need to evict anything
        if self.count_tokens(msgs) <= self._trigger_limit():
            return msgs

        # 4. block evict — extract facts, archive, leave breadcrumb
        msgs = self._block_evict(msgs)

        # update current_goal from the now-updated core state
        if self.core_state.data.current_topic:
            self.current_goal = self.core_state.data.current_topic

        # 6. re-inject updated core state (now has new facts from the evicted block)
        msgs = self._inject_core_state(msgs)

        return msgs
