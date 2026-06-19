import json
import os
import time

from langchain_core.vectorstores import InMemoryVectorStore
from contextgc.core.scorer import MessageScorer
from contextgc.core.archive import MessageArchive
from contextgc.core.state import CoreState

limits_path = os.path.join(os.path.dirname(__file__), "..", "model_limits.json")
try:
    with open(limits_path) as f:
        MODEL_LIMITS = json.load(f)
except Exception:
    MODEL_LIMITS = {}

def get_token_limit(model):
    if model in MODEL_LIMITS:
        return MODEL_LIMITS[model]
    return 32000

def tok(text):
    words_count = len(text.split())
    estimated_tokens = int(words_count * 1.3)
    return max(1, estimated_tokens)

class EvictionOrchestrator:
    def __init__(self, model="llama3.2:3b", vectorstore=None, max_tokens=None, watermark=0.8, state_path=None):
        self.model = model
        self.max_tokens = max_tokens
        self.watermark = watermark
        self.current_goal = "general assistance"
        self.total_evictions = 0
        self.total_recalls = 0

        if vectorstore is None:
            scorer = MessageScorer()
            embed_model = scorer.model
            vectorstore = InMemoryVectorStore(embedding=embed_model)
            
        self.vectorstore = vectorstore
        self.archive = MessageArchive(self.vectorstore)

        if state_path is None:
            state_path = "chatbot_memory_state.json"
            
        self.core_state = CoreState(model, state_path)

    def count_tokens(self, msgs):
        total = 0
        for m in msgs:
            total += tok(m["content"])
        return total

    def trigger_limit(self):
        if self.max_tokens is not None:
            ceiling = self.max_tokens
        else:
            ceiling = get_token_limit(self.model)
            
        return int(ceiling * self.watermark)

    def recall(self, msgs):
        user_msgs = []
        for m in msgs:
            if m["role"] == "user":
                user_msgs.append(m)
                
        if len(user_msgs) == 0:
            return msgs
        
        last_msg = user_msgs[-1]
        query = last_msg["content"]

        hits = self.archive.recall_relevant_messages(query, threshold=0.50)
        
        if len(hits) == 0:
            return msgs

        existing_ids = set()
        for m in msgs:
            existing_ids.add(m["id"])
            
        recalled_items = []
        for item, score in hits:
            if item["id"] not in existing_ids:
                recalled_items.append(item)

        if len(recalled_items) == 0:
            return msgs

        recalled_items.sort(key=lambda x: x.get("timestamp", 0))

        memory_text = "--- RECALLED MEMORY CONTEXT ---\nThe following are past messages relevant to the current query:\n\n"
        for item in recalled_items:
            role_label = item["role"].upper()
            content = item["content"]
            memory_text += f"{role_label}: {content}\n\n"
            
        memory_text += "---------------------------------\n"

        current_time = time.time()
        memory_msg = {
            "id": f"gc_recalled_{int(current_time * 1000)}",
            "role": "system",
            "content": memory_text,
            "timestamp": current_time,
            "metadata": {"gc_recalled": True}
        }

        insert_index = 0
        for i in range(len(msgs)):
            m = msgs[i]
            if m["role"] == "system":
                insert_index = i + 1

        msgs.insert(insert_index, memory_msg)
        
        self.total_recalls += len(recalled_items)
        return msgs

    def strip_injected_state(self, msgs):
        cleaned_msgs = []
        for m in msgs:
            meta = m.get("metadata", {})
            if not meta.get("gc_core_state"):
                cleaned_msgs.append(m)
        return cleaned_msgs

    def inject_core_state(self, msgs):
        text = self.core_state.as_system_message()
        if not text:
            return msgs

        msgs = self.strip_injected_state(msgs)
        
        current_time = time.time()
        state_msg = {
            "id": "gc_core_state",
            "role": "system",
            "content": text,
            "timestamp": current_time,
            "metadata": {"gc_core_state": True},
        }
        
        insert_index = 0
        for i in range(len(msgs)):
            m = msgs[i]
            if m["role"] == "system":
                insert_index = i + 1
            else:
                break
                
        msgs.insert(insert_index, state_msg)
        return msgs

    def block_evict(self, msgs):
        sys_msgs = []
        chat_msgs = []
        
        for m in msgs:
            if m["role"] == "system":
                sys_msgs.append(m)
            else:
                chat_msgs.append(m)

        protect_count = 6
        if len(chat_msgs) <= protect_count:
            return msgs

        to_evict = chat_msgs[:-protect_count]
        keep = chat_msgs[-protect_count:]

        self.core_state.extract_from(to_evict)

        for msg in to_evict:
            msg_id = msg["id"]
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", time.time())
            metadata = msg.get("metadata", {})
            
            self.archive.archive_message(
                msg_id, role, content, timestamp, metadata
            )

        self.total_evictions += len(to_evict)

        topic = self.core_state.data.current_topic
        if not topic:
            topic = f"{len(to_evict)} turns"
            
        current_time = time.time()
        breadcrumb = {
            "id": f"gc_crumb_{int(current_time)}",
            "role": "system",
            "content": f"[Archived {len(to_evict)} messages — Topic: {topic}. Key facts preserved in memory above.]",
            "timestamp": current_time,
            "metadata": {"gc_breadcrumb": True},
        }

        final_msgs = []
        final_msgs.extend(sys_msgs)
        final_msgs.append(breadcrumb)
        final_msgs.extend(keep)
        
        return final_msgs

    def process(self, msgs):
        msgs = list(msgs)

        msgs = self.inject_core_state(msgs)

        current_tokens = self.count_tokens(msgs)
        limit = self.trigger_limit()
        
        if current_tokens > limit:
            msgs = self.block_evict(msgs)

            if self.core_state.data.current_topic:
                self.current_goal = self.core_state.data.current_topic

            msgs = self.inject_core_state(msgs)

        msgs = self.recall(msgs)

        return msgs
