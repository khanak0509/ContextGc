import time
import uuid

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from contextgc.core.eviction import EvictionOrchestrator

class ContextGCGraphNode:
    def __init__(self, model="qwen2.5", max_tokens=None, watermark=0.8, state_path=None):
        self.gc = EvictionOrchestrator(
            model=model, 
            max_tokens=max_tokens, 
            watermark=watermark, 
            state_path=state_path
        )

    def normalize(self, messages):
        raw_msgs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "system"
                
            msg_id = getattr(msg, "id", None)
            if not msg_id:
                msg_id = str(uuid.uuid4())
                
            raw_msgs.append({
                "id": msg_id,
                "role": role,
                "content": str(msg.content),
                "timestamp": time.time(),
                "metadata": {}
            })
            
        return raw_msgs

    def denormalize(self, raw_msgs):
        lc_msgs = []
        for m in raw_msgs:
            if m["role"] == "system":
                lc_msgs.append(SystemMessage(content=m["content"], id=m["id"]))
            elif m["role"] == "user":
                lc_msgs.append(HumanMessage(content=m["content"], id=m["id"]))
            elif m["role"] == "assistant":
                lc_msgs.append(AIMessage(content=m["content"], id=m["id"]))
                
        return lc_msgs

    def __call__(self, state):
        raw_msgs = self.normalize(state["messages"])
        processed_msgs = self.gc.process(raw_msgs)
        state["messages"] = self.denormalize(processed_msgs)
        
        return {"messages": state["messages"]}
