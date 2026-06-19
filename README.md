# ContextGC

ContextGC is an automatic context window manager for LLM chat applications. 

If your chatbot runs long enough, it will eventually hit its token limit. Most solutions just blindly truncate old messages, causing the LLM to forget important facts (like the user's name or what they just discussed 10 turns ago). ContextGC intercepts your message array before it hits the LLM, evicts old messages, extracts the core facts into a persistent state, and uses BM25 + Vector Search to pull deeply archived messages back into context exactly when the user asks about them.

## Features

- **Fact Extraction:** Summarizes and saves core entities, topics, and user preferences into a structured JSON state.
- **Smart Eviction:** Triggers only when you hit a memory watermark, cleanly evicting old messages while protecting recent chat flow.
- **Hybrid Recall:** Combines BM25 (keyword search) and Vector Search (semantic similarity) to retrieve evicted messages with high precision.
- **100% Local by Default:** Built to run perfectly offline with Ollama models (`llama3.2:3b`, `qwen2.5`, etc.).

## Installation

```bash
git clone https://github.com/khanak0509/ContextGc.git
cd ContextGc
pip install -e .
```

Make sure you have Ollama installed and your preferred model pulled:

```bash
ollama pull qwen2.5
```

## Quick Start (LangChain Integration)

ContextGC is designed to seamlessly sit in front of any LLM call. Here is a complete, end-to-end example of how to use it with LangChain:

```python
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from contextgc.core.eviction import EvictionOrchestrator

# 1. Initialize your LLM
llm = ChatOllama(model="qwen2.5", temperature=0)

# 2. Initialize the Context Garbage Collector
gc = EvictionOrchestrator(
    model="qwen2.5", 
    max_tokens=2000,   # Set your model's context limit
    watermark=0.80,    # Evict when 80% full
    state_path="memory_state.json"
)

# 3. Create your chat history in ContextGC's standard dictionary format
messages = [
    {"id": "msg_1", "role": "system", "content": "You are a helpful assistant.", "timestamp": time.time()},
    {"id": "msg_2", "role": "user", "content": "Hi, my name is Khanak and I built a startup called Likha.", "timestamp": time.time()},
]

# 4. Process the context window (Evict old messages & inject state if needed)
messages = gc.process(messages)

# 5. Convert the processed dictionaries into LangChain Message objects
langchain_msgs = []
for m in messages:
    if m["role"] == "system":
        langchain_msgs.append(SystemMessage(content=m["content"]))
    elif m["role"] == "user":
        langchain_msgs.append(HumanMessage(content=m["content"]))
    elif m["role"] == "assistant":
        langchain_msgs.append(AIMessage(content=m["content"]))

# 6. Call your LLM!
response = llm.invoke(langchain_msgs)
print("AI Response:", response.content)

# 7. (Optional) Save the AI's response back to your history for the next turn
messages.append({
    "id": f"msg_{time.time()}", 
    "role": "assistant", 
    "content": response.content, 
    "timestamp": time.time()
})
```

## How It Works Under the Hood

When you call `gc.process(messages)`:

1. **State Injection:** It injects a system message containing the current known facts about the user.
2. **Limit Checking:** It counts the tokens. If you are under the watermark, it does nothing else.
3. **Eviction:** If you are over the limit, it evicts older messages, runs them through the LLM to extract new facts, and archives the raw text into an in-memory vector store and BM25 index.
4. **Recall:** It looks at the user's latest query. If the query matches anything in the archive, it injects those specific past messages back into the context window as a `RECALLED MEMORY CONTEXT` block.

## Architecture & Integrations

ContextGC is completely framework-agnostic. Because it just takes a list of standard dictionary messages and returns a modified list of messages, you can easily plug it into LangChain, LlamaIndex, or raw OpenAI/Ollama API calls.

## Requirements
- `langchain_ollama`
- `langchain_core`
- `rank_bm25`
- `pydantic`

## Contributing

Pull requests are welcome! Keep things simple and ensure any new logic is covered by tests.
