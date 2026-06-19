# ContextGC

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=2ecc71&width=435&lines=ContextGC;Automatic+Context+Window+Manager;Never+hit+a+token+limit+again!" alt="Typing Animation" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/Powered_by-Ollama-white?style=for-the-badge" alt="Ollama Badge"/>
  <img src="https://img.shields.io/badge/100%25-Local-success?style=for-the-badge" alt="Local Badge"/>
  <img src="https://img.shields.io/badge/LangChain-Compatible-orange?style=for-the-badge" alt="LangChain Badge"/>
  <img src="https://img.shields.io/badge/LangGraph-Ready-blueviolet?style=for-the-badge" alt="LangGraph Badge"/>
</p>

> **If your chatbot runs long enough, it will eventually hit its token limit.** To prevent the LLM from forgetting important facts (like the user's name or what was discussed 10 turns ago), ContextGC intercepts your message array before it hits the LLM, evicts old messages, extracts the core facts into a persistent state, and uses BM25 + Vector Search to pull deeply archived messages back into context exactly when the user asks about them.

## ✨ Features

- **Fact Extraction:** Summarizes and saves core entities, topics, and user preferences into a structured JSON state.
- **Smart Eviction:** Triggers only when you hit a memory watermark, cleanly evicting old messages while protecting recent chat flow.
- **Hybrid Recall:** Combines BM25 (keyword search) and Vector Search (semantic similarity) to retrieve evicted messages with high precision.
- **100% Local by Default:** Built to run perfectly offline with Ollama models (`llama3.2:3b`, `qwen2.5`, etc.).

## ⚡ Performance & Benchmarks
ContextGC is designed to be completely invisible to the user until an eviction is necessary:
- **Bounded Context Growth**: Guarantees your message array will *never* exceed the token limit you set, no matter how long the conversation lasts (e.g., 10,000+ words are cleanly archived to stay within a strict 2000 token limit).
- **Token Counting Overhead**: `< 0.01 seconds` (Runs on every turn)
- **Hybrid Recall (BM25 + Vector)**: `< 0.10 seconds` (Runs on every turn)
- **State Extraction & Compression**: `~1.5 - 3.0 seconds` (Only runs when the watermark is hit, entirely dependent on your local LLM speed)

## 📊 Evaluation Results

ContextGC has been rigorously benchmarked using completely synthetic local evaluations (available in the `eval/` directory).

### 1. Adversarial Hybrid Recall Precision
We simulated long 50-turn and 100-turn conversations, randomly injected facts, and forced an eviction. We then aggressively queried for the deleted facts using **adversarial queries** (zero keyword overlap/paraphrase gap) and injected distracting facts to confuse the retriever.
- **Recall@1**: `70.0%` (Even with zero keyword overlap and active distractors, the exact deleted fact was the #1 ranked message 70% of the time)
- **Recall@3**: `100.0%` (The deleted fact was **never** lost; it always appeared within the top 3 results)
- **Mean Reciprocal Rank (MRR)**: `0.833`

### 2. Deep State Extraction Quality
State extraction depends entirely on the intelligence of your chosen local LLM. We generated twenty **50-turn conversations** where a single fact was buried deep in the noise, forced a massive block eviction, and measured what percentage of core facts survived the summarization process:
- **`qwen2.5:latest` (7B)**: `100.0%` extraction accuracy (Flawless)
- **`llama3.2:3b` (3B)**: `45.0%` extraction accuracy (Significant degradation. We strongly recommend 7B+ models for highly robust summarization)

## 📦 Installation

```bash
git clone https://github.com/khanak0509/ContextGc.git
cd ContextGc
pip install -e .
```

Make sure you have Ollama installed and your preferred model pulled:

```bash
ollama pull qwen2.5
```

## 🚀 Quick Start (Core API)

ContextGC is entirely framework-agnostic. At its core, it just takes a list of standard python dictionaries and returns a cleaned list of dictionaries. Here is how you use the core `EvictionOrchestrator` if you are using raw LLM APIs (like `openai` or `ollama`):

```python
import time
from contextgc.core.eviction import EvictionOrchestrator

# 1. Initialize the Context Garbage Collector
gc = EvictionOrchestrator(
    model="qwen2.5", 
    max_tokens=2000,   # Set your model's context limit
    watermark=0.80,    # Evict when 80% full
    state_path="memory_state.json"
)

# 2. Create your chat history in ContextGC's standard dictionary format
messages = [
    {"id": "msg_1", "role": "system", "content": "You are a helpful assistant.", "timestamp": time.time()},
    {"id": "msg_2", "role": "user", "content": "Hi, my name is Khanak.", "timestamp": time.time()},
]

# 3. Process the context window (Evicts old messages, saves facts, & injects state)
clean_messages = gc.process(messages)

# 4. Pass the cleaned messages directly to your local LLM!
# For example, using the ollama python client:
# response = ollama.chat(model="qwen2.5", messages=clean_messages)
```

## 🔌 High-Level Integrations

ContextGC provides native wrappers for LangChain and LangGraph to eliminate boilerplate. Completely working, interactive demo files for both frameworks can be found in the repository (`demo_langchain.py` and `demo_langgraph.py`).

### 1. LangChain Integration
`ContextGCMemory` acts as a seamless drop-in replacement for any LangChain memory class.

```python
from contextgc.integrations.langchain import ContextGCMemory
from langchain_classic.chains import ConversationChain

memory = ContextGCMemory(
    model="qwen2.5",
    max_tokens=2000,
    watermark=0.80,
    state_path="chatbot_memory_state.json"
)

chain = ConversationChain(llm=llm, memory=memory)
chain.invoke({"input": "Hello!"})
```

### 2. LangGraph Integration
`ContextGCGraphNode` provides a clean state node that intercepts and cleans your `MessagesState` array before it hits the LLM.

```python
from contextgc.integrations.langgraph import ContextGCGraphNode
from langgraph.graph import StateGraph, START, END, MessagesState

gc_node = ContextGCGraphNode(
    model="qwen2.5",
    max_tokens=2000,
    watermark=0.80,
    state_path="chatbot_memory_state.json"
)

workflow = StateGraph(MessagesState)
workflow.add_node("contextgc", gc_node)
workflow.add_node("agent", agent_node)

# Flow: User Input -> ContextGC (cleans memory) -> LLM Agent
workflow.add_edge(START, "contextgc")
workflow.add_edge("contextgc", "agent")
workflow.add_edge("agent", END)
```

## ⚙️ How It Works Under the Hood

When you call `gc.process(messages)`:

1. **State Injection:** It injects a system message containing the current known facts about the user.
2. **Limit Checking:** It counts the tokens. If you are under the watermark, it does nothing else.
3. **Eviction:** If you are over the limit, it evicts older messages, runs them through the LLM to extract new facts, and archives the raw text into an in-memory vector store and BM25 index.
4. **Recall:** It looks at the user's latest query. If the query matches anything in the archive, it injects those specific past messages back into the context window as a `RECALLED MEMORY CONTEXT` block.

## 🧠 Architecture Explained

At a high level, ContextGC behaves like an operating system's memory pager, but for LLM context windows:

1. **EvictionOrchestrator**: The central brain. It monitors the total token count of your conversation array. When the tokens hit the `watermark` threshold (e.g. 80% of max capacity), it triggers an eviction.
2. **CoreState Extraction**: Instead of just deleting old messages, the orchestrator passes the evicted messages to the LLM behind the scenes. It asks the LLM to extract hard facts, user preferences, and topics, which are then saved to a lightweight persistent JSON file (`memory_state.json`).
3. **MessageArchive (BM25 + VectorStore)**: The raw text of the evicted messages is embedded and stored in a lightweight, built-in vector database (zero external dependencies). Simultaneously, we use `rank_bm25` to index the exact keywords. 
4. **Hybrid Recall**: When the user asks a new question, ContextGC searches both the BM25 index (for exact keyword matches) and the VectorStore (for semantic meaning). If a strong match is found in the graveyard of evicted messages, it pulls them out and temporarily injects them back into the top of your prompt!

ContextGC is completely framework-agnostic. Because it just takes a list of standard dictionary messages and returns a modified list of messages, you can easily plug it into LangChain, LangGraph, LlamaIndex, or raw OpenAI/Ollama API calls.

## 🛠 Requirements
- `langchain_ollama`
- `langchain_core`
- `rank_bm25`
- `pydantic`

