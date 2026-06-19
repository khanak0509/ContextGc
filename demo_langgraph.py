from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from contextgc.integrations.langgraph import ContextGCGraphNode

llm = ChatOllama(model="qwen2.5", temperature=0)

# ContextGC Node
# We use aggressive limits here just to force eviction early for the demo
gc_node = ContextGCGraphNode(
    model="qwen2.5",
    max_tokens=1000,
    watermark=0.80,
    state_path="chatbot_memory_state.json"
)

def agent_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("contextgc", gc_node)
workflow.add_node("agent", agent_node)

# The flow: User Input -> ContextGC (cleans memory) -> LLM Agent
workflow.add_edge(START, "contextgc")
workflow.add_edge("contextgc", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()

print("=========================================")
print(" ContextGC + LangGraph Demo ")
print(" Type 'quit' to exit.")
print("=========================================\n")

current_state = {"messages": []}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
        
    current_state["messages"].append(HumanMessage(content=user_input))
    
    # Run the graph
    result = app.invoke(current_state)
    
    # Update local state with the final cleaned messages from the graph
    current_state = result
    
    print("AI:", result["messages"][-1].content)
    print()
