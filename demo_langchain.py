from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationChain
from contextgc.integrations.langchain import ContextGCMemory

llm = ChatOllama(model="qwen2.5", temperature=0)

# ContextGCMemory is a drop-in replacement for standard LangChain memory
memory = ContextGCMemory(
    model="qwen2.5",
    max_tokens=1000,
    watermark=0.80,
    state_path="chatbot_memory_state.json"
)

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
        
    result = chain.invoke({"input": user_input})
    print("AI:", result["response"])

