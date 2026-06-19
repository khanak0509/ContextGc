import os
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

DEFAULT_MODEL = "llama3.2:3b"

def extractive(content):
    text = content.strip()
    text = text.replace("\n", " ")
    if not text:
        return "[empty]"
        
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    valid_sentences = []
    for s in sentences:
        if s.strip():
            valid_sentences.append(s)
            
    if len(valid_sentences) > 0:
        return " ".join(valid_sentences[:2])
        
    words = text.split()
    return " ".join(words[:20])

class MessageCompressor:
    def __init__(self, model=None):
        if model is None:
            model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
            
        self.model = model
        self.llm = ChatOllama(model=model, temperature=0)

    def summarize(self, content, role="user"):
        prompt_text = f"Compress this into one dense sentence, keep key facts and error codes: {content}"
        msg = HumanMessage(content=prompt_text)
        
        try:
            result = self.llm.invoke([msg]).content
            result = result.strip()
            
            if result:
                return result.replace("\n", " ")
                
            return extractive(content)
        except Exception:
            return extractive(content)
