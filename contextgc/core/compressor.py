import os
import re
import requests

DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def _extractive(content):
    text = content.strip().replace("\n", " ")
    if not text:
        return "[empty]"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    return " ".join(sentences[:2]) if sentences else " ".join(text.split()[:20])


class MessageCompressor:
    def __init__(self, model=None):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    def summarize(self, content, role="user"):
        prompt = (
            f"Compress this into one dense sentence, keep key facts and error codes: {content}"
        )
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=8,
            )
            result = r.json().get("response", "").strip()
            return result.replace("\n", " ") if result else _extractive(content)
        except Exception:
            return _extractive(content)
