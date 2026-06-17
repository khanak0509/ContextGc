import numpy as np
from langchain_ollama import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class MessageScorer:
    def __init__(self, model="nomic-embed-text"):
        self.model = OllamaEmbeddings(model=model)

    def compute_embeddings(self, texts):
        if not texts:
            return []
        return self.model.embed_documents(texts)

    def score_history(self, msgs, goal):
        if not msgs:
            return []
        n = len(msgs)
        texts = [m["content"] for m in msgs]
        recency = [float(i + 1) / n for i in range(n)]
        relevance = self._relevance(texts, goal)
        density = self._density(texts, n)
        return [
            0.2 * recency[i] + 0.5 * relevance[i] + 0.3 * density[i]
            for i in range(n)
        ]

    def _relevance(self, texts, goal):
        goal_emb = np.array(self.embed.encode(goal), dtype=np.float32)
        goal_norm = np.linalg.norm(goal_emb)
        msg_embs = self.embed.encode(texts, convert_to_numpy=True)
        scores = []
        for emb in msg_embs:
            norm = np.linalg.norm(emb)
            if goal_norm == 0 or norm == 0:
                scores.append(0.0)
            else:
                sim = float(np.dot(goal_emb, emb) / (goal_norm * norm))
                scores.append(max(0.0, min(1.0, sim)))
        return scores

    def _density(self, texts, n):
        if n <= 1:
            return [1.0]
        try:
            mat = TfidfVectorizer().fit_transform(texts).toarray()
            raw = []
            for row in mat:
                nz = row[row > 0]
                raw.append(float(np.mean(nz)) if nz.size > 0 else 0.0)
            mx = max(raw) or 1.0
            return [s / mx for s in raw]
        except Exception:
            return [1.0] * n
