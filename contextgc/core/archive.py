import json
import string
from langchain_core.documents import Document

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

def _tokenize(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, ' ')
    words = text.split()
    stop_words = {"a", "an", "and", "the", "in", "is", "it", "to", "of", "for", "on", "with", "as", "by", "at", "from", "what", "where", "how", "why", "who", "when", "do", "did", "you", "my", "i", "me", "we", "this", "that", "are", "can", "will", "would", "could", "should"}
    return [w for w in words if w not in stop_words]

class MessageArchive:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self._documents = {}  # msg_id -> Document

    def archive_message(self, msg_id, role, content, timestamp, metadata=None):
        doc = Document(
            page_content=content,
            metadata={
                "id": msg_id,
                "role": role,
                "timestamp": timestamp,
                "metadata_json": json.dumps(metadata or {})
            }
        )
        # Add to vector store
        try:
            self.vectorstore.add_documents([doc], ids=[msg_id])
        except Exception:
            self.vectorstore.add_documents([doc])
            
        # Add to local BM25 cache
        self._documents[msg_id] = doc

    def recall_relevant_messages(self, goal_text, threshold=0.3):
        results_dict = {}  # doc_id -> (Document, score)

        # 1. Semantic Search (Vector)
        docs_and_scores = self.vectorstore.similarity_search_with_score(goal_text, k=10)
        for doc, score in docs_and_scores:
            if score >= threshold:
                doc_id = doc.metadata.get("id")
                results_dict[doc_id] = (doc, score)

        # 2. Keyword Search (BM25)
        if BM25Okapi and self._documents:
            tokenized_query = _tokenize(goal_text)
            if tokenized_query:
                doc_list = list(self._documents.values())
                corpus = [_tokenize(d.page_content) for d in doc_list]
                
                # Only run BM25 if there's actually a corpus with valid tokens
                if any(corpus):
                    bm25 = BM25Okapi(corpus)
                    bm25_scores = bm25.get_scores(tokenized_query)
                    
                    bm25_results = sorted(zip(doc_list, bm25_scores), key=lambda x: x[1], reverse=True)
                    
                    # Take top 3 BM25 keyword matches
                    for doc, bm25_score in bm25_results[:3]:
                        if bm25_score > 0:
                            doc_id = doc.metadata.get("id")
                            if doc_id in results_dict:
                                old_doc, old_score = results_dict[doc_id]
                                results_dict[doc_id] = (old_doc, max(old_score, 0.95))
                            else:
                                results_dict[doc_id] = (doc, 0.90)

        # Format results
        results = []
        for doc_id, (doc, score) in results_dict.items():
            meta = doc.metadata
            results.append(({
                "id": meta.get("id"),
                "role": meta.get("role"),
                "content": doc.page_content,
                "timestamp": meta.get("timestamp"),
                "metadata": json.loads(meta.get("metadata_json", "{}")),
            }, score))

        # Sort by highest score first
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def delete_messages(self, ids):
        if not ids:
            return
        # Delete from vector store
        try:
            self.vectorstore.delete(ids=ids)
        except Exception:
            pass
            
        # Delete from local BM25 cache
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
