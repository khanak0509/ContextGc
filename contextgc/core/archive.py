import json
import string
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

def tokenize(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, ' ')
    
    words = text.split()
    
    stop_words = {
        "a", "an", "and", "the", "in", "is", "it", "to", "of", "for", "on", "with", "as", "by", "at", "from", 
        "what", "where", "how", "why", "who", "when", "do", "did", "you", "my", "i", "me", "we", "this", 
        "that", "are", "can", "will", "would", "could", "should"
    }
    
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
            
    return filtered_words

class MessageArchive:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.documents = {}

    def archive_message(self, msg_id, role, content, timestamp, metadata=None):
        if metadata is None:
            metadata = {}
            
        doc = Document(
            page_content=content,
            metadata={
                "id": msg_id,
                "role": role,
                "timestamp": timestamp,
                "metadata_json": json.dumps(metadata)
            }
        )
        
        try:
            self.vectorstore.add_documents([doc], ids=[msg_id])
        except Exception:
            self.vectorstore.add_documents([doc])
            
        self.documents[msg_id] = doc

    def recall_relevant_messages(self, goal_text, threshold=0.3):
        results_dict = {}

        docs_and_scores = self.vectorstore.similarity_search_with_score(goal_text, k=10)
        
        for doc, score in docs_and_scores:
            if score >= threshold:
                doc_id = doc.metadata.get("id")
                results_dict[doc_id] = (doc, score)

        if BM25Okapi and self.documents:
            tokenized_query = tokenize(goal_text)
            
            if tokenized_query:
                doc_list = list(self.documents.values())
                corpus = []
                for d in doc_list:
                    corpus.append(tokenize(d.page_content))
                
                has_tokens = False
                for tokens in corpus:
                    if len(tokens) > 0:
                        has_tokens = True
                        break
                
                if has_tokens:
                    bm25 = BM25Okapi(corpus)
                    bm25_scores = bm25.get_scores(tokenized_query)
                    
                    bm25_results = []
                    for i in range(len(doc_list)):
                        bm25_results.append((doc_list[i], bm25_scores[i]))
                        
                    bm25_results.sort(key=lambda x: x[1], reverse=True)
                    
                    for doc, bm25_score in bm25_results[:3]:
                        if bm25_score > 0:
                            doc_id = doc.metadata.get("id")
                            if doc_id in results_dict:
                                old_doc, old_score = results_dict[doc_id]
                                results_dict[doc_id] = (old_doc, max(old_score, 0.95))
                            else:
                                results_dict[doc_id] = (doc, 0.90)

        results = []
        for doc_id, (doc, score) in results_dict.items():
            meta = doc.metadata
            metadata_str = meta.get("metadata_json", "{}")
            parsed_metadata = json.loads(metadata_str)
            
            item = {
                "id": meta.get("id"),
                "role": meta.get("role"),
                "content": doc.page_content,
                "timestamp": meta.get("timestamp"),
                "metadata": parsed_metadata,
            }
            results.append((item, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def delete_messages(self, ids):
        if not ids:
            return
            
        try:
            self.vectorstore.delete(ids=ids)
        except Exception:
            pass
            
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
