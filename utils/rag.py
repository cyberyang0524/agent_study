
import os
import math
from typing import List, Dict, Any, Tuple
from utils.llm_client import LLMClient

FAQ_FILE = r"d:\agent\faq.txt"

class KnowledgeBase:
    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] = []
        self.client = LLMClient()
        self._load_knowledge()

    def _load_knowledge(self):
        if not os.path.exists(FAQ_FILE):
            return
        
        with open(FAQ_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Simple splitting by double newline (paragraphs)
        raw_chunks = content.split("\n\n")
        self.chunks = [c.strip() for c in raw_chunks if c.strip()]
        
        # In a real app, we would cache embeddings to disk
        # Here we compute them on startup (slow for large files, ok for demo)
        print(f"Loading knowledge base with {len(self.chunks)} chunks...")
        for chunk in self.chunks:
            emb = self.client.get_embedding(chunk)
            self.embeddings.append(emb)
            
    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self.chunks:
            return []
            
        query_emb = self.client.get_embedding(query)
        if not query_emb:
            return []
            
        scores = []
        for i, doc_emb in enumerate(self.embeddings):
            score = self._cosine_similarity(query_emb, doc_emb)
            scores.append((score, self.chunks[i]))
            
        # Sort by score desc
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [chunk for score, chunk in scores[:top_k]]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

# Singleton instance
_KB_INSTANCE = None

def get_knowledge_base():
    global _KB_INSTANCE
    if _KB_INSTANCE is None:
        _KB_INSTANCE = KnowledgeBase()
    return _KB_INSTANCE
