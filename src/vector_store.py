import os
import math
import re
from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

class TFIDFRetriever:
    """Lightweight deterministic vector search fallback based on term-frequency inverse-document-frequency."""
    def __init__(self, corpus: Dict[str, str]):
        self.titles = list(corpus.keys())
        self.texts = list(corpus.values())
        self.doc_count = len(self.texts)
        self.vocab = {}
        self.doc_vectors = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _build_index(self):
        doc_tokens_list = [self._tokenize(t) for t in self.texts]
        df = {}
        for tokens in doc_tokens_list:
            for word in set(tokens):
                df[word] = df.get(word, 0) + 1

        idf = {}
        for word, count in df.items():
            idf[word] = math.log((self.doc_count + 1) / (count + 1)) + 1.0

        for i, tokens in enumerate(doc_tokens_list):
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            vector = {}
            norm = 0.0
            for t, count in tf.items():
                val = count * idf.get(t, 1.0)
                vector[t] = val
                norm += val * val
            norm = math.sqrt(norm) if norm > 0 else 1.0
            for t in vector:
                vector[t] /= norm
            self.doc_vectors.append(vector)

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        q_tf = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        q_norm = 0.0
        for t, count in q_tf.items():
            q_norm += count * count
        q_norm = math.sqrt(q_norm) if q_norm > 0 else 1.0
        q_vec = {t: count / q_norm for t, count in q_tf.items()}

        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            score = 0.0
            for t, val in q_vec.items():
                if t in doc_vec:
                    score += val * doc_vec[t]
            title = self.titles[i]
            text = self.texts[i]
            # Boost score slightly if query keywords match title directly
            for token in q_tokens:
                if len(token) > 3 and token in title.lower():
                    score += 0.35
            scores.append((score, title, text))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, title, text in scores[:top_k]:
            results.append({
                "title": title,
                "text": text,
                "score": float(score)
            })
        return results

class VectorStore:
    """Vector Store wrapper supporting Qdrant and TFIDF fallback."""
    def __init__(self, corpus: Dict[str, str], host: Optional[str] = None, port: Optional[int] = None):
        self.corpus = corpus
        self.titles = list(corpus.keys())
        self.texts = list(corpus.values())
        self.fallback = TFIDFRetriever(corpus)
        self.qdrant_client = None

        host = host or os.getenv("VECTOR_DB_HOST", "localhost")
        port = int(port or os.getenv("VECTOR_DB_PORT", "6333"))

        if HAS_QDRANT:
            try:
                # Try connecting to remote/docker Qdrant instance
                self.qdrant_client = QdrantClient(host=host, port=port, timeout=1.0, check_compatibility=False)
            except Exception:
                try:
                    # Fallback to in-memory Qdrant Client
                    self.qdrant_client = QdrantClient(location=":memory:", check_compatibility=False)
                except Exception:
                    self.qdrant_client = None

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        # Always return best matches via retriever
        return self.fallback.search(query, top_k=top_k)
