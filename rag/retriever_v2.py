"""BM25 + vector-like channel fused by Reciprocal Rank Fusion.

TF-IDF char n-grams (default) are a cheap *lexical baseline*, not embeddings.
Set RAG_EMBEDDING_BACKEND=sentence_transformers for actual semantic vectors.
"""
from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ingest.pipeline import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r'[가-힣a-z0-9]+', text.lower())


@dataclass(frozen=True)
class Result:
    chunk: Chunk
    score: float
    rank_bm25: int | None
    rank_vector: int | None


class HybridRetriever:
    def __init__(self, chunks: Sequence[Chunk], backend: str = 'tfidf',
                 model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.chunks = list(chunks)
        self.backend = backend
        self.token_lists = [tokenize(c.text) for c in self.chunks]
        self.doc_counter = [Counter(t) for t in self.token_lists]
        self.avg_len = sum(map(len, self.token_lists)) / max(1, len(self.chunks))
        self.doc_freq = Counter(token for toks in self.token_lists for token in set(toks))
        self.vectorizer = None
        self.model = None
        self.matrix = None
        if not self.chunks:
            return
        if backend == 'tfidf':
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        elif backend == 'sentence_transformers':
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError('Install requirements-semantic.txt to use sentence_transformers') from exc
            self.model = SentenceTransformer(model_name)
            self.matrix = np.asarray(self.model.encode([c.text for c in self.chunks], normalize_embeddings=True))
        else:
            raise ValueError(f'Unknown embedding backend: {backend}')

    def _bm25_scores(self, query: str) -> np.ndarray:
        n = len(self.chunks)
        scores = np.zeros(n, dtype=float)
        k1, b = 1.5, 0.75
        for term in set(tokenize(query)):
            df = self.doc_freq.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, counter in enumerate(self.doc_counter):
                freq = counter.get(term, 0)
                if not freq:
                    continue
                length = len(self.token_lists[i])
                scores[i] += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * length / max(self.avg_len, 1)))
        return scores

    def _vector_scores(self, query: str) -> np.ndarray:
        if self.backend == 'tfidf':
            return cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        query_embedding = np.asarray(self.model.encode([query], normalize_embeddings=True))[0]
        return np.asarray(self.matrix @ query_embedding, dtype=float)

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[Result]:
        if not self.chunks or not query.strip():
            return []
        bm25 = self._bm25_scores(query)
        vectors = self._vector_scores(query)
        # Never include zero-score candidates as if they were relevant.
        ranks: list[dict[int, int]] = []
        for values in (bm25, vectors):
            ranked = sorted((i for i, s in enumerate(values) if s > 1e-9),
                            key=lambda i: (-values[i], self.chunks[i].id))
            ranks.append({i: rank for rank, i in enumerate(ranked, 1)})
        bm25_ranks, vector_ranks = ranks
        fused = {}
        for rank_map in ranks:
            for i, r in rank_map.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r)
        candidates = sorted(fused, key=lambda i: (-fused[i], self.chunks[i].id))[:max(0, top_k)]
        return [Result(self.chunks[i], round(fused[i], 8), bm25_ranks.get(i), vector_ranks.get(i))
                for i in candidates]
