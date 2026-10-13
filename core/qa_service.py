"""Channel-independent query service and basic runtime observability."""
from time import perf_counter
from prometheus_client import Counter, Histogram

from rag.agent import RAGAgent
from rag.generator import RAGGenerator

REQUESTS = Counter('rag_requests_total', 'Total RAG requests by outcome', ['mode'])
LATENCY = Histogram('rag_request_duration_seconds', 'RAG query duration in seconds')


class QAService:
    def __init__(self, agent: RAGAgent, generator: RAGGenerator, default_top_k: int = 5):
        self.agent = agent
        self.generator = generator
        self.default_top_k = default_top_k

    def process_query(self, question: str, top_k: int | None = None) -> dict:
        start = perf_counter()
        try:
            results = self.agent.run(question, top_k=top_k or self.default_top_k)
            answer, mode, warning = self.generator.generate(question, results)
            REQUESTS.labels(mode=mode).inc()
            return {
                'answer': answer,
                'mode': mode,
                'warning': warning,
                'sources': [
                    {
                        'id': r.chunk.id,
                        'source': r.chunk.source,
                        'title': r.chunk.title,
                        'score': r.score,
                        'bm25_rank': r.rank_bm25,
                        'vector_rank': r.rank_vector,
                        'excerpt': r.chunk.text[:250],
                    }
                    for r in results
                ],
            }
        finally:
            LATENCY.observe(perf_counter() - start)
