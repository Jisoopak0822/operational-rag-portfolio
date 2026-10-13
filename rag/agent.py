"""A slim orchestration boundary; HyDE, Multi-Query, reranking are future stages."""
from rag.retriever_v2 import HybridRetriever, Result


class RAGAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, question: str, top_k: int) -> list[Result]:
        return self.retriever.search(question, top_k=top_k)
