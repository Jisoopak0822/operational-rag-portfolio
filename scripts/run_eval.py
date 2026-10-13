"""LLM-independent retrieval evaluation: hit@k, recall@k, MRR@k.
Usage: python scripts/run_eval.py --retrieval-only --k 3
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import settings  # noqa: E402
from ingest.pipeline import load_knowledge_base  # noqa: E402
from rag.retriever_v2 import HybridRetriever  # noqa: E402


def evaluate(retriever, golden: list[dict], k: int) -> dict:
    if not golden:
        raise ValueError('Golden set cannot be empty')
    hits, recalls, reciprocal_ranks = [], [], []
    per_query = []
    for item in golden:
        results = retriever.search(item['question'], top_k=k)
        ranked = [r.chunk.source for r in results]
        gold = set(item['gold_sources'])
        found = gold.intersection(ranked)
        first = next((i for i, name in enumerate(ranked, 1) if name in gold), None)
        hits.append(int(bool(found)))
        recalls.append(len(found) / len(gold))
        reciprocal_ranks.append(1 / first if first else 0.0)
        per_query.append({'question': item['question'], 'top_sources': ranked,
                          'hit': bool(found), 'first_rank': first})
    n = len(golden)
    return {'k': k, 'questions': n, 'hit_at_k': round(sum(hits) / n, 4),
            'recall_at_k': round(sum(recalls) / n, 4),
            'mrr_at_k': round(sum(reciprocal_ranks) / n, 4),
            'per_query': per_query}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval-only', action='store_true', help='No LLM calls are made regardless of this flag')
    parser.add_argument('--k', type=int, default=3)
    args = parser.parse_args()
    retriever = HybridRetriever(load_knowledge_base(settings.docs_dir),
                                backend=settings.embedding_backend,
                                model_name=settings.embedding_model)
    golden = json.loads((ROOT / 'data' / 'golden.json').read_text(encoding='utf-8'))
    print(json.dumps(evaluate(retriever, golden, args.k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
