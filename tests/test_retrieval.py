import json
from config import ROOT
from ingest.pipeline import load_knowledge_base
from rag.retriever_v2 import HybridRetriever
from scripts.run_eval import evaluate


def test_vpn_query_returns_vpn_doc():
    retriever = HybridRetriever(load_knowledge_base(ROOT / 'data' / 'docs'))
    results = retriever.search('VPN 오류 720')
    assert results[0].chunk.source == 'vpn_guide.md'
    assert results[0].score > 0


def test_unknown_query_no_evidence():
    retriever = HybridRetriever(load_knowledge_base(ROOT / 'data' / 'docs'))
    assert not retriever.search('zxqw12345')


def test_eval_metrics_range():
    retriever = HybridRetriever(load_knowledge_base(ROOT / 'data' / 'docs'))
    golden = json.loads((ROOT / 'data' / 'golden.json').read_text(encoding='utf-8'))
    scores = evaluate(retriever, golden, k=3)
    assert scores['questions'] == len(golden)
    assert all(0 <= scores[key] <= 1 for key in ('hit_at_k', 'recall_at_k', 'mrr_at_k'))
