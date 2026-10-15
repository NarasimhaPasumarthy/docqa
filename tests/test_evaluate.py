from pathlib import Path

from docqa.evaluate import evaluate_retrieval, load_cases
from docqa.ingest import ingest_paths
from docqa.retriever import BM25Index

ROOT = Path(__file__).parent.parent


def test_retrieval_quality_on_sample_docs():
    """Quality gate: retrieval must find the right document for every labelled question."""
    index = BM25Index()
    ingest_paths(index, [ROOT / "sample_docs"])
    report = evaluate_retrieval(index, load_cases(ROOT / "eval" / "cases.jsonl"), k=3)
    misses = [q for q, rank in report.ranks if rank is None]
    assert report.hit_rate >= 0.9, misses
