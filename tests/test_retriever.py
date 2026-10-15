from pathlib import Path

from docqa.chunking import Chunk
from docqa.ingest import ingest_paths
from docqa.retriever import BM25Index, tokenize

SAMPLE_DOCS = Path(__file__).parent.parent / "sample_docs"


def _chunk(source: str, text: str, i: int = 0) -> Chunk:
    return Chunk(id=f"{source}#{i}", source=source, position=i, text=text)


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("How many Days do I get?") == ["many", "day", "get"]


def test_relevant_chunk_ranks_first():
    index = BM25Index()
    index.upsert_source("pto.md", [_chunk("pto.md", "Employees receive 25 vacation days.")])
    index.upsert_source("it.md", [_chunk("it.md", "Laptops lock after 5 minutes.")])
    hits = index.search("how many vacation days")
    assert hits[0].chunk.source == "pto.md"
    assert all(h.chunk.source != "it.md" for h in hits)


def test_query_with_no_known_terms_returns_nothing():
    index = BM25Index()
    index.upsert_source("a.md", [_chunk("a.md", "alpha beta")])
    assert index.search("zebra") == []
    assert index.search("the and of") == []


def test_upsert_replaces_previous_version_of_a_source():
    index = BM25Index()
    index.upsert_source("a.md", [_chunk("a.md", "old", 0), _chunk("a.md", "older", 1)])
    index.upsert_source("a.md", [_chunk("a.md", "new")])
    assert index.sources() == {"a.md": 1}
    assert index.search("old") == []


def test_remove_source():
    index = BM25Index()
    index.upsert_source("a.md", [_chunk("a.md", "alpha")])
    assert index.remove_source("a.md") == 1
    assert len(index) == 0
    assert index.remove_source("a.md") == 0


def test_save_and_load_round_trip(tmp_path):
    index = BM25Index()
    ingest_paths(index, [SAMPLE_DOCS])
    path = tmp_path / "nested" / "index.json"
    index.save(path)

    loaded = BM25Index.load(path)
    assert loaded.sources() == index.sources()
    query = "minimum password length"
    assert [h.chunk.id for h in loaded.search(query)] == [h.chunk.id for h in index.search(query)]


def test_load_missing_file_gives_empty_index(tmp_path):
    assert len(BM25Index.load(tmp_path / "missing.json")) == 0
