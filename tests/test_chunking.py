import pytest

from docqa.chunking import chunk_text


def test_short_document_is_one_chunk():
    chunks = chunk_text("# Title\n\nOne paragraph.", "a.md")
    assert len(chunks) == 1
    assert chunks[0].id == "a.md#0"
    assert chunks[0].text == "# Title\n\nOne paragraph."


def test_chunks_respect_max_chars():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(30))
    chunks = chunk_text(text, "big.md", max_chars=500, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_short_heading_is_carried_into_next_chunk():
    heading = "## Remote work"
    body = "x" * 450
    chunks = chunk_text(f"{'y' * 450}\n\n{heading}\n\n{body}", "h.md", max_chars=500)
    assert chunks[1].text.startswith(heading)


def test_long_paragraph_is_split_on_sentences():
    para = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(para, "long.md", max_chars=300, overlap_chars=50)
    assert all(len(c.text) <= 300 for c in chunks)
    assert "".join(c.text for c in chunks).count("Sentence number") == 100


def test_blank_document_has_no_chunks():
    assert chunk_text("  \n\n  ", "empty.md") == []


def test_overlap_must_be_smaller_than_max():
    with pytest.raises(ValueError):
        chunk_text("text", "a.md", max_chars=100, overlap_chars=100)
