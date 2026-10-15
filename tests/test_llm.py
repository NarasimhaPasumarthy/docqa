from types import SimpleNamespace

from docqa.chunking import Chunk
from docqa.llm import NO_CONTEXT_ANSWER, ClaudeAnswerer

CHUNKS = [
    Chunk("handbook.md#0", "handbook.md", 0, "Employees receive 25 vacation days per year."),
    Chunk("handbook.md#1", "handbook.md", 1, "Up to 5 unused days carry over."),
]


def _text(text, *cites):
    citations = [SimpleNamespace(document_index=i, cited_text=q) for i, q in cites]
    return SimpleNamespace(type="text", text=text, citations=citations or None)


class FakeClient:
    """Stands in for anthropic.Anthropic(): records the request, returns a canned response."""

    def __init__(self, response):
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._response


def test_request_sends_chunks_as_citable_documents():
    request = ClaudeAnswerer("claude-opus-5-5").build_request("How much PTO?", CHUNKS)
    content = request["messages"][0]["content"]
    assert [b["type"] for b in content] == ["document", "document", "text"]
    assert content[0]["citations"] == {"enabled": True}
    assert content[0]["source"]["data"] == CHUNKS[0].text
    assert content[-1]["text"] == "How much PTO?"
    assert request["fallbacks"] == "default"


def test_citations_become_numbered_markers():
    response = SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            _text("You get 25 days", (0, "Employees receive 25 vacation days per year.")),
            _text(" and can carry over 5", (1, "Up to 5 unused days carry over.")),
            _text("."),
        ],
    )
    answer = ClaudeAnswerer("m", client=FakeClient(response)).answer("PTO?", CHUNKS)
    assert answer.text == "You get 25 days [1] and can carry over 5 [2]."
    assert [c.chunk_id for c in answer.citations] == ["handbook.md#0", "handbook.md#1"]
    assert answer.model == "claude-opus-5-5"


def test_repeated_quote_reuses_its_number():
    quote = (0, "Employees receive 25 vacation days per year.")
    response = SimpleNamespace(
        model="m", stop_reason="end_turn", content=[_text("A", quote), _text(" B", quote)]
    )
    answer = ClaudeAnswerer("m", client=FakeClient(response)).answer("q", CHUNKS)
    assert answer.text == "A [1] B [1]"
    assert len(answer.citations) == 1


def test_no_chunks_skips_the_model_call():
    client = FakeClient(None)
    answer = ClaudeAnswerer("m", client=client).answer("anything", [])
    assert answer.text == NO_CONTEXT_ANSWER
    assert client.requests == []


def test_refusal_is_reported():
    response = SimpleNamespace(model="m", stop_reason="refusal", content=[])
    answer = ClaudeAnswerer("m", client=FakeClient(response)).answer("q", CHUNKS)
    assert answer.refused
    assert answer.citations == []
