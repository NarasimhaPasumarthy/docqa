"""Generate grounded answers with Claude, using the API's native citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from .chunking import Chunk

SYSTEM_PROMPT = """\
You answer employees' questions about their company's internal documents.
Answer only from the documents in the user's message, and cite the passages that support
each claim. If the documents don't contain the answer, say so in one sentence and don't
guess. Keep answers short and direct."""

NO_CONTEXT_ANSWER = "I couldn't find anything about that in the indexed documents."
REFUSAL_ANSWER = "The model declined to answer this question."

# Server-side fallback: if a request is declined by a safety classifier, the API retries it
# on a suitable fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source: str
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    model: str | None = None
    refused: bool = False


class MissingCredentialsError(RuntimeError):
    pass


class Answerer(Protocol):
    def answer(self, question: str, chunks: list[Chunk]) -> Answer: ...


class ClaudeAnswerer:
    def __init__(self, model: str, effort: str = "medium", client: Any = None) -> None:
        self.model = model
        self.effort = effort
        self._client = client

    @property
    def client(self) -> Any:
        # Created lazily so the server starts (and the retrieval eval runs) without a key.
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def build_request(self, question: str, chunks: list[Chunk]) -> dict[str, Any]:
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": c.text},
                "title": c.source,
                "context": f"Excerpt {c.id}",
                "citations": {"enabled": True},
            }
            for c in chunks
        ]
        return {
            "model": self.model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
            "messages": [
                {"role": "user", "content": [*documents, {"type": "text", "text": question}]}
            ],
        }

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        if not chunks:
            return Answer(NO_CONTEXT_ANSWER)
        try:
            response = self.client.beta.messages.create(**self.build_request(question, chunks))
        except TypeError as e:
            # The SDK reports "no credentials configured" as a TypeError, not an API error.
            if "authentication method" in str(e):
                raise MissingCredentialsError(
                    "No Claude credentials found; set ANTHROPIC_API_KEY"
                ) from e
            raise
        if response.stop_reason == "refusal":
            return Answer(REFUSAL_ANSWER, model=response.model, refused=True)
        return parse_response(response, chunks)


def parse_response(response: Any, chunks: list[Chunk]) -> Answer:
    """Join the text blocks and turn each citation into a numbered [n] marker."""
    parts: list[str] = []
    citations: list[Citation] = []
    numbers: dict[tuple[str, str], int] = {}
    for block in response.content:
        if block.type != "text":  # skip thinking and fallback blocks
            continue
        parts.append(block.text)
        for cit in getattr(block, "citations", None) or []:
            chunk = chunks[cit.document_index]
            quote = cit.cited_text.strip()
            key = (chunk.id, quote)
            if key not in numbers:
                citations.append(Citation(chunk.id, chunk.source, quote))
                numbers[key] = len(citations)
            parts.append(f" [{numbers[key]}]")
    return Answer("".join(parts).strip(), citations, model=response.model)
