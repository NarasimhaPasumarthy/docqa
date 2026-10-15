"""Measure retrieval and answer quality against a labelled question set.

Retrieval metrics need no API key, so they run in CI as a quality gate. Answer metrics
call Claude and are run on demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .llm import Answerer
from .retriever import BM25Index


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_source: str | None  # None: the documents don't answer it; expect abstention
    expected_keywords: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[EvalCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalCase(**json.loads(line)) for line in lines if line.strip()]


@dataclass
class RetrievalReport:
    k: int
    ranks: list[tuple[str, int | None]]  # (question, 1-based rank of first correct chunk)

    @property
    def hit_rate(self) -> float:
        return sum(r is not None for _, r in self.ranks) / len(self.ranks)

    @property
    def mrr(self) -> float:
        return sum(1 / r for _, r in self.ranks if r) / len(self.ranks)


def evaluate_retrieval(index: BM25Index, cases: list[EvalCase], k: int) -> RetrievalReport:
    ranks = []
    for case in cases:
        if case.expected_source is None:
            continue
        hits = index.search(case.question, k=k)
        rank = next(
            (i for i, h in enumerate(hits, 1) if h.chunk.source == case.expected_source), None
        )
        ranks.append((case.question, rank))
    return RetrievalReport(k, ranks)


@dataclass
class AnswerResult:
    question: str
    passed: bool
    reason: str
    answer: str


def evaluate_answers(
    index: BM25Index, answerer: Answerer, cases: list[EvalCase], k: int
) -> list[AnswerResult]:
    """Answerable cases pass when every keyword appears and the expected source is cited.
    Unanswerable cases pass when the answer cites nothing (the model abstained)."""
    results = []
    for case in cases:
        hits = index.search(case.question, k=k)
        answer = answerer.answer(case.question, [h.chunk for h in hits])
        cited = {c.source for c in answer.citations}
        if case.expected_source is None:
            passed, reason = not cited, "abstained" if not cited else f"cited {sorted(cited)}"
        else:
            missing = [kw for kw in case.expected_keywords if kw.lower() not in answer.text.lower()]
            if missing:
                passed, reason = False, f"missing {missing}"
            elif case.expected_source not in cited:
                passed, reason = False, f"did not cite {case.expected_source}"
            else:
                passed, reason = True, "ok"
        results.append(AnswerResult(case.question, passed, reason, answer.text))
    return results
