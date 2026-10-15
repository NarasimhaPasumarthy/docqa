"""Command-line interface: docqa ingest | ask | eval | serve"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .evaluate import evaluate_answers, evaluate_retrieval, load_cases
from .ingest import ingest_paths
from .llm import ClaudeAnswerer
from .retriever import BM25Index


def _ingest(args: argparse.Namespace, settings: Settings) -> int:
    index = BM25Index.load(settings.index_path)
    for name, n in ingest_paths(index, args.paths).items():
        print(f"indexed {name}: {n} chunks")
    index.save(settings.index_path)
    print(f"{len(index)} chunks in {settings.index_path}")
    return 0


def _ask(args: argparse.Namespace, settings: Settings) -> int:
    index = BM25Index.load(settings.index_path)
    hits = index.search(args.question, k=settings.top_k)
    answer = ClaudeAnswerer(settings.model, settings.effort).answer(
        args.question, [h.chunk for h in hits]
    )
    print(answer.text)
    for i, c in enumerate(answer.citations, 1):
        print(f'\n[{i}] {c.source}: "{c.quote}"')
    return 0


def _eval(args: argparse.Namespace, settings: Settings) -> int:
    index = BM25Index()
    ingest_paths(index, [args.docs])
    cases = load_cases(args.cases)

    report = evaluate_retrieval(index, cases, args.k)
    print(f"## Retrieval ({len(report.ranks)} answerable questions, k={args.k})\n")
    print("| metric | value |\n|---|---|")
    print(f"| hit rate@{args.k} | {report.hit_rate:.2f} |\n| MRR | {report.mrr:.2f} |")
    for question, rank in report.ranks:
        if rank is None:
            print(f"  MISS: {question}")
    ok = report.hit_rate >= args.min_hit_rate

    if args.answers:
        answerer = ClaudeAnswerer(settings.model, settings.effort)
        results = evaluate_answers(index, answerer, cases, args.k)
        passed = sum(r.passed for r in results)
        print(f"\n## Answers ({passed}/{len(results)} passed)\n")
        print("| | question | result |\n|---|---|---|")
        for r in results:
            print(f"| {'PASS' if r.passed else 'FAIL'} | {r.question} | {r.reason} |")

    if not ok:
        print(f"\nhit rate {report.hit_rate:.2f} is below --min-hit-rate {args.min_hit_rate}")
    return 0 if ok else 1


def _serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    uvicorn.run("docqa.api:create_app", factory=True, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docqa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index files or directories")
    p.add_argument("paths", nargs="+", type=Path)
    p.set_defaults(func=_ingest)

    p = sub.add_parser("ask", help="ask a question from the command line")
    p.add_argument("question")
    p.set_defaults(func=_ask)

    p = sub.add_parser("eval", help="measure retrieval (and optionally answer) quality")
    p.add_argument("--docs", type=Path, default=Path("sample_docs"))
    p.add_argument("--cases", type=Path, default=Path("eval/cases.jsonl"))
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--min-hit-rate", type=float, default=0.0)
    p.add_argument("--answers", action="store_true", help="also grade Claude's answers")
    p.set_defaults(func=_eval)

    p = sub.add_parser("serve", help="run the API and web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    return args.func(args, Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())
