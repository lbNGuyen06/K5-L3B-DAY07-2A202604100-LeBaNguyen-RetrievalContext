"""Benchmark script: compare Dense / Lexical / Hybrid retrieval on a split.

Usage (from project root):
    python starter/benchmark_retrieval.py \
        --chunks  starter/data/converted/chunks.json \
        --questions starter/data/converted/questions.json \
        --split dev \
        --strategy B \
        --output starter/results/benchmark-dev

Output files in <output>/:
    traces_dense.json    -- per-question trace, dense retrieval
    traces_lexical.json  -- per-question trace, lexical (TF-IDF)
    traces_hybrid.json   -- per-question trace, hybrid (RRF)
    summary.json         -- comparison table: Hit@1, Hit@3, Context Hit Rate

Gold answers are never passed to the retriever or query; only used to compute
hit metrics after retrieval.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Inject starter/ vào sys.path để import baseline, retrievers, config
# ---------------------------------------------------------------------------
_STARTER = Path(__file__).resolve().parent
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))

from baseline import TfidfIndex, pack_context, validate   # noqa: E402
from config import load_settings                           # noqa: E402
from retrievers import DenseRetriever, HybridRetriever, LexicalRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit(chunk: dict, gold: list[dict]) -> bool:
    """True nếu chunk chứa ít nhất một evidence từ gold list."""
    return any(
        chunk["source_id"] == g["source_id"]
        and str(chunk["page"]) == str(g["page"])
        and g["evidence"] in chunk["text"]
        for g in gold
    )


def _run_one(retriever, question: str, gold: list[dict], top_k: int, budget: int) -> dict:
    """Chạy một câu hỏi qua retriever, trả về trace dict."""
    t0 = time.perf_counter()
    results = retriever.search(question, filters=None, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    context, context_ids = pack_context(results, budget=budget)
    marked = [{**c, "hit": _hit(c, gold)} for c in results]

    return {
        "question": question,
        "top_k": top_k,
        "budget_chars": budget,
        "results": marked,
        "context": context,
        "context_ids": context_ids,
        "context_chars": len(context),
        "elapsed_ms": round(elapsed_ms, 2),
        "hit1": bool(marked and marked[0]["hit"]),
        "hit3": any(c["hit"] for c in marked[:3]),
        "context_hit": any(c["hit"] and c["id"] in context_ids for c in marked),
    }


def _evaluate(name: str, retriever, questions: list[dict], top_k: int, budget: int) -> list[dict]:
    """Chạy toàn bộ dev split, in tiến trình, trả về danh sách trace."""
    traces = []
    n = len(questions)
    for i, q in enumerate(questions, 1):
        print(f"  [{name}] {i}/{n}: {q['question'][:60]}", flush=True)
        trace = _run_one(retriever, q["question"], q["gold"], top_k=top_k, budget=budget)
        traces.append({"query_id": q["id"], "split": q["split"], **trace})
    return traces


def _summarise(name: str, traces: list[dict]) -> dict:
    answerable = [t for t in traces]   # tất cả câu đều tính retrieval metric
    n = len(answerable)
    if n == 0:
        return {"strategy": name, "n": 0}
    return {
        "strategy": name,
        "n": n,
        "hit1": sum(t["hit1"] for t in answerable),
        "hit1_rate": round(sum(t["hit1"] for t in answerable) / n, 4),
        "hit3": sum(t["hit3"] for t in answerable),
        "hit3_rate": round(sum(t["hit3"] for t in answerable) / n, 4),
        "context_hit": sum(t["context_hit"] for t in answerable),
        "context_hit_rate": round(sum(t["context_hit"] for t in answerable) / n, 4),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in answerable), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in answerable), 2),
    }


def _print_table(summaries: list[dict]) -> None:
    """In bảng so sánh ra stdout."""
    header = f"{'Strategy':<10} {'N':>4} {'Hit@1':>7} {'Hit@1%':>8} {'Hit@3':>7} {'Hit@3%':>8} {'CtxHit%':>9} {'Median ms':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['strategy']:<10} {s['n']:>4} "
            f"{s['hit1']:>7} {s['hit1_rate']:>8.1%} "
            f"{s['hit3']:>7} {s['hit3_rate']:>8.1%} "
            f"{s['context_hit_rate']:>9.1%} "
            f"{s['median_ms']:>10.1f}"
        )
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Dense / Lexical / Hybrid retrieval on a split."
    )
    parser.add_argument("--chunks", required=True, type=Path,
                        help="JSON array of chunks (Day 7 schema)")
    parser.add_argument("--questions", required=True, type=Path,
                        help="JSON array of questions with gold evidence")
    parser.add_argument("--split", required=True, choices=("dev", "test"),
                        help="Only benchmark this split")
    parser.add_argument("--strategy", default="B", choices=("A", "B", "C"),
                        help="Chunking strategy stored in pgvector (default: B)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Max chunks per search call (default: 3)")
    parser.add_argument("--budget", type=int, default=800,
                        help="Context budget in characters (default: 800)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory; existing files will be overwritten")
    args = parser.parse_args()

    # -- Load data -----------------------------------------------------------
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions_all = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [q for q in questions_all if q.get("split") == args.split]
    if not questions:
        raise ValueError(f"No questions found for split '{args.split}'")

    # Validate schema (baseline.validate kiểm tra cả gold nằm trong chunk)
    validate(chunks, questions)
    print(f"Loaded {len(chunks)} chunks, {len(questions)} questions (split={args.split})")

    # -- Load settings (DATABASE_URL, GEMINI_API_KEY, model) ----------------
    cfg = load_settings()

    # -- Init 3 retrievers ---------------------------------------------------
    print("\nInit DenseRetriever (loading embedding model, connecting pgvector)...")
    t_dense_init = time.perf_counter()
    dense = DenseRetriever(
        database_url=cfg.database_url,
        strategy=args.strategy,
        model_name=cfg.embedding_model,
    )
    print(f"  -> DenseRetriever ready ({time.perf_counter() - t_dense_init:.1f}s)")

    print("Init LexicalRetriever (building TF-IDF index)...")
    t_lex_init = time.perf_counter()
    lexical = LexicalRetriever(chunks)
    print(f"  -> LexicalRetriever ready ({time.perf_counter() - t_lex_init:.1f}s)")

    print("Init HybridRetriever (RRF over Dense + Lexical)...")
    hybrid = HybridRetriever(dense, lexical)
    print("  -> HybridRetriever ready")

    # -- Chạy benchmark ------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    top_k = args.top_k
    budget = args.budget
    summaries = []

    for name, retriever in [
        ("dense", dense),
        ("lexical", lexical),
        ("hybrid", hybrid),
    ]:
        print(f"\n--- {name.upper()} (top_k={top_k}, budget={budget}) ---")
        t0 = time.perf_counter()
        traces = _evaluate(name, retriever, questions, top_k=top_k, budget=budget)
        elapsed = time.perf_counter() - t0

        # Lưu trace
        trace_path = args.output / f"traces_{name}.json"
        trace_path.write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  Traces saved -> {trace_path}  ({elapsed:.1f}s total)")

        s = _summarise(name, traces)
        summaries.append(s)

    # -- Lưu summary + in bảng -----------------------------------------------
    summary = {
        "split": args.split,
        "strategy": args.strategy,
        "top_k": top_k,
        "budget_chars": budget,
        "n_questions": len(questions),
        "results": summaries,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary saved -> {summary_path}")
    _print_table(summaries)


if __name__ == "__main__":
    main()
