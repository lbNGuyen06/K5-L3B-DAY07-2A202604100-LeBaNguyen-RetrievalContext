"""Run the end-to-end agent loop (GeminiController + DenseRetriever) on a split.

Usage:
    python starter/run_agent_eval.py \
        --cases starter/data/converted/cases.json \
        --split dev \
        --top-k 3 \
        --budget 2000 \
        --max-searches 3 \
        --output starter/results/agent-dev
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add starter/ to sys.path
_STARTER = Path(__file__).resolve().parent
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))

from agentic import run_agent              # noqa: E402
from config import load_settings           # noqa: E402
from controller import GeminiController   # noqa: E402
from retrievers import DenseRetriever, HybridRetriever, LexicalRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real agent loop on benchmark cases.")
    parser.add_argument("--cases", required=True, type=Path, help="Path to cases.json")
    parser.add_argument("--split", required=True, choices=("dev", "test"), help="Split to evaluate")
    parser.add_argument("--strategy", default="B", choices=("A", "B", "C"), help="Chunk strategy in pgvector")
    parser.add_argument("--top-k", type=int, default=3, help="Max chunks per search (default: 3)")
    parser.add_argument("--budget", type=int, default=2000, help="Context budget in chars (default: 2000)")
    parser.add_argument("--max-searches", type=int, default=3, help="Max search turns (default: 3)")
    parser.add_argument("--retriever", default="dense", choices=("dense", "lexical", "hybrid"),
                        help="Retrieval method used by the agent (default: dense)")
    parser.add_argument("--chunks", type=Path, default=_STARTER / "data" / "converted" / "chunks.json",
                        help="Chunks JSON, only needed for lexical/hybrid")
    parser.add_argument("--no-repeat", action="store_true",
                        help="Tell the controller not to repeat earlier queries (experiment)")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    # Load cases
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c.get("split") == args.split]
    if not cases:
        raise ValueError(f"No cases found for split '{args.split}'")

    print(f"Loaded {len(cases)} cases for split '{args.split}' "
          f"({sum(c['answerable'] for c in cases)} answerable, "
          f"{sum(not c['answerable'] for c in cases)} unanswerable).")

    # Load settings
    cfg = load_settings()

    print("\nInitializing GeminiController...")
    controller = GeminiController(api_key=cfg.gemini_api_key, no_repeat=args.no_repeat)

    print(f"Initializing {args.retriever} retriever...")
    if args.retriever == "lexical":
        retriever = LexicalRetriever(json.loads(args.chunks.read_text(encoding="utf-8")))
    else:
        dense = DenseRetriever(
            database_url=cfg.database_url,
            strategy=args.strategy,
            model_name=cfg.embedding_model,
        )
        retriever = dense
        if args.retriever == "hybrid":
            lexical = LexicalRetriever(json.loads(args.chunks.read_text(encoding="utf-8")))
            retriever = HybridRetriever(dense, lexical)

    args.output.mkdir(parents=True, exist_ok=True)
    traces = []

    print(f"\nRunning agent loop on {len(cases)} cases (top_k={args.top_k}, budget={args.budget})...\n")

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        q = case["question"]
        ans_type = "answerable" if case["answerable"] else "unanswerable"
        print(f"[{i}/{len(cases)}] Case {cid} ({ans_type}): {q[:60]}...")

        t0 = time.perf_counter()
        try:
            result = run_agent(
                question=q,
                controller=controller,
                retriever=retriever,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            status = result["status"]
            answer = result["answer"]
            citations = result["citations"]
            events = result["events"]
            err_msg = None
            print(f"       -> Status: {status} | Citations: {citations} | Searches: {sum(e.get('event') == 'search' for e in events)} | Time: {elapsed_ms:.0f}ms")
            safe_ans = answer[:80].encode('ascii', errors='replace').decode('ascii')
            if answer:
                print(f"       -> Answer: {safe_ans}...")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            status = "error"
            answer = ""
            citations = []
            events = []
            err_msg = str(e)
            print(f"       -> ERROR: {e}")

        trace_entry = {
            "query_id": cid,
            "split": case["split"],
            "question": q,
            "answerable": case["answerable"],
            "status": status,
            "answer": answer,
            "citations": citations,
            "events": events,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        if err_msg:
            trace_entry["error"] = err_msg
        traces.append(trace_entry)

        # Small delay between cases to stay comfortably within rate limits
        time.sleep(3.5)

    # Save traces
    traces_path = args.output / "traces.json"
    traces_path.write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nTraces saved -> {traces_path}")
    print("Grading is manual: fill reviews.json by hand from traces.json (RULES/BENCHMARK).")


if __name__ == "__main__":
    main()
