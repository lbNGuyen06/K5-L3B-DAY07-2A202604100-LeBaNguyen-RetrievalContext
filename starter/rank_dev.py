"""Rank of the first gold chunk per dev question for dense / lexical / hybrid (top-20).

Run from the repo root: python starter/rank_dev.py
"""
import json, sys
from pathlib import Path
_STARTER = Path(__file__).resolve().parent
sys.path.insert(0, str(_STARTER))
from config import load_settings
from retrievers import DenseRetriever, LexicalRetriever, HybridRetriever
root = _STARTER / "data" / "converted"
chunks = json.load(open(root / "chunks.json", encoding="utf-8"))
qs = [q for q in json.load(open(root / "questions.json", encoding="utf-8")) if q["split"] == "dev"]
cfg = load_settings()
dense = DenseRetriever(cfg.database_url, "B", cfg.embedding_model)
lex = LexicalRetriever(chunks)
hyb = HybridRetriever(dense, lex)
def hit(c, gold):
    return any(c["source_id"] == g["source_id"] and str(c["page"]) == str(g["page"]) and g["evidence"] in c["text"] for g in gold)
def rank(r, q, k=20):
    res = r.search(q["question"], filters=None, top_k=k)
    for i, c in enumerate(res, 1):
        if hit(c, q["gold"]): return i
    return None
print("rank of first gold chunk (k=20; None = not in top-20)")
print("%-5s %-6s %-8s %-7s %s" % ("id", "dense", "lexical", "hybrid", "question"))
for q in qs:
    d, l, h = rank(dense, q), rank(lex, q), rank(hyb, q)
    flag = "  <-- miss@3" if (d is None or d > 3) else ""
    print("%-5s %-6s %-8s %-7s %s%s" % (q["id"], d, l, h, q["question"][:55], flag))
