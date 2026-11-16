"""Export Day 6 strategy B chunks and benchmark questions to Day 7 format."""
import json
import re
import sys
from pathlib import Path

import psycopg

from config import load_settings  # reads DATABASE_URL from the root .env (not committed)

# Paths (relative to this repo; the Day 6 repo is expected as a sibling folder)
ROOT = Path(__file__).resolve().parents[1]
CONVERTED_DIR = ROOT / "starter" / "data" / "converted"
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"

DAY6_ROOT = ROOT.parent / "K5-L3B-Day06-LeBaNguyen-2A202604100-DataIngestionEngineering"
DAY6_Q_PATH = DAY6_ROOT / "benchmark" / "questions.jsonl"

# 1. Fetch Strategy B Chunks from Postgres
print("1. Fetching Strategy B chunks from Postgres...")
conn = psycopg.connect(load_settings().database_url)
with conn.cursor() as cur:
    cur.execute("""
        SELECT chunk_id, document_id, source, page, text, metadata
        FROM chunks WHERE strategy = 'B' ORDER BY chunk_id
    """)
    chunk_rows = cur.fetchall()
conn.close()

chunks = [
    {
        "id": r[0],
        "source_id": r[1],
        "source": r[2],
        "page": r[3],
        "text": r[4],
        "metadata": r[5] if isinstance(r[5], dict) else json.loads(r[5]),
    }
    for r in chunk_rows
]
print(f"   -> Fetched {len(chunks)} chunks.")

# Index chunks by page for fast alignment
chunks_by_page = {}
for c in chunks:
    chunks_by_page.setdefault(c["page"], []).append(c)


# Helper: find exact substring matching evidence in page chunks
def find_exact_quote(page: int, quote: str, doc_id: str = "wbg-ar-2026"):
    tokens = quote.strip().split()
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    regex = re.compile(pattern, re.IGNORECASE)
    candidates = chunks_by_page.get(page, [])
    for c in candidates:
        if c["source_id"] == doc_id:
            m = regex.search(c["text"])
            if m:
                return c["id"], m.group(0)
    return None, None


# 2. Process Answerable Questions
print("2. Processing 22 Day 6 benchmark questions with exact evidence alignment...")
raw_questions = [
    json.loads(line)
    for line in DAY6_Q_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

questions = []
cases = []

for q in raw_questions:
    cid, exact_text = find_exact_quote(q["gold_page"], q["evidence_quote"])
    if not exact_text:
        raise ValueError(f"Could not find exact text for {q['id']} on page {q['gold_page']}")

    gold = [
        {
            "source_id": q["gold_source_id"],
            "page": q["gold_page"],
            "evidence": exact_text,
        }
    ]

    for alt in q.get("alt_gold", []):
        alt_cid, alt_exact = find_exact_quote(alt["page"], alt["evidence_quote"])
        if alt_exact:
            gold.append(
                {
                    "source_id": alt["source_id"],
                    "page": alt["page"],
                    "evidence": alt_exact,
                }
            )

    q_dict = {
        "id": q["id"],
        "split": q["split"],
        "question": q["question"],
        "gold": gold,
        "expected_answer": q["expected_answer"],
    }
    if "filter" in q:
        q_dict["filter"] = q["filter"]
    questions.append(q_dict)

    cases.append(
        {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
        }
    )

print(f"   -> Processed {len(questions)} answerable questions.")

# 3. Add Unanswerable Cases (2 dev, 2 test)
print("3. Adding 4 unanswerable cases (2 dev, 2 test)...")
unanswerable = [
    {
        "id": "u01",
        "split": "dev",
        "question": "What was the total compensation and salary paid to the World Bank Group President in fiscal 2026?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The Annual Report 2026 does not publish individual executive compensation or salary figures.",
    },
    {
        "id": "u02",
        "split": "dev",
        "question": "Which commercial airline was chosen as the exclusive corporate travel partner for World Bank Group operations in fiscal 2026?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The Annual Report 2026 does not disclose corporate travel vendors or airline procurement contracts.",
    },
    {
        "id": "u03",
        "split": "test",
        "question": "What was the total financial allocation approved by the World Bank Group specifically for sovereign cryptocurrency regulation in Kenya in fiscal 2026?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The report discusses digital economy and financial inclusion generally, but contains no project allocation for sovereign cryptocurrency regulation in Kenya.",
    },
    {
        "id": "u04",
        "split": "test",
        "question": "How many electric vehicles were added to the World Bank Group headquarters fleet in Washington D.C. in fiscal 2026?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Internal headquarters fleet statistics and vehicle procurement details are not reported in the 2026 Annual Report.",
    },
]
cases.extend(unanswerable)
print(f"   -> Total cases: {len(cases)} ({sum(c['split']=='dev' for c in cases)} dev, {sum(c['split']=='test' for c in cases)} test).")

# 4. Save to files
print("4. Saving files...")
for out_dir in (CONVERTED_DIR, DATA_DIR):
    (out_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
print(f"   Saved to {CONVERTED_DIR} and {DATA_DIR}.")

# 5. Validate with baseline.validate()
print("5. Validating schema with baseline.validate()...")
sys.path.insert(0, str(ROOT / "starter"))
from baseline import validate

dev_qs = [q for q in questions if q["split"] == "dev"]
test_qs = [q for q in questions if q["split"] == "test"]
validate(chunks, dev_qs)
print(f"   Dev validation: PASSED ({len(dev_qs)} questions)")
validate(chunks, test_qs)
print(f"   Test validation: PASSED ({len(test_qs)} questions)")

print("\nAll data preparation steps COMPLETED successfully!")
