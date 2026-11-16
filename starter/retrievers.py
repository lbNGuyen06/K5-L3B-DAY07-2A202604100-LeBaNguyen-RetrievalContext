"""Three Retriever implementations for the Day 7 agentic loop.

Each class exposes:
    search(query, *, filters, top_k) -> list[dict]

where every dict contains: id, text, source_id, source, page, metadata, score.
This contract matches the Retriever Protocol in agentic.py.

Day 6 source code is imported via sys.path rather than copied, so both labs
stay in sync when Day 6 code is updated.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Inject Day 6 src/ into the import path so we can reuse its embeddings and
# store without duplicating code.  Adjust the sibling-folder name if needed.
# ---------------------------------------------------------------------------
_DAY6_STARTER = (
    Path(__file__).resolve().parents[2]
    / "K5-L3B-Day06-LeBaNguyen-2A202604100-DataIngestionEngineering"
    / "starter"
)
if str(_DAY6_STARTER) not in sys.path:
    sys.path.insert(0, str(_DAY6_STARTER))

# ---------------------------------------------------------------------------
# TfidfIndex lives in baseline.py inside the same starter/ directory.
# ---------------------------------------------------------------------------
_STARTER_DIR = Path(__file__).resolve().parent
if str(_STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(_STARTER_DIR))


# =============================================================================
# 4a. DenseRetriever – pgvector cosine search via Day 6 PgVectorStore
# =============================================================================

class DenseRetriever:
    """Semantic retrieval: embeds the query with SentenceTransformerEmbedder,
    then queries pgvector using PgVectorStore.search from Day 6.

    Args:
        database_url: PostgreSQL connection string (from .env DATABASE_URL).
        strategy: Chunking strategy stored in the DB, e.g. 'A' or 'B'.
        model_name: SentenceTransformer model; must match the indexed model.
        table: pgvector table name (default: 'chunks').
    """

    def __init__(
        self,
        database_url: str,
        strategy: str = "B",
        model_name: str = "intfloat/multilingual-e5-small",
        table: str = "chunks",
    ) -> None:
        # Import lazily so callers that only use LexicalRetriever don't need
        # sentence-transformers or psycopg installed.
        from src.embeddings import SentenceTransformerEmbedder  # Day 6 src/embeddings.py
        from src.store import PgVectorStore                     # Day 6 src/store.py

        self.strategy = strategy
        self._embedder = SentenceTransformerEmbedder(model_name)
        self._store = PgVectorStore(
            database_url=database_url,
            dimensions=self._embedder.dimensions,
            table=table,
        )

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Embed *query* and retrieve the top-k most similar chunks.

        Returns dicts with keys: id, text, source_id, source, page,
        metadata, score (cosine similarity, 0-1).
        """
        query_vector = self._embedder.embed_query(query)
        raw = self._store.search(
            self.strategy,
            query_vector,
            top_k=top_k,
            filters=filters,
        )
        # Rename fields to match the Day 7 / agentic.py contract.
        return [
            {
                "id": r["chunk_id"],           # chunk_id → id
                "text": r["text"],
                "source_id": r["document_id"], # document_id → source_id
                "source": r["source"],
                "page": r["page"],
                "metadata": r["metadata"],
                "score": r["score"],
            }
            for r in raw
        ]


# =============================================================================
# 4b. LexicalRetriever – TF-IDF keyword search (wraps baseline.TfidfIndex)
# =============================================================================

class LexicalRetriever:
    """Keyword-based retrieval using the TF-IDF index from baseline.py.

    Args:
        chunks: list of chunk dicts (Day 7 schema: id, text, source_id,
                source, page, metadata).  Pass the same list used for
                baseline evaluation.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        from baseline import TfidfIndex  # same starter/ directory

        self._index = TfidfIndex(chunks)

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,  # noqa: ARG002 – TF-IDF has no filter support
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return top-k chunks by TF-IDF cosine similarity.

        Note: *filters* is accepted for interface compatibility but is not
        applied; TfidfIndex has no metadata-filter support.
        """
        raw = self._index.search(query, top_k=top_k)
        # TfidfIndex already returns dicts with id, text, source_id, source,
        # page, metadata, score (the Day 7 schema used in baseline.py).
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "source_id": r["source_id"],
                "source": r["source"],
                "page": r["page"],
                "metadata": r["metadata"],
                "score": r["score"],
            }
            for r in raw
        ]


# =============================================================================
# 4c. HybridRetriever – Reciprocal Rank Fusion of dense + lexical results
# =============================================================================

class HybridRetriever:
    """Combines DenseRetriever and LexicalRetriever via Reciprocal Rank Fusion.

    RRF score for chunk c:  sum_over_lists( 1 / (60 + rank_in_list(c)) )
    Score is normalisation-free, robust to scale differences between the two
    retrieval methods, and guaranteed to be unique per chunk in the output.

    Args:
        dense: a DenseRetriever instance.
        lexical: a LexicalRetriever instance.
        rrf_k: the RRF constant (default 60, standard from the original paper).
        fetch_factor: how many extra candidates to fetch from each sub-retriever
            before merging (fetch_factor * top_k per list).  Higher values give
            RRF more candidates to rank from.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        lexical: LexicalRetriever,
        rrf_k: int = 60,
        fetch_factor: int = 3,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._rrf_k = rrf_k
        self._fetch_factor = fetch_factor

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Merge dense and lexical results with RRF and return top-k chunks.

        Each chunk appears at most once in the output (deduplicated by id).
        """
        fetch_k = max(top_k * self._fetch_factor, top_k)

        dense_results = self._dense.search(query, filters=filters, top_k=fetch_k)
        lexical_results = self._lexical.search(query, filters=filters, top_k=fetch_k)

        # Accumulate RRF scores; also keep the full chunk dict for the winner.
        rrf_scores: dict[str, float] = {}
        chunk_store: dict[str, dict[str, Any]] = {}

        for ranked_list in (dense_results, lexical_results):
            for rank_0, chunk in enumerate(ranked_list):
                cid = chunk["id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank_0 + 1)
                # Prefer the dense chunk dict (higher quality score) when both
                # lists contain the same chunk.
                if cid not in chunk_store:
                    chunk_store[cid] = chunk

        # Sort by descending RRF score, then by id for determinism.
        ranked = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

        return [
            {**chunk_store[cid], "score": round(score, 6)}
            for cid, score in ranked
        ]
