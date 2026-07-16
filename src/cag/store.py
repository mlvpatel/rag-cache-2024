"""Storage for rag-cache-2024: the document corpus and the semantic answer cache.

The corpus is the full text of every document, stored in one table and
preloaded into the prompt at answer time, because cache augmented generation
does not retrieve fragments. The cache is a small pgvector index of past
questions: a new question that is close enough to a cached one returns the
stored answer without calling the model at all.

The connection pool is shared with src.api.db_utils and imported lazily inside
each function, so importing this module opens no database connection.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_COLLECTION = "rag_cache_cache"

_cache_store = None


def _pool():
    from src.api.db_utils import _get_pool

    return _get_pool()


def create_corpus_table() -> None:
    """Create the corpus table if missing."""
    with _pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cag_documents (
                    id         BIGSERIAL PRIMARY KEY,
                    file_id    BIGINT,
                    filename   TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)


def insert_document(file_id: int, filename: str, content: str) -> None:
    """Add one document's full text to the corpus."""
    with _pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cag_documents (file_id, filename, content) "
                "VALUES (%s, %s, %s)",
                (file_id, filename, content),
            )


def delete_document(file_id: int) -> bool:
    """Remove a document from the corpus by id."""
    try:
        with _pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM cag_documents WHERE file_id = %s", (file_id,))
        return True
    except Exception as exc:
        logger.error("Failed to delete corpus document %s: %s", file_id, exc)
        return False


def corpus_text(max_chars: int) -> Tuple[str, int]:
    """Return the concatenated corpus (capped) and how many documents were used."""
    with _pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, content FROM cag_documents ORDER BY id")
            rows = cur.fetchall()
    parts = []
    total = 0
    used = 0
    for row in rows:
        block = f"# {row['filename']}\n{row['content']}"
        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)
        used += 1
    return "\n\n".join(parts)[:max_chars], used


def get_cache_store():
    """The pgvector collection that indexes past questions by embedding."""
    global _cache_store
    if _cache_store is None:
        from langchain_postgres import PGVector

        from src.embeddings.vectorstore_utils import (
            _sqlalchemy_url,
            get_query_embeddings,
        )

        _cache_store = PGVector(
            embeddings=get_query_embeddings(),
            collection_name=CACHE_COLLECTION,
            connection=_sqlalchemy_url(),
            use_jsonb=True,
        )
    return _cache_store


def cache_lookup(question: str, threshold: float) -> Tuple[Optional[str], float]:
    """Return (answer, similarity) for the nearest cached question.

    The cache hits when the nearest past question's cosine similarity is at or
    above the threshold; otherwise the answer is None and the best similarity is
    returned for observability.
    """
    try:
        hits = get_cache_store().similarity_search_with_score(question, k=1)
    except Exception as exc:
        logger.warning("Cache lookup skipped: %s", exc)
        return None, 0.0
    if not hits:
        return None, 0.0
    doc, distance = hits[0]
    similarity = 1.0 - float(distance)
    if similarity >= threshold:
        return doc.metadata.get("answer", ""), similarity
    return None, similarity


def cache_clear() -> None:
    """Drop every cached answer.

    Called whenever the corpus changes: a cached answer was generated against
    the old corpus, and serving it after an upload or delete would be answering
    from documents that no longer say that. Full invalidation is deliberately
    simple; the cache refills on the next misses.
    """
    global _cache_store
    try:
        get_cache_store().delete_collection()
    except Exception as exc:
        logger.warning("Cache clear skipped: %s", exc)
    finally:
        _cache_store = None


def cache_store(question: str, answer: str) -> None:
    """Remember a question and its answer for future cache hits."""
    from langchain_core.documents import Document

    try:
        get_cache_store().add_documents(
            [Document(page_content=question, metadata={"answer": answer})]
        )
    except Exception as exc:
        logger.warning("Cache store skipped: %s", exc)
