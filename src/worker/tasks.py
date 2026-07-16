"""Background task: asynchronously add a document to the cache corpus."""

import logging

from src.api.db_utils import delete_document_record, insert_document_record
from src.cag.engine import index_corpus_document
from src.cag.store import cache_clear
from src.worker.celery_app import celery_app

logger = logging.getLogger("rag_cache")


@celery_app.task(name="process_document")
def process_document(file_path: str, filename: str) -> dict:
    """Add a document to the preloaded corpus.

    Ordering matters: the database record is inserted first so we get a real
    integer file_id, which tags the corpus entry. If reading the document
    fails, the record is rolled back so we never list a document with no
    content behind it.
    """
    file_id = insert_document_record(filename)
    stored = index_corpus_document(file_path, file_id, filename)
    if not stored:
        delete_document_record(file_id)
        logger.error(
            "Corpus load failed for %s, rolled back record %s", filename, file_id
        )
        return {"status": "failed", "file_id": file_id}
    # The corpus changed, so every cached answer is stale by definition.
    cache_clear()
    return {"status": "completed", "file_id": file_id}
