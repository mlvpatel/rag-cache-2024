"""End to end cache augmented generation test through Ollama.

Proves the full pipeline works with local models and no paid key: a document is
loaded into the corpus, the first question is answered from it, and an identical
second question is served from the semantic cache with no model call. Skipped
automatically when Ollama is not running.
"""

import urllib.request

import psycopg
import pytest

import src.cag.engine as engine
import src.cag.store as store
import src.core.config as config_mod
import src.embeddings.vectorstore_utils as vs

FILE_ID = 987654


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _cleanup():
    store.delete_document(FILE_ID)
    with psycopg.connect(config_mod.settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM langchain_pg_embedding e "
                "USING langchain_pg_collection c "
                "WHERE e.collection_id = c.uuid AND c.name = %s",
                (store.CACHE_COLLECTION,),
            )


@pytest.mark.skipif(not _ollama_running(), reason="ollama server not running")
def test_end_to_end_cache_hit_on_repeat(pg_available, monkeypatch):
    monkeypatch.setattr(config_mod.settings, "embedding_provider", "ollama")
    monkeypatch.setattr(vs, "_query_embeddings", None)
    monkeypatch.setattr(store, "_cache_store", None)

    store.create_corpus_table()
    _cleanup()
    store.insert_document(FILE_ID, "facts.txt", "The capital of France is Paris.")

    try:
        first = engine.run_cag("llama3.2:3b", "What is the capital of France?")
        second = engine.run_cag("llama3.2:3b", "What is the capital of France?")
    finally:
        _cleanup()

    assert first["cached"] is False
    assert "paris" in first["answer"].lower()
    assert second["cached"] is True
