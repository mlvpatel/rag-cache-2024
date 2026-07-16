"""Unit tests for the CAG engine and semantic cache logic (all IO mocked)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import src.cag.engine as engine
import src.cag.store as store


def test_cache_hit_at_threshold():
    """similarity = 1 - distance; a distance of 0.15 with threshold 0.85 hits."""
    fake = MagicMock()
    fake.similarity_search_with_score.return_value = [
        (Document(page_content="q", metadata={"answer": "cached!"}), 0.15)
    ]
    with patch.object(store, "get_cache_store", return_value=fake):
        answer, sim = store.cache_lookup("question", 0.85)
    assert answer == "cached!"
    assert round(sim, 2) == 0.85


def test_cache_miss_below_threshold():
    fake = MagicMock()
    fake.similarity_search_with_score.return_value = [
        (Document(page_content="q", metadata={"answer": "cached!"}), 0.30)
    ]
    with patch.object(store, "get_cache_store", return_value=fake):
        answer, sim = store.cache_lookup("question", 0.85)
    assert answer is None
    assert round(sim, 2) == 0.70


def test_cache_clear_drops_collection_and_singleton():
    fake = MagicMock()
    with patch.object(store, "get_cache_store", return_value=fake):
        store.cache_clear()
    fake.delete_collection.assert_called_once()
    assert store._cache_store is None


def test_run_cag_keys_the_cache_on_the_standalone_question(monkeypatch):
    """A follow-up must be reformulated BEFORE the cache lookup, and the
    rewrite is what gets cached; raw follow-ups would poison the cache."""
    fake_llm = FakeListChatModel(
        responses=["what is the Nimbus Pro plan price?", "42 dollars"]
    )
    monkeypatch.setattr(engine, "_make_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(engine.store, "corpus_text", lambda n: ("# doc\ntext", 1))
    lookups, stored = [], []
    monkeypatch.setattr(
        engine.store, "cache_lookup", lambda q, t: lookups.append(q) or (None, 0.0)
    )
    monkeypatch.setattr(engine.store, "cache_store", lambda q, a: stored.append((q, a)))

    history = [{"role": "human", "content": "tell me about Nimbus Pro"}]
    result = engine.run_cag("qwen2.5:7b-instruct", "and its price?", history)

    assert lookups == ["what is the Nimbus Pro plan price?"]
    assert stored == [("what is the Nimbus Pro plan price?", "42 dollars")]
    assert result["cached"] is False
    assert any(s["step"] == "reformulate" for s in result["steps"])


def test_run_cag_serves_from_cache_without_model_call(monkeypatch):
    monkeypatch.setattr(
        engine, "_make_llm", lambda *a, **k: FakeListChatModel(responses=["unused"])
    )
    monkeypatch.setattr(
        engine.store, "cache_lookup", lambda q, t: ("cached answer", 0.93)
    )
    called = []
    monkeypatch.setattr(
        engine.store, "corpus_text", lambda n: called.append(n) or ("", 0)
    )

    result = engine.run_cag("qwen2.5:7b-instruct", "repeat question", None)

    assert result["cached"] is True
    assert result["answer"] == "cached answer"
    assert not called, "a cache hit must not load the corpus"


def test_corpus_text_greedy_packing_respects_cap(monkeypatch):
    rows = [
        {"filename": "a.txt", "content": "x" * 50},
        {"filename": "b.txt", "content": "y" * 50},
        {"filename": "c.txt", "content": "z" * 5000},
    ]
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = lambda s, *a: None
    pool = MagicMock()
    pool.connection.return_value.__enter__ = lambda s: conn
    pool.connection.return_value.__exit__ = lambda s, *a: None
    with patch.object(store, "_pool", return_value=pool):
        text, used = store.corpus_text(200)
    assert used == 2, "the oversized third document must not enter the prompt"
    assert len(text) <= 200
