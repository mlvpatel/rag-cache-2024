"""Cache augmented generation engine for rag-cache-2024.

Two ideas. First, the whole small corpus is preloaded into the prompt, so the
model answers from documents already in context, with no retrieval step.
Second, a semantic cache of past questions returns a stored answer instantly
when a new question is close enough, with no model call at all.

Keyless on Ollama. The trace reports whether the answer came from the cache or
from the model reading the preloaded corpus.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.cag import store
from src.core.config import settings
from src.core.langchain_utils import _make_llm
from src.embeddings.vectorstore_utils import load_document_text

logger = logging.getLogger(__name__)

_QA_SYSTEM = (
    "You answer questions for rag-cache-2024 using only the documents below, which "
    "are preloaded in full. If the documents do not contain the answer, say you "
    "do not have that information rather than inventing one.\n\n"
    "Documents:\n{corpus}"
)


def run_cag(model: str, question: str, chat_history=None) -> dict:
    """Answer a question from the semantic cache, or from the preloaded corpus."""
    steps: List[Dict[str, Any]] = []

    # 1. Semantic cache: a close enough past question returns instantly.
    cached_answer, similarity = store.cache_lookup(
        question, settings.cag_similarity_threshold
    )
    sim = round(similarity, 3)
    if cached_answer is not None:
        steps.append({"step": "cache_lookup", "hit": True, "similarity": sim})
        return {
            "answer": cached_answer,
            "cached": True,
            "similarity": sim,
            "steps": steps,
        }
    steps.append({"step": "cache_lookup", "hit": False, "similarity": sim})

    # 2. Preload the whole corpus into context, no retrieval.
    corpus, used = store.corpus_text(settings.cag_max_context_chars)
    steps.append({"step": "load_corpus", "documents": used, "chars": len(corpus)})

    # 3. Generate grounded in the preloaded corpus.
    llm = _make_llm(model, temperature=0)
    system = _QA_SYSTEM.format(corpus=corpus or "None.")
    answer = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=question)]
    ).content
    steps.append({"step": "grounded_answer"})

    # 4. Store the answer so a repeat question is served from the cache.
    store.cache_store(question, answer)
    steps.append({"step": "cache_store"})

    return {"answer": answer, "cached": False, "similarity": sim, "steps": steps}


def index_corpus_document(file_path: str, file_id: int, filename: str) -> bool:
    """Read a document's full text and add it to the preloaded corpus."""
    try:
        content = load_document_text(file_path)
    except Exception as exc:
        logger.error("CAG could not read %s: %s", file_path, exc)
        return False
    store.insert_document(file_id, filename, content)
    logger.info(
        "CAG stored corpus document file_id=%s (%d chars)", file_id, len(content)
    )
    return True
