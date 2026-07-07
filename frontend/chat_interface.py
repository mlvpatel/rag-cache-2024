"""Chat interface for the RagFlowCache Streamlit app.

Shows the answer plus a trace of how it was produced: a cache hit returns a
past answer with no model call, a miss answers from the whole corpus preloaded
into context and stores the result so the next repeat is instant.
"""

import streamlit as st

from frontend import api_utils

_STEP_LABEL = {
    "cache_lookup": "Cache lookup",
    "load_corpus": "Load corpus",
    "grounded_answer": "Grounded answer",
    "cache_store": "Cache store",
}


def _render_trace(steps, cached, similarity) -> None:
    if not steps:
        return
    if cached:
        header = "Cache hit"
        if similarity is not None:
            header += f", similarity {similarity:.2f}"
        header += ", no model call"
    else:
        header = "Cache miss, answered from the preloaded corpus"
    with st.expander(header, expanded=False):
        path = ", ".join(
            _STEP_LABEL.get(step.get("step"), step.get("step", "")) for step in steps
        )
        st.caption(path)
        for index, step in enumerate(steps, 1):
            label = _STEP_LABEL.get(step.get("step"), step.get("step", ""))
            detail = {key: value for key, value in step.items() if key != "step"}
            st.markdown(f"**{index}. {label}**")
            if detail:
                st.json(detail, expanded=False)


def display_chat_interface() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("steps"):
                _render_trace(
                    message["steps"],
                    message.get("cached", False),
                    message.get("similarity"),
                )

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        model = st.session_state.get("model", "qwen2.5:7b-instruct")
        with st.spinner("Checking the cache, then the corpus..."):
            try:
                result = api_utils.chat(prompt, st.session_state.session_id, model)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return
        st.session_state.session_id = result.get("session_id")
        answer = result.get("answer", "")
        st.markdown(answer)
        _render_trace(
            result.get("steps", []),
            result.get("cached", False),
            result.get("similarity"),
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "steps": result.get("steps", []),
            "cached": result.get("cached", False),
            "similarity": result.get("similarity"),
        }
    )
