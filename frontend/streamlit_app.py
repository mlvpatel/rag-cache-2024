"""RagFlowCache Streamlit application entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from frontend.chat_interface import display_chat_interface  # noqa: E402
from frontend.sidebar import display_sidebar  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="RagFlowCache", layout="wide")
    st.title("RagFlowCache")
    st.caption(
        "Cache augmented RAG, 2025. The whole corpus is preloaded into context, "
        "and repeated questions are served instantly from a semantic cache."
    )
    display_sidebar()
    display_chat_interface()


if __name__ == "__main__":
    main()
