"""
Streamlit chat UI for the ML/AI RAG chatbot.

Run locally with: streamlit run app.py
Deploy for free on Streamlit Community Cloud (share.streamlit.io) -- see
README.md for step-by-step deployment instructions.

Requires GROQ_API_KEY, either as:
  - a local .env file (see .env.example), for running on your own machine, or
  - a "Secret" set in the Streamlit Community Cloud app dashboard, for the
    deployed version (never commit real keys to the repo).
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rag import RAGNotIndexedError, answer_question, ensure_index  # noqa: E402

# Load a local .env file if python-dotenv is installed and one exists -- keeps
# the API key out of shell history / code while still being easy to set
# locally. Has no effect on Streamlit Community Cloud (it uses its own
# Secrets mechanism instead, exposed as normal environment variables).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# On Streamlit Community Cloud, secrets are set in the app dashboard and
# show up in st.secrets rather than the shell environment -- mirror it into
# os.environ so the rest of the code only has to check one place. Locally
# (no secrets.toml file at all) st.secrets raises rather than behaving like
# an empty dict, so this has to be guarded.
if not os.environ.get("GROQ_API_KEY"):
    try:
        groq_secret = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_secret = None
    if groq_secret:
        os.environ["GROQ_API_KEY"] = groq_secret

st.set_page_config(page_title="ML/AI Study Assistant", page_icon="🤖", layout="centered")

st.title("🤖 ML/AI Study Assistant")
st.caption(
    "A retrieval-augmented chatbot answering questions from a small curated knowledge "
    "base of core Machine Learning / AI concepts (sourced from Wikipedia, CC BY-SA)."
)


@st.cache_resource(show_spinner="Building the knowledge base index (first run only)...")
def _ensure_index_cached():
    ensure_index()
    return True


with st.sidebar:
    st.header("About this project")
    st.markdown(
        "This is a Retrieval-Augmented Generation (RAG) demo:\n\n"
        "1. Your question is embedded and matched against a local vector "
        "database of ML/AI study notes.\n"
        "2. The most relevant excerpts are retrieved.\n"
        "3. An LLM (via the free Groq API) answers **using only those "
        "excerpts** -- reducing hallucination compared to asking the model "
        "cold.\n\n"
        "Try asking about supervised vs. unsupervised learning, "
        "backpropagation, transformers, overfitting, or RAG itself.\n\n"
        "[View the source on GitHub](https://github.com/osei-daniel-plateau/rag-ml-chatbot)"
    )
    st.divider()
    n_results = st.slider("Chunks retrieved per question", min_value=2, max_value=6, value=4)
    show_sources = st.checkbox("Show retrieved sources", value=True)

_ensure_index_cached()

if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "**GROQ_API_KEY is not set.** Get a free key at "
        "[console.groq.com/keys](https://console.groq.com/keys), then add it to a `.env` "
        "file locally (see `.env.example`) or, on Streamlit Community Cloud, as a Secret "
        "in the app's settings.",
        icon="⚠️",
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if question := st.chat_input("Ask about a machine learning / AI concept..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context and generating answer..."):
                answer, chunks = answer_question(question, n_results=n_results)
            st.markdown(answer)
            sources = sorted({c["source"] for c in chunks})
            if show_sources:
                with st.expander("Sources used"):
                    for s in sources:
                        st.markdown(f"- {s}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except RAGNotIndexedError as e:
            st.error(str(e))
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong calling the LLM: {e}")
