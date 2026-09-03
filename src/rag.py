"""
Core RAG logic shared by the Streamlit app and the notebook: loads the
persisted vector index, retrieves relevant chunks for a question, builds a
grounded prompt, and calls Groq's free-tier API to generate an answer.
"""

import os
import pickle
from pathlib import Path

import chromadb
from groq import Groq

from ingest import CHROMA_DIR, COLLECTION_NAME, VECTORIZER_PATH, build_index

# Groq's free tier includes several Llama models. 8B is fast and plenty
# capable for this use case; swap for "llama-3.3-70b-versatile" for higher
# quality answers at the cost of a bit more latency.
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a helpful study-assistant chatbot that answers questions about \
machine learning and AI concepts using ONLY the provided context excerpts. \
Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so clearly \
instead of guessing.
- Keep answers clear and well-organized, as if explaining to a Masters student.
- When helpful, mention which topic(s) the information came from.
"""


class RAGNotIndexedError(RuntimeError):
    """Raised when the vector index hasn't been built yet and couldn't be built automatically."""


def ensure_index():
    """Builds the vector index if it doesn't exist yet. Safe to call on every
    app startup -- it's a no-op once chroma_db/ has been populated."""
    if not VECTORIZER_PATH.exists():
        build_index()


def _load_index():
    if not VECTORIZER_PATH.exists():
        raise RAGNotIndexedError(
            "No index found and it could not be built automatically. "
            "Run `python src/ingest.py` to build the vector database."
        )
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return vectorizer, collection


def retrieve(question: str, n_results: int = 4):
    """Return the top-n most relevant chunks for a question, each as a dict
    with 'text', 'source', and 'distance' (lower = more relevant)."""
    vectorizer, collection = _load_index()
    query_embedding = vectorizer.transform([question]).toarray().tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "source": meta["source"], "distance": dist})
    return chunks


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[Excerpt {i+1} - source: {c['source']}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context excerpts above."
    )


def answer_question(question: str, n_results: int = 4, api_key: str | None = None):
    """Runs the full RAG pipeline: retrieve -> build prompt -> call Groq.

    Returns (answer_text, chunks_used). Raises RuntimeError if no Groq API
    key is available -- the caller is expected to surface that to the user
    rather than silently failing.
    """
    chunks = retrieve(question, n_results=n_results)

    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Groq API key found. Set the GROQ_API_KEY environment variable "
            "(see .env.example) -- get a free key at https://console.groq.com/keys"
        )

    client = Groq(api_key=api_key)
    prompt = build_prompt(question, chunks)

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    answer = completion.choices[0].message.content
    return answer, chunks

