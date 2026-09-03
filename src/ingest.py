"""
Builds the vector index for the RAG chatbot.

Reads the study-note text files in data/, splits them into overlapping
chunks, embeds the chunks with a local TF-IDF vectorizer, and stores
everything in a persistent Chroma vector database.

Why TF-IDF instead of a neural embedding model?
This project deliberately uses scikit-learn's TfidfVectorizer -- a
classic, fully local sparse-embedding technique -- instead of a
sentence-transformers/Hugging Face model. That means `pip install` is all
you need; there's no multi-hundred-MB model download and no dependency on
Hugging Face being reachable. Retrieval quality is a notch below dense
neural embeddings, but for a focused, single-domain corpus like this one
it works well. See README.md's "Design choices" section for the upgrade
path to dense embeddings.

Run with: python src/ingest.py
(app.py also calls build_index() automatically on first run if no index
exists yet, so this rarely needs to be run by hand.)
"""

import pickle
import re
from pathlib import Path

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
VECTORIZER_PATH = CHROMA_DIR / "tfidf_vectorizer.pkl"
COLLECTION_NAME = "ml_ai_knowledge_base"

CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 200  # trailing fragments shorter than this get merged into the previous chunk


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            # A short trailing fragment carries almost no TF-IDF signal (it can
            # even end up as an all-zero vector) and pollutes retrieval, so
            # merge it into the previous chunk instead of indexing it alone.
            if chunks and len(chunk) < MIN_CHUNK_CHARS:
                chunks[-1]["text"] = (chunks[-1]["text"] + " " + chunk).strip()
            else:
                chunks.append({"text": chunk, "source": source})
        start += chunk_size - overlap
    return chunks


def load_corpus():
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {DATA_DIR}.")

    all_chunks = []
    for path in txt_files:
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        first_line = cleaned.splitlines()[0] if cleaned else path.stem
        source_label = first_line.replace("Source: ", "") if first_line.startswith("Source:") else path.stem
        chunks = chunk_text(cleaned, source=source_label)
        all_chunks.extend(chunks)
        print(f"  {path.name}: {len(cleaned):,} chars -> {len(chunks)} chunks")
    return all_chunks


def build_index():
    CHROMA_DIR.mkdir(exist_ok=True)

    print(f"Loading corpus from {DATA_DIR} ...")
    all_chunks = load_corpus()
    print(f"Total chunks: {len(all_chunks)}")

    documents = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"]} for c in all_chunks]

    print("Fitting TF-IDF vectorizer on the corpus (fully local, no downloads)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    embeddings_arr = vectorizer.fit_transform(documents).toarray()

    # Safety net: drop any degenerate all-zero-vector chunk -- see MIN_CHUNK_CHARS note above.
    norms = (embeddings_arr ** 2).sum(axis=1) ** 0.5
    keep = norms > 0
    if (~keep).sum():
        print(f"  Dropping {(~keep).sum()} degenerate zero-vector chunk(s)")
    documents = [d for d, k in zip(documents, keep) if k]
    metadatas = [m for m, k in zip(metadatas, keep) if k]
    ids = [f"chunk_{i}" for i in range(len(documents))]
    embeddings = embeddings_arr[keep].tolist()

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
        )
        print(f"  Indexed {min(i + batch_size, len(documents))}/{len(documents)} chunks")

    print(f"Done. Vector database saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()

