# ML/AI Study Assistant — A Retrieval-Augmented Generation (RAG) Chatbot

A chatbot that answers questions about core Machine Learning / AI concepts by
retrieving relevant excerpts from a local knowledge base and grounding its answers in
them — a minimal, fully working implementation of the RAG pattern used in real GenAI
applications.

**Author:** Daniel Osei

**Live demo:** _add your Streamlit Community Cloud URL here once deployed (see below)_

## What this project demonstrates

- **Retrieval-augmented generation (RAG)**: answers are grounded in retrieved source
  text rather than the model's unguided output, reducing hallucination.
- **A local vector search pipeline**: text chunking, embedding, and similarity search
  using [Chroma](https://www.trychroma.com/) as the vector store.
- **An LLM call** via [Groq](https://groq.com/)'s free-tier API, with the response
  constrained to the retrieved context.
- **Source attribution**: every answer shows which knowledge-base articles it drew on.
- **A deployed, public web app**, not just a script — see "Live demo" above.

## How it works

```
User question
     │
     ▼
Embed question (TF-IDF vectorizer, fit on the corpus)
     │
     ▼
Similarity search against Chroma vector store  →  top-k relevant chunks
     │
     ▼
Build a prompt: "Using only this context, answer the question"
     │
     ▼
Groq LLM (Llama 3.1) generates the answer
     │
     ▼
Answer + cited sources
```

## Repository contents

| File / folder | Description |
|---|---|
| `app.py` | Streamlit chat app — the deployable web interface |
| `rag_ml_chatbot.ipynb` | The same project as a notebook: ingestion, indexing, retrieval, and an interactive Q&A cell, for studying/demoing the pipeline step by step |
| `src/ingest.py` | Chunks the corpus, builds TF-IDF embeddings, and stores everything in a local Chroma vector database |
| `src/rag.py` | Core RAG logic: retrieval, prompt construction, and the Groq LLM call |
| `data/` | Study-note text files (the knowledge base), sourced from Wikipedia articles on core ML/AI topics |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for the required `GROQ_API_KEY` (local use) |

`app.py` (via `src/`) and the notebook implement the same pipeline — chunking, TF-IDF
embedding, Chroma storage, retrieval, and the Groq call — but the notebook has it
inlined cell by cell for studying/demoing the steps individually, while `app.py`
imports the shared logic from `src/` and is what's actually deployed as the live app.

## Running it locally

**1. Clone and install dependencies**

```bash
git clone <this-repo-url>
cd rag-ml-chatbot
pip install -r requirements.txt
```

**2. Get a free Groq API key**

Sign up (free, no credit card) at [console.groq.com/keys](https://console.groq.com/keys)
and create an API key.

**3. Add your key**

```bash
cp .env.example .env
# then edit .env and paste your key in place of the placeholder
```

`.env` is gitignored — your key never gets committed.

**4. Run the app**

```bash
streamlit run app.py
```

The first run automatically builds the vector index from `data/` (a few seconds) —
no separate step needed. Open the URL Streamlit prints (usually
`http://localhost:8501`) and start asking questions, e.g. *"What's the difference
between supervised and unsupervised learning?"*

(To explore the pipeline interactively instead, run `jupyter notebook
rag_ml_chatbot.ipynb`.)

## Deploying it for free on Streamlit Community Cloud

This makes the app reachable at a public URL like `yourapp.streamlit.app` that anyone
can open — no install required on their end.

1. Push this repo to your own GitHub account (already done if you're reading this
   from there).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick this repo, branch `main`, and main file `app.py`.
4. Before deploying, open **Advanced settings → Secrets** and add:
   ```
   GROQ_API_KEY = "your-real-key-here"
   ```
   This is entered directly on Streamlit's own site and stored securely there — it
   never goes into the repo or anywhere else.
5. Click **Deploy**. The first build installs dependencies and auto-builds the vector
   index on first run, then the app is live.
6. Copy the app's URL into the "Live demo" line at the top of this README so anyone
   viewing the repo can try it immediately.

Any time you push new commits to `main`, Streamlit Community Cloud redeploys
automatically.

## Design choices worth knowing for an interview

**Why TF-IDF embeddings instead of a neural embedding model?**
This project deliberately uses scikit-learn's `TfidfVectorizer` — a classic, fully
local sparse-embedding technique — instead of a sentence-transformers/Hugging Face
model. That keeps setup to a single `pip install` with no multi-hundred-MB model
download and no dependency on Hugging Face being reachable — important for a fast,
reliable Streamlit Cloud deployment. Retrieval quality is a notch below dense neural
embeddings on more diverse corpora, but for a focused, single-domain knowledge base
like this one it performs well.

**Upgrading retrieval quality.** For a corpus with more topic diversity, swapping in
dense embeddings is a small, contained change: replace the `TfidfVectorizer` step in
`src/ingest.py` and `src/rag.py` with a `sentence-transformers` model (e.g.
`all-MiniLM-L6-v2`) via Chroma's built-in `SentenceTransformerEmbeddingFunction`. The
rest of the pipeline (chunking, storage, retrieval, prompt construction) stays the
same — that's the point of separating retrieval from generation in a RAG system.

**Why Groq instead of OpenAI?** Groq offers a genuinely free tier with fast inference
on open models (Llama 3.1/3.3), which makes this project runnable and deployable by
anyone without a paid API key. Swapping to OpenAI, Anthropic, or another provider only
requires changing the client and model name in `src/rag.py`.

**Why auto-build the index on startup?** `app.py` calls `ensure_index()` (wrapped in
`st.cache_resource` so it only runs once per app instance) rather than requiring a
separate manual step. This matters for a hosted deployment: Streamlit Community Cloud
spins up a fresh container from the repo, and `chroma_db/` is intentionally gitignored
(it's derived data) — so the app needs to be able to build its own index on first
request rather than expecting a pre-built one to already exist.

## Attribution

The knowledge base in `data/` is adapted from Wikipedia articles, available under the
[Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).
Each file in `data/` cites its source article and URL in its header line.

## Known limitations

- The knowledge base covers 12 core ML/AI topics — questions outside that scope will
  correctly get a "not enough information" response rather than a fabricated answer.
- TF-IDF retrieval is keyword/n-gram based rather than semantic, so paraphrased
  questions that share little vocabulary with the source text may retrieve weaker
  matches than a dense-embedding model would.
- Streamlit Community Cloud's free tier sleeps an app after a period of inactivity;
  the first visitor after a while will see a short "waking up" delay.

