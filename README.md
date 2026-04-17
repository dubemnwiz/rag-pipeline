# RAG Pipeline

A Python Retrieval-Augmented Generation (RAG) API backed by **ChromaDB** and **Claude 3.5 Sonnet**, featuring a self-auditing verification loop where Claude checks whether the retrieved context is actually sufficient before answering.

---

## What makes this different from basic RAG?

Standard RAG is: retrieve → stuff context into prompt → hope for the best.

This pipeline adds verification within a loop:

```
User question
     │
     ▼
Query ChromaDB  ◄──────────────────────────────┐
     │                                          │
     ▼                                          │
Claude: "Is this context sufficient?"           │
     │                                          │
     ├── confidence_score: float (0–1)          │
     ├── needs_more_data: bool                  │
     └── new_search_term: str | None            │
              │                                 │
              ├── needs_more_data=True  ────────┘  (re-query with new term)
              │
              └── needs_more_data=False
                       │
                       ▼
              Claude: generate final answer
                       │
                       ▼
              { answer, confidence_score, sources, iterations }
```

Claude audits its own retrieved context and can re-query ChromaDB with a rephrased or narrowed search term — up to `MAX_VERIFICATION_LOOPS` times — before committing to a final answer.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first install downloads `all-MiniLM-L6-v2` (~22 MB) from HuggingFace.
> Subsequent runs use the local cache.

### 3. Configure secrets

```bash
cp .env.example .env
```

Open `.env` and set your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys).

### 4. Add documents

Drop `.txt` or `.md` files into the `data/` folder. A sample document (`data/sample.txt`) is included so you can test immediately.

### 5. Ingest documents

```bash
python ingest.py
```

This chunks each document, embeds the chunks using `all-MiniLM-L6-v2`, and stores everything in a local ChromaDB database (`.chroma/` folder). You only need to re-run this when you add or change documents.

```bash
python ingest.py --clear          # wipe and re-embed everything
python ingest.py --chunk-size 600 # override chunk size
python ingest.py --help           # see all options
```

### 6. Start the server

```bash
uvicorn app.main:app --reload
```

The API is now live at `http://localhost:8000`.

---

## Using the API

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "claude-3-5-sonnet-20241022",
  "chunk_count": 9
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the limitations of RAG?"}'
```

```json
{
  "answer": "Based on the retrieved context, RAG has several limitations...",
  "confidence_score": 0.92,
  "sources": ["sample.txt"],
  "iterations": 1
}
```

**`iterations: 1`** — the first retrieval was sufficient.
**`iterations: 2`** — Claude re-queried once with a different search term.
**`iterations: 3`** — hit the loop cap; Claude answered with the best context it found.

### Interactive API docs

Open `http://localhost:8000/docs` in your browser for a full Swagger UI where you can explore and test both endpoints.

---

### Four Pydantic models define all data flowing through the system:

| Model | Used by | Purpose |
|---|---|---|
| `RetrievalAssessment` | `rag_chain.py` | Claude's structured self-audit output |
| `AskRequest` | `main.py` | Validates the incoming POST body |
| `AskResponse` | `main.py` | Shapes the JSON response |
| `RetrievedContext` | `embedder.py` | Wraps a single ChromaDB chunk |

`RetrievalAssessment` has a cross-field validation rule: if `needs_more_data=True`, then `new_search_term` must also be provided. Pydantic enforces this automatically via `@model_validator`.

---

## Configuration reference

All settings live in `.env`:

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Your Claude API key |
| `CLAUDE_MODEL` | `claude-3-5-sonnet-20241022` | Claude model for both assessment and generation |
| `CHROMA_PERSIST_DIR` | `.chroma` | Where ChromaDB saves vectors to disk |
| `CHROMA_COLLECTION` | `documents` | Collection name inside ChromaDB |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `MAX_VERIFICATION_LOOPS` | `3` | Maximum re-query iterations |

---
