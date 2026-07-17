# learnpdf-agent

> A document Q&A learning tool for any PDF.

Upload any PDF — a book, lecture notes, a document — and ask questions about it.
It ingests the PDF into a vector store and answers natural-language questions
through the Claude API, **always citing its sources** (page number + passage).

Stack: Python 3.11 · FastAPI · Pydantic v2 · Claude API · ChromaDB ·
sentence-transformers · PyMuPDF · Poetry

## How it works

A fixed **RAG** pipeline — there is no agentic "should I search?" decision:

1. **Ingest** — the PDF is split into overlapping, word-based chunks (manual, no
   LangChain), embedded with a multilingual model, and stored in ChromaDB.
   Re-ingesting the same PDF (matched by name + size) skips re-embedding.
2. **Ask** — the question is embedded, the top-k passages are retrieved
   automatically, then those passages + the question are sent to Claude in a
   single call. The answer always cites the source pages.

## Setup

```bash
poetry install
cp .env.example .env        # set ANTHROPIC_API_KEY=... in .env
```

## Run

```bash
poetry run uvicorn api.main:app --reload
```

Open <http://127.0.0.1:8000>: upload a PDF, ask a question.

## API

```bash
# Ingest a PDF
curl -F "file=@book.pdf" http://127.0.0.1:8000/ingest

# Ask a question
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the aim of the study?"}'
```

## Project Structure

```
learnpdf-agent/
├── src/
│   ├── schemas.py          # Pydantic v2 request/response models
│   ├── ingest.py           # PDF → manual chunking → embed → ChromaDB (+ cache)
│   ├── retriever.py        # ChromaDB similarity query (top_k=3)
│   ├── prompts.py          # System prompt (RAG) + formatting helpers
│   └── agent.py            # RAG flow: auto-retrieve → single Claude call
├── api/
│   └── main.py             # FastAPI endpoints (/ingest, /ask, /)
├── static/
│   └── index.html          # Plain HTML + fetch() UI
├── tests/
│   └── test_agent.py       # Schema, prompt, chunking, and mocked-agent tests
├── data/                   # PDFs are placed here — not tracked
├── analysis.md             # System analysis (purpose, stack, invariants)
├── state.md                # Session checkpoint (progress, next action)
├── .claude/CLAUDE.md       # Agent contract (rules, architecture, dev order)
├── pyproject.toml          # Poetry dependencies
└── README.md
```

The ChromaDB store is persisted under `./chroma_db/`; both `data/` and
`chroma_db/` are git-ignored.

## LLM-friendly by design

Three plain-text files keep an AI coding agent (e.g. Claude Code) on-task and
context-aware across sessions:

- **`.claude/CLAUDE.md`** — the contract: role, hard rules (allowed dirs, no
  hardcoded keys, no LangChain/Jinja2), the RAG architecture, and the build order.
- **`analysis.md`** — the system analysis: purpose, stack, directory map, and
  invariants — the "why" that the code alone can't convey.
- **`state.md`** — a running checkpoint: what changed, open decisions, and the
  next action — so a fresh session resumes without re-discovering context.

## Test

```bash
poetry run pytest
```

Tests mock the Claude API call — no real key or network access required.
