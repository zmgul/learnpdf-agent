# research-agent

> A ReAct Document Q&A agent over PDFs.

Ingests a PDF into a vector store and answers natural-language questions through
the Claude API, **always citing its sources** (page number + passage).

Stack: Python 3.11 · FastAPI · Pydantic v2 · Claude API · ChromaDB ·
sentence-transformers · PyMuPDF · Poetry

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
curl -F "file=@thesis.pdf" http://127.0.0.1:8000/ingest

# Ask a question
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the aim of the study?"}'
```

## Project Structure

```
research-agent/
├── src/
│   ├── schemas.py          # Pydantic v2 request/response models
│   ├── ingest.py           # PDF → manual chunking → embed → ChromaDB
│   ├── retriever.py        # ChromaDB query tool (top_k=3)
│   ├── prompts.py          # System prompt + Claude tool-use definition
│   └── agent.py            # ReAct loop over the Claude API
├── api/
│   └── main.py             # FastAPI endpoints (/ingest, /ask, /)
├── static/
│   └── index.html          # Plain HTML + fetch() UI
├── tests/
│   └── test_agent.py       # Schema, prompt, chunking, and mocked-agent tests
├── data/                   # PDFs are placed here — not tracked
├── pyproject.toml          # Poetry dependencies
└── README.md
```

The ChromaDB store is persisted under `./chroma_db/`; both `data/` and
`chroma_db/` are git-ignored.

## Test

```bash
poetry run pytest
```

Tests mock the Claude API call — no real key or network access required.
