"""Temel testler.

Hafif testler (schemas, prompts) her zaman çalışır; ağır bağımlılık gerektiren
testler (ingest/retriever/agent) bağımlılık yoksa atlanır. Claude API çağrısı
mock'lanır — gerçek bir ANTHROPIC_API_KEY veya ağ erişimi gerektirmez.
"""

from __future__ import annotations

import pytest

from src.schemas import AskResponse, SourceChunk
from src import prompts


# ---------------------------------------------------------------------------
# Schemas (Pydantic v2)
# ---------------------------------------------------------------------------


def test_source_chunk_page_must_be_positive():
    with pytest.raises(Exception):
        SourceChunk(page=0, passage="x")


def test_ask_response_requires_at_least_one_source():
    with pytest.raises(Exception):
        AskResponse(answer="yanıt", sources=[])


def test_ask_response_valid():
    resp = AskResponse(
        answer="yanıt",
        sources=[SourceChunk(page=3, passage="ilgili pasaj")],
    )
    assert resp.iterations == 1
    assert resp.sources[0].page == 3


# ---------------------------------------------------------------------------
# Prompts biçimlendirme
# ---------------------------------------------------------------------------


def test_format_chunks_empty():
    out = prompts.format_chunks_for_tool_result([])
    assert "bulunamadı" in out.lower()


def test_format_chunks_includes_page():
    chunks = [SourceChunk(page=7, passage="bir metin")]
    out = prompts.format_chunks_for_tool_result(chunks)
    assert "sayfa 7" in out
    assert "bir metin" in out


def test_format_sources_truncates_long_passage():
    long_passage = "a" * 500
    out = prompts.format_sources_for_answer([SourceChunk(page=1, passage=long_passage)])
    assert "…" in out
    assert "sayfa 1" in out


def test_retrieve_tool_schema_shape():
    assert prompts.RETRIEVE_TOOL["name"] == "retrieve"
    assert "query" in prompts.RETRIEVE_TOOL["input_schema"]["required"]


# ---------------------------------------------------------------------------
# Chunking (ingest) — ağır bağımlılık varsa
# ---------------------------------------------------------------------------


def test_chunk_page_size_and_overlap():
    ingest = pytest.importorskip("src.ingest")
    words = " ".join(f"w{i}" for i in range(120))
    chunks = ingest.chunk_page(page=2, text=words, chunk_size=50, overlap=10)
    assert chunks, "en az bir chunk üretilmeli"
    assert all(c.page == 2 for c in chunks)
    # İlk chunk en fazla chunk_size kelime içerir.
    assert len(chunks[0].text.split()) <= 50


def test_chunk_page_empty_text():
    ingest = pytest.importorskip("src.ingest")
    assert ingest.chunk_page(page=1, text="   ", chunk_size=50, overlap=10) == []


def test_chunk_page_overlap_must_be_smaller():
    ingest = pytest.importorskip("src.ingest")
    with pytest.raises(ValueError):
        ingest.chunk_page(page=1, text="a b c", chunk_size=10, overlap=10)


# ---------------------------------------------------------------------------
# Agent ReAct döngüsü — Claude API mock'lanmış
# ---------------------------------------------------------------------------


class _Block:
    """Sahte içerik bloğu (tool_use veya text)."""

    def __init__(self, type, text=None, id=None, input=None):
        self.type = type
        self.text = text
        self.id = id
        self.input = input


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_agent_ask_collects_sources(monkeypatch):
    agent = pytest.importorskip("src.agent")

    # retrieve'i sabit bir kaynak döndürecek şekilde değiştir.
    def fake_retrieve(query, collection="default", top_k=3):
        return [SourceChunk(page=4, passage="ilgili pasaj", chunk_id="p4-c0")]

    monkeypatch.setattr(agent, "retrieve", fake_retrieve)

    # İki adımlı yanıt: önce tool_use, sonra end_turn (nihai metin).
    responses = [
        _Resp("tool_use", [_Block("tool_use", id="t1", input={"query": "soru"})]),
        _Resp("end_turn", [_Block("text", text="Yanıt: 4. sayfaya göre…")]),
    ]
    monkeypatch.setattr(agent, "_client", lambda: _FakeClient(responses))

    result = agent.ask("PDF neyle ilgili?")
    assert result.answer.startswith("Yanıt")
    assert len(result.sources) == 1
    assert result.sources[0].page == 4
    assert result.iterations == 2


def test_agent_ask_no_sources_still_returns_source(monkeypatch):
    agent = pytest.importorskip("src.agent")

    monkeypatch.setattr(agent, "retrieve", lambda *a, **k: [])
    responses = [_Resp("end_turn", [_Block("text", text="Kaynak yok.")])]
    monkeypatch.setattr(agent, "_client", lambda: _FakeClient(responses))

    result = agent.ask("alakasız soru")
    # Kaynak gösterimi zorunlu: boşken bile en az bir SourceChunk dönmeli.
    assert len(result.sources) >= 1
