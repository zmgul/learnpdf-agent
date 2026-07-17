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


def test_ask_response_allows_empty_sources():
    """İlgili içerik yoksa kaynak uydurulmaz; boş liste geçerlidir."""
    resp = AskResponse(answer="ilgili pasaj bulunamadı", sources=[])
    assert resp.sources == []


def test_ask_response_valid():
    resp = AskResponse(
        answer="yanıt",
        sources=[SourceChunk(page=3, passage="ilgili pasaj")],
    )
    assert resp.answer == "yanıt"
    assert resp.sources[0].page == 3


# ---------------------------------------------------------------------------
# Prompts biçimlendirme
# ---------------------------------------------------------------------------


def test_format_chunks_empty():
    out = prompts.format_chunks_for_context([])
    assert "bulunamadı" in out.lower()


def test_format_chunks_includes_page():
    chunks = [SourceChunk(page=7, passage="bir metin")]
    out = prompts.format_chunks_for_context(chunks)
    assert "sayfa 7" in out
    assert "bir metin" in out


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


def test_chunk_ids_do_not_collide_across_documents():
    """Aynı koleksiyona iki farklı PDF alındığında chunk_id'ler çakışmamalı."""
    ingest = pytest.importorskip("src.ingest")
    pages = [(1, "a b c"), (2, "d e f")]

    key_a = ingest._doc_key("kitap.pdf:1000")
    key_b = ingest._doc_key("ders-notu.pdf:2000")
    ids_a = {c.chunk_id for c in ingest.build_chunks(pages, doc_key=key_a)}
    ids_b = {c.chunk_id for c in ingest.build_chunks(pages, doc_key=key_b)}

    assert ids_a and ids_b
    assert not (ids_a & ids_b), "farklı belgelerin chunk_id'leri çakışıyor"


def test_doc_key_is_deterministic():
    """Aynı PDF yeniden ingest edilince aynı chunk_id'ler üretilmeli (cache/upsert)."""
    ingest = pytest.importorskip("src.ingest")
    assert ingest._doc_key("kitap.pdf:1000") == ingest._doc_key("kitap.pdf:1000")


class _FakeCollection:
    """ingest_pdf'i ağır bağımlılık olmadan sınamak için sahte koleksiyon."""

    def __init__(self):
        self.ids: list[str] = []
        self.metas: list[dict] = []

    def get(self, where=None, include=None):
        if where and "doc_id" in where:
            keep = [
                (i, m) for i, m in zip(self.ids, self.metas)
                if m.get("doc_id") == where["doc_id"]
            ]
            return {"ids": [i for i, _ in keep], "metadatas": [m for _, m in keep]}
        return {"ids": list(self.ids), "metadatas": list(self.metas)}

    def delete(self, ids):
        keep = [(i, m) for i, m in zip(self.ids, self.metas) if i not in set(ids)]
        self.ids = [i for i, _ in keep]
        self.metas = [m for _, m in keep]

    def upsert(self, ids, documents, metadatas, embeddings):
        self.ids.extend(ids)
        self.metas.extend(metadatas)


def test_new_pdf_replaces_previous_single_document(monkeypatch):
    """Yeni PDF alınınca koleksiyonda yalnızca son belge kalmalı (tek belge kuralı)."""
    ingest = pytest.importorskip("src.ingest")

    fake = _FakeCollection()
    monkeypatch.setattr(ingest, "_collection", lambda name="default": fake)
    class _Vec(list):
        def tolist(self):
            return list(self)

    monkeypatch.setattr(ingest, "_embedder",
                        lambda: type("E", (), {"encode": lambda self, xs, **k: [_Vec([0.0]) for _ in xs]})())
    monkeypatch.setattr(ingest, "extract_pages", lambda p: [(1, "tek sayfa metni")])
    # doc_id dosya yolundan üretiliyor; getsize'ı sabitle.
    monkeypatch.setattr(ingest.os.path, "getsize", lambda p: 111)

    ingest.ingest_pdf("kitap_a.pdf")
    doc_ids_after_a = {m["doc_id"] for m in fake.metas}
    assert doc_ids_after_a == {"kitap_a.pdf:111"}

    ingest.ingest_pdf("kitap_b.pdf")
    doc_ids_after_b = {m["doc_id"] for m in fake.metas}
    # A tamamen gitmeli, yalnızca B kalmalı.
    assert doc_ids_after_b == {"kitap_b.pdf:111"}


# ---------------------------------------------------------------------------
# Agent RAG akışı — Claude API mock'lanmış
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
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_agent_ask_collects_sources(monkeypatch):
    agent = pytest.importorskip("src.agent")

    # retrieve otomatik çağrılır; sabit bir kaynak döndürecek şekilde değiştir.
    # İmza gerçek retrieve ile uyumlu (özet yolu min_score geçirir).
    def fake_retrieve(question, collection="default", top_k=3, min_score=None):
        return [SourceChunk(page=4, passage="ilgili pasaj", chunk_id="p4-c0")]

    monkeypatch.setattr(agent, "retrieve", fake_retrieve)

    # RAG: tek Claude çağrısı, bağlam mesaja gömülü, doğrudan cevap.
    responses = [_Resp("end_turn", [_Block("text", text="Yanıt: 4. sayfaya göre…")])]
    fake = _FakeClient(responses)
    monkeypatch.setattr(agent, "_client", lambda: fake)

    result = agent.ask("PDF neyle ilgili?")
    assert result.answer.startswith("Yanıt")
    assert len(result.sources) == 1
    assert result.sources[0].page == 4
    # Sabit RAG akışı: bağlam mesaja gömülü olduğu için tek Claude çağrısı yeter.
    assert fake.messages.call_count == 1


def test_summary_query_detection():
    """Özet/genel bakış ifadeleri özet olarak algılanmalı; özel sorular değil."""
    agent = pytest.importorskip("src.agent")
    for q in ["pdfi özetle", "özet çıkar", "metni özetle", "bu metin ne anlatıyor",
              "konusu ne", "summarize this"]:
        assert agent._is_summary_query(q), q
    for q in ["kırlangıçlar neden ayrıldı", "kaç sayfa var", "yazar kim"]:
        assert not agent._is_summary_query(q), q


def test_summary_query_bypasses_threshold(monkeypatch):
    """Özet sorgusunda retrieve eşiksiz (min_score=0.0) çağrılmalı."""
    agent = pytest.importorskip("src.agent")
    seen = {}

    def fake_retrieve(question, collection="default", top_k=3, min_score=None):
        seen["min_score"] = min_score
        return [SourceChunk(page=1, passage="p", chunk_id="p1-c0")]

    monkeypatch.setattr(agent, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        agent, "_client",
        lambda: _FakeClient([_Resp("end_turn", [_Block("text", text="Özet…")])]),
    )
    agent.ask("pdfi özetle")
    assert seen["min_score"] == 0.0


def test_agent_ask_no_sources_returns_empty_and_says_so(monkeypatch):
    agent = pytest.importorskip("src.agent")

    monkeypatch.setattr(agent, "retrieve", lambda *a, **k: [])
    responses = [_Resp("end_turn", [_Block("text", text="Kaynak yok.")])]
    monkeypatch.setattr(agent, "_client", lambda: _FakeClient(responses))

    result = agent.ask("alakasız soru")
    # Kaynak uydurulmaz: ilgili pasaj yoksa sources boş döner.
    assert result.sources == []
    assert result.answer


# ---------------------------------------------------------------------------
# API /ask kota davranışı — başarısız yanıt kotayı tüketmemeli
# ---------------------------------------------------------------------------


def test_failed_ask_does_not_consume_quota(monkeypatch):
    """agent_ask hata verirse günlük kota düşülmemeli; başarıda düşülmeli."""
    main = pytest.importorskip("api.main")
    from fastapi.testclient import TestClient
    from src.schemas import AskResponse

    main._question_counter.update({"date": "", "count": 0})  # sayacı sıfırla
    client = TestClient(main.app)

    # 1) agent_ask hata fırlatır → 503, kota düşmez (used=0).
    monkeypatch.setattr(main, "agent_ask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("anahtar yok")))
    r = client.post("/ask", json={"question": "soru"})
    assert r.status_code == 503
    assert client.get("/limits").json()["used"] == 0

    # 2) agent_ask başarılı → 200, kota bir düşer (used=1).
    monkeypatch.setattr(
        main, "agent_ask",
        lambda *a, **k: AskResponse(answer="cevap", sources=[]),
    )
    r = client.post("/ask", json={"question": "soru"})
    assert r.status_code == 200
    assert client.get("/limits").json()["used"] == 1
