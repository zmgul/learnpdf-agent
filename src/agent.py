"""RAG akışı + Claude API.

Sabit boru hattı:
  soru → retriever.retrieve ile otomatik bağlam çekimi →
  bağlam + soru tek bir Claude çağrısına verilir → yanıt üretilir.
Modelin "arama yapayım mı" kararı yoktur; retrieve her zaman çalışır.

Kurallar:
  - API anahtarı yalnızca ortam değişkeninden (ANTHROPIC_API_KEY) okunur.
  - Bağlama dayanan yanıtlar kaynak gösterir; ilgili içerik yoksa yanıt bunu
    açıkça belirtir ve kaynak uydurulmaz (sources boş döner).
"""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from src.prompts import SYSTEM_PROMPT, format_chunks_for_context
from src.retriever import DEFAULT_TOP_K, retrieve
from src.schemas import AskResponse

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 512  # yanıt (çıktı) token tavanı — token verimliliği için sınırlı

# Fiyatlandırma (USD / 1M token) — claude-sonnet-4-6. Model değişirse güncelle.
PRICE_INPUT_PER_1M = 3.00
PRICE_OUTPUT_PER_1M = 15.00


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Giriş/çıkış token sayısından tahmini USD maliyeti hesaplar."""
    return round(
        input_tokens / 1_000_000 * PRICE_INPUT_PER_1M
        + output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M,
        6,
    )

# Özet/genel bakış niyeti taşıyan ifadeler. Bu tür sorgular belgenin tamamını
# ister; hiçbir tek pasaja anlamca benzemedikleri için benzerlik skorları düşük
# kalır ve eşiğe takılır. Bu durumda eşik atlanıp belge geniş kapsanır.
_SUMMARY_HINTS = (
    "özet", "özetle", "kısaca", "genel olarak", "genel bir bakış",
    "ne anlat", "neyle ilgili", "ne hakkında", "konusu ne", "konu ne",
    "summary", "summarize", "overview",
)
SUMMARY_TOP_K = 10  # özet için belgeyi geniş kapsa (retrieve, chunk sayısıyla sınırlar)


def _is_summary_query(question: str) -> bool:
    """Soru bir özet/genel bakış talebi mi?"""
    q = question.casefold()
    return any(hint in q for hint in _SUMMARY_HINTS)


def _client() -> anthropic.Anthropic:
    """Anthropic istemcisini ortam değişkeninden okunan anahtarla döndürür."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY ortam değişkeni tanımlı değil")
    return anthropic.Anthropic()


def ask(
    question: str,
    collection: str = "default",
    top_k: int = DEFAULT_TOP_K,
) -> AskResponse:
    """Soruyu RAG akışıyla yanıtlar: otomatik retrieve → tek Claude çağrısı.

    Özet talepleri benzerlik eşiğini atlar ve belgeyi geniş kapsar; özel
    sorular ise eşikle korunur (alakasızsa kaynak dönmez).
    """
    if _is_summary_query(question):
        chunks = retrieve(
            question, collection=collection, top_k=max(top_k, SUMMARY_TOP_K), min_score=0.0
        )
    else:
        chunks = retrieve(question, collection=collection, top_k=top_k)
    context = format_chunks_for_context(chunks)

    response = _client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"}],
    )
    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Token kullanımı ve tahmini maliyet (usage yoksa 0'a düşer — test mock'ları).
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cost = _cost_usd(input_tokens, output_tokens)

    if not chunks:
        # İlgili pasaj yok: kaynak uydurma, durumu açıkça bildir.
        return AskResponse(
            answer=answer_text
            or "Soruyu yanıtlayacak ilgili bir pasaj PDF'te bulunamadı.",
            sources=[],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    return AskResponse(
        answer=answer_text or "Yanıt üretilemedi.",
        sources=chunks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )
