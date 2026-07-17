"""RAG akışı + Claude API.

Sabit boru hattı (ReAct değil):
  soru → retriever.retrieve ile otomatik bağlam çekimi →
  bağlam + soru tek bir Claude çağrısına verilir → yanıt üretilir.
Modelin "arama yapayım mı" kararı yoktur; retrieve her zaman çalışır.

Kurallar:
  - API anahtarı yalnızca ortam değişkeninden (ANTHROPIC_API_KEY) okunur.
  - Her yanıt kaynak içermek zorundadır; kaynak yoksa bu açıkça bildirilir.
"""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from src.prompts import SYSTEM_PROMPT, format_chunks_for_tool_result
from src.retriever import DEFAULT_TOP_K, retrieve
from src.schemas import AskResponse, SourceChunk

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048


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
    """Soruyu RAG akışıyla yanıtlar: otomatik retrieve → tek Claude çağrısı."""
    chunks = retrieve(question, collection=collection, top_k=top_k)
    context = format_chunks_for_tool_result(chunks)

    response = _client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"}],
    )
    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    if not chunks:
        # Kaynak gösterimi zorunlu; hiç pasaj getirilmediyse bunu açıkça belirt.
        return AskResponse(
            answer=answer_text
            or "Soruyu yanıtlayacak ilgili bir kaynak PDF'te bulunamadı.",
            sources=[
                SourceChunk(
                    page=1,
                    passage="(Uygun kaynak bulunamadı — önce bir PDF ingest edildiğinden emin olun.)",
                )
            ],
        )

    return AskResponse(answer=answer_text or "Yanıt üretilemedi.", sources=chunks)
