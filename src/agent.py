"""ReAct döngüsü + Claude API.

Manuel tool-use döngüsü: Claude `retrieve` aracını çağırır, biz aracı
`src.retriever` ile çalıştırıp sonucu geri besleriz; `stop_reason == "end_turn"`
olunca dururuz. Döngü boyunca getirilen tüm `SourceChunk`'lar toplanır ve
nihai `AskResponse` (answer + sources) üretilir.

Kurallar:
  - API anahtarı yalnızca ortam değişkeninden (ANTHROPIC_API_KEY) okunur.
  - Her yanıt kaynak içermek zorundadır; kaynak yoksa bu açıkça bildirilir.
"""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from src.prompts import (
    RETRIEVE_TOOL,
    SYSTEM_PROMPT,
    format_chunks_for_tool_result,
)
from src.retriever import DEFAULT_TOP_K, retrieve
from src.schemas import AskResponse, SourceChunk

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
MAX_ITERATIONS = 5  # sonsuz tool-use döngüsüne karşı koruma


def _client() -> anthropic.Anthropic:
    """Anthropic istemcisini ortam değişkeninden okunan anahtarla döndürür."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY ortam değişkeni tanımlı değil")
    return anthropic.Anthropic()


def _run_retrieve(tool_input: dict, collection: str, default_top_k: int) -> list[SourceChunk]:
    """`retrieve` araç çağrısını çalıştırır."""
    query = tool_input.get("query", "")
    top_k = int(tool_input.get("top_k") or default_top_k)
    return retrieve(query, collection=collection, top_k=top_k)


def ask(
    question: str,
    collection: str = "default",
    top_k: int = DEFAULT_TOP_K,
) -> AskResponse:
    """Soruyu ReAct döngüsüyle yanıtlar ve kaynak gösteren bir yanıt döndürür."""
    client = _client()
    messages: list[dict] = [{"role": "user", "content": question}]

    collected: dict[str, SourceChunk] = {}  # chunk_id → SourceChunk (yinelenmeyi önler)
    answer_text = ""
    iterations = 0

    for iterations in range(1, MAX_ITERATIONS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[RETRIEVE_TOOL],
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Asistanın yanıtını (tool_use blokları dahil) geçmişe ekle.
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                chunks = _run_retrieve(block.input, collection, top_k)
                for chunk in chunks:
                    key = chunk.chunk_id or f"p{chunk.page}-{hash(chunk.passage)}"
                    collected[key] = chunk
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": format_chunks_for_tool_result(chunks),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        # end_turn (veya başka bir terminal sebep): nihai metni topla ve çık.
        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        break

    sources = list(collected.values())
    if not sources:
        # Kaynak gösterimi zorunlu; hiç pasaj getirilmediyse bunu açıkça belirt.
        return AskResponse(
            answer=(
                answer_text
                or "Soruyu yanıtlayacak ilgili bir kaynak PDF'te bulunamadı."
            ),
            sources=[
                SourceChunk(
                    page=1,
                    passage="(Uygun kaynak bulunamadı — önce bir PDF ingest edildiğinden emin olun.)",
                )
            ],
            iterations=iterations,
        )

    return AskResponse(
        answer=answer_text or "Yanıt üretilemedi.",
        sources=sources,
        iterations=iterations,
    )
