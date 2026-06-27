"""Sistem promptu, Claude tool-use araç tanımı ve biçimlendirme yardımcıları.

ReAct ajanı `retrieve` aracını kullanarak PDF'ten kaynak çeker ve yanıtını
yalnızca bu kaynaklara dayandırır. Kaynak göstermeden yanıt verilmez.
"""

from __future__ import annotations

from src.schemas import SourceChunk

# ---------------------------------------------------------------------------
# Sistem promptu
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Sen bir PDF Doküman Soru-Cevap ajanısın. Görevin, kullanıcının sorusunu
yüklenen PDF'in içeriğine dayanarak yanıtlamaktır.

Çalışma kuralların:
1. Soruyu yanıtlamak için önce `retrieve` aracını çağırarak en ilgili
   pasajları getir. Bilgiyi asla önceden bildiğini varsayma.
2. Yanıtını YALNIZCA getirilen pasajlara dayandır. Pasajlar soruyu
   yanıtlamıyorsa bunu açıkça belirt; uydurma yapma.
3. Her yanıt en az bir kaynak göstermek zorundadır: ilgili sayfa numarası
   ve kısa bir alıntı.
4. Gerekirse aramayı farklı anahtar kelimelerle birden fazla kez yapabilirsin.
5. Yanıtını sorunun dilinde (genellikle Türkçe) ver; kısa ve doğru ol.

Yeterli bağlamı topladığında nihai yanıtını yaz."""


# ---------------------------------------------------------------------------
# Claude tool-use araç tanımı
# ---------------------------------------------------------------------------

RETRIEVE_TOOL = {
    "name": "retrieve",
    "description": (
        "Yüklenen PDF'ten, verilen soruya en benzer metin pasajlarını getirir. "
        "Bilgiye dayalı her soruyu yanıtlamadan önce bu aracı çağır. "
        "Sonuçlar sayfa numarası ve pasaj metni içerir."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Vektör deposunda aranacak sorgu metni",
            },
            "top_k": {
                "type": "integer",
                "description": "Getirilecek pasaj sayısı (varsayılan 3)",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Biçimlendirme yardımcıları
# ---------------------------------------------------------------------------


def format_chunks_for_tool_result(chunks: list[SourceChunk]) -> str:
    """Getirilen chunk'ları Claude'a tool_result olarak verilecek metne çevirir."""
    if not chunks:
        return "Bu sorguya uygun pasaj bulunamadı. Farklı anahtar kelimelerle dene."

    return "\n\n".join(
        f"[Kaynak {i} | sayfa {chunk.page}]\n{chunk.passage}"
        for i, chunk in enumerate(chunks, start=1)
    )


def format_sources_for_answer(chunks: list[SourceChunk], max_chars: int = 200) -> str:
    """Kaynakları kullanıcıya gösterilecek kısa özet biçimine dönüştürür."""
    lines: list[str] = []
    for chunk in chunks:
        passage = chunk.passage.strip().replace("\n", " ")
        if len(passage) > max_chars:
            passage = passage[:max_chars].rstrip() + "…"
        lines.append(f"- (sayfa {chunk.page}) {passage}")
    return "\n".join(lines)
