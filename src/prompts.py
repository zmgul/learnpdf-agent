"""Sistem promptu ve biçimlendirme yardımcıları (RAG).

Bağlam pasajları soruyla birlikte Claude'a verilir; yanıt yalnızca bu
pasajlara dayanır ve kaynak (sayfa no) gösterilir.
"""

from __future__ import annotations

from src.schemas import SourceChunk

# --- Sistem promptu (Role / Task / Constraints / Output) ------------------

SYSTEM_PROMPT = """\
Rol: PDF içeriğine dayalı soru-cevap asistanı.
Görev: Verilen bağlam pasajlarını kullanarak soruyu yanıtla.
Kısıtlar:
- Yalnızca bağlamdaki pasajlara dayan; bilgi uydurma.
- Bağlam soruyu yanıtlamıyorsa açıkça belirt.
- Yanıtı sorunun dilinde ver; kısa ve doğru ol.
Çıktı: Doğrudan cevap ve dayandığın sayfa numaraları."""


# --- Biçimlendirme yardımcıları -------------------------------------------


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
