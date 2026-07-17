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
- Düz metin yaz. Markdown kullanma: başlık (#), kalın (*), madde imi (-),
  ayraç (---) yok. Gereksiz boş satır bırakma; birkaç kısa cümle/paragraf yeter.
- Kullandığın bilgiyi, dayandığın kaynağın numarasıyla belirt: ilgili cümlenin
  sonuna köşeli parantez içinde numara koy, örn. [1]. Numara sana verilen
  "Kaynak N" ile birebir aynı olmalı. Birden çok kaynak için ayrı yaz: [1][2].
  Yalnızca gerçekten kullandığın kaynakları göster.
Çıktı: Doğrudan, sade cevap (kaynak atıflarıyla)."""


# --- Biçimlendirme yardımcıları -------------------------------------------


def format_chunks_for_context(chunks: list[SourceChunk]) -> str:
    """Getirilen chunk'ları Claude mesajına gömülecek bağlam metnine çevirir."""
    if not chunks:
        return "Bağlam boş: bu soruya uygun pasaj bulunamadı."

    return "\n\n".join(
        f"[Kaynak {i} | sayfa {chunk.page}]\n{chunk.passage}"
        for i, chunk in enumerate(chunks, start=1)
    )
