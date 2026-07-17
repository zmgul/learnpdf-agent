"""Pydantic v2 request/response modelleri.

Tüm API sözleşmeleri (request/response) bu dosyada tek noktada tanımlanır.
Kural: bağlama dayanan yanıtlar kaynak (sayfa no + pasaj) gösterir; ilgili
içerik yoksa kaynak uydurulmaz ve sources boş liste döner.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Ortak / yardımcı modeller
# ---------------------------------------------------------------------------


class SourceChunk(BaseModel):
    """Yanıtın dayandığı tek bir kaynak pasaj.

    AskResponse.sources bu chunk'lardan oluşur; ilgili içerik yoksa liste
    boş olabilir (kaynak uydurulmaz).
    """

    model_config = ConfigDict(extra="forbid")

    page: int = Field(..., ge=1, description="Pasajın geçtiği PDF sayfa numarası (1 tabanlı)")
    passage: str = Field(..., min_length=1, description="Kaynak metin pasajı")
    score: float | None = Field(
        default=None,
        description="Retriever benzerlik skoru (0–1); yoksa None",
    )
    chunk_id: str | None = Field(
        default=None, description="ChromaDB içindeki chunk kimliği"
    )


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    """PDF ingest sonucu."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(..., description="Yüklenen PDF dosya adı")
    collection: str = Field(..., description="Yazılan ChromaDB koleksiyon adı")
    pages: int = Field(..., ge=0, description="İşlenen sayfa sayısı")
    chunks: int = Field(..., ge=0, description="Oluşturulan ve embed edilen chunk sayısı")


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Doğal dil soru isteği."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Kullanıcının doğal dil sorusu (en fazla 300 karakter)",
    )
    collection: str = Field(
        default="default",
        min_length=1,
        description="Sorgulanacak ChromaDB koleksiyon adı",
    )
    top_k: int = Field(
        default=3, ge=1, le=10, description="Claude'a iletilecek chunk sayısı"
    )


class AskResponse(BaseModel):
    """Ajanın ürettiği yanıt ve dayandığı kaynaklar.

    Bağlama dayanan yanıtlar kaynak gösterir. Bağlamda ilgili içerik
    bulunamazsa `sources` boş liste döner — kaynak uydurulmaz; yanıt
    metni durumu açıkça belirtir.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, description="Claude'un ürettiği nihai yanıt")
    sources: list[SourceChunk] = Field(
        ...,
        min_length=0,
        description="Yanıtın dayandığı kaynak chunk'lar; ilgili içerik yoksa boş",
    )
    input_tokens: int = Field(default=0, ge=0, description="Bu soru için giriş token sayısı")
    output_tokens: int = Field(default=0, ge=0, description="Bu soru için çıkış token sayısı")
    cost_usd: float = Field(default=0.0, ge=0, description="Bu sorunun tahmini maliyeti (USD)")


# ---------------------------------------------------------------------------
# /limits — kullanım limiti durumu
# ---------------------------------------------------------------------------


class LimitStatus(BaseModel):
    """Günlük soru limiti durumu (arayüzün buton durumunu ayarlaması için)."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(..., ge=0, description="Günlük izin verilen toplam soru sayısı")
    used: int = Field(..., ge=0, description="Bugün kullanılan soru sayısı")
    remaining: int = Field(..., ge=0, description="Bugün kalan soru sayısı")
