"""Pydantic v2 request/response modelleri.

Tüm API sözleşmeleri (request/response) bu dosyada tek noktada tanımlanır.
Kural: her yanıt en az bir kaynak chunk'ı (sayfa no + pasaj) içerir.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Ortak / yardımcı modeller
# ---------------------------------------------------------------------------


class SourceChunk(BaseModel):
    """Yanıtın dayandığı tek bir kaynak pasaj.

    Her AskResponse en az bir SourceChunk içermek zorundadır.
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


class IngestRequest(BaseModel):
    """PDF ingest isteği.

    Dosya yüklemesi endpoint'te multipart/form-data ile alınır; bu model
    yalnızca ingest davranışını ayarlayan opsiyonel parametreleri taşır.
    """

    model_config = ConfigDict(extra="forbid")

    collection: str = Field(
        default="default",
        min_length=1,
        description="Chunk'ların yazılacağı ChromaDB koleksiyon adı",
    )
    chunk_size: int = Field(
        default=500, ge=100, le=2000, description="Hedef chunk boyutu (token)"
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=500, description="Ardışık chunk'lar arası örtüşme (token)"
    )


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
        ..., min_length=1, max_length=2000, description="Kullanıcının doğal dil sorusu"
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

    `sources` en az bir öğe içermelidir — kaynaksız yanıt geçersizdir.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, description="Claude'un ürettiği nihai yanıt")
    sources: list[SourceChunk] = Field(
        ..., min_length=1, description="Yanıtın dayandığı kaynak chunk'lar"
    )
    iterations: int = Field(
        default=1, ge=1, description="ReAct döngüsünün tamamlanma adım sayısı"
    )


# ---------------------------------------------------------------------------
# Hata modeli
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standart hata gövdesi."""

    model_config = ConfigDict(extra="forbid")

    detail: str = Field(..., description="İnsan tarafından okunabilir hata açıklaması")
