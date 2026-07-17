"""PDF → chunk → embed → ChromaDB ingest hattı.

Akış:
  1. PyMuPDF (fitz) ile her sayfanın metnini çıkar.
  2. Manuel chunking (kelime tabanlı, ~chunk_size token, overlap örtüşme).
  3. paraphrase-multilingual-mpnet-base-v2 ile embed et.
  4. ChromaDB persist koleksiyonuna sayfa numarası metadata'sı ile yaz.

Kurallar:
  - LangChain yok; chunking burada elle yazılır.
  - Her chunk, kaynak gösterimi için `page` metadata'sı taşır.
  - Vektör DB yalnızca bu modül üzerinden doldurulur.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import chromadb
import fitz  # PyMuPDF
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.schemas import IngestResponse

# --- Sabitler -------------------------------------------------------------

EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


# --- Veri yapıları --------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """Tek bir embed edilebilir metin parçası ve kaynak sayfası."""

    chunk_id: str
    page: int
    text: str


# --- Lazy yüklenen kaynaklar ---------------------------------------------


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    """Embedding modelini tek sefer yükler (çok dilli)."""
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    """Persist edilen ChromaDB istemcisini döndürür."""
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _collection(name: str):
    """İlgili koleksiyonu döndürür; yoksa oluşturur (cosine uzaklığı)."""
    return _client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# --- PDF okuma ------------------------------------------------------------


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """PDF'i açar ve (1 tabanlı sayfa no, metin) çiftleri döndürür.

    Boş (metinsiz) sayfalar atlanır.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF bulunamadı: {pdf_path}")

    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append((index + 1, text))
    return pages


# --- Manuel chunking ------------------------------------------------------


def chunk_page(
    page: int,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Bir sayfanın metnini kelime tabanlı, örtüşmeli chunk'lara böler.

    Token yaklaşıklığı olarak boşlukla ayrılmış kelimeler kullanılır
    (harici tokenizer bağımlılığı eklemeden manuel yaklaşım).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap, chunk_size'dan küçük olmalı")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        piece = " ".join(window)
        chunk_id = f"p{page}-c{start // step}"
        chunks.append(Chunk(chunk_id=chunk_id, page=page, text=piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(
    pages: list[tuple[int, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Tüm sayfaları sırayla chunk'lara böler."""
    chunks: list[Chunk] = []
    for page, text in pages:
        chunks.extend(chunk_page(page, text, chunk_size, overlap))
    return chunks


# --- Ana ingest -----------------------------------------------------------


def _doc_id(pdf_path: str) -> str:
    """PDF'i tanımlayan basit kimlik: dosya adı + bayt boyutu."""
    return f"{os.path.basename(pdf_path)}:{os.path.getsize(pdf_path)}"


def ingest_pdf(
    pdf_path: str,
    collection: str = "default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestResponse:
    """PDF'i okuyup chunk'layıp embed ederek ChromaDB'ye yazar.

    Embedding cache: aynı PDF (dosya adı + boyut) koleksiyonda zaten varsa
    yeniden embed edilmez; mevcut kayıtlardan bir özet döndürülür. Böylece
    pahalı embedding adımı ve model yüklemesi tekrarlanmaz.
    """
    filename = os.path.basename(pdf_path)
    doc_id = _doc_id(pdf_path)
    col = _collection(collection)

    existing = col.get(where={"doc_id": doc_id})
    if existing["ids"]:
        metas = existing["metadatas"] or [{}]
        return IngestResponse(
            filename=filename,
            collection=collection,
            pages=int(metas[0].get("pages", 0)),
            chunks=len(existing["ids"]),
        )

    pages = extract_pages(pdf_path)
    chunks = build_chunks(pages, chunk_size, overlap)

    if chunks:
        embeddings = _embedder().encode(
            [c.text for c in chunks],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {"page": c.page, "doc_id": doc_id, "pages": len(pages)} for c in chunks
            ],
            embeddings=[e.tolist() for e in embeddings],
        )

    return IngestResponse(
        filename=filename,
        collection=collection,
        pages=len(pages),
        chunks=len(chunks),
    )
