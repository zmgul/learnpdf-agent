"""ChromaDB sorgu aracı — ReAct ajanının "retrieve" aracı.

Soruyu ingest hattıyla aynı çok dilli modelle embed eder, ilgili
koleksiyonu sorgular ve sonuçları `SourceChunk` listesine dönüştürür.
Embedding modeli ve ChromaDB istemcisi `src.ingest` ile paylaşılır;
böylece sorgu ve ingest aynı vektör uzayında çalışır.
"""

from __future__ import annotations

from src.ingest import _collection, _embedder
from src.schemas import SourceChunk

DEFAULT_TOP_K = 3


def retrieve(
    question: str,
    collection: str = "default",
    top_k: int = DEFAULT_TOP_K,
) -> list[SourceChunk]:
    """Soruya en benzer chunk'ları döndürür (page + passage + score).

    Cosine uzaklığı `score = 1 - distance` ile benzerliğe çevrilir ve
    [0, 1] aralığına kırpılır. Koleksiyon boşsa boş liste döner.
    """
    if not question.strip():
        raise ValueError("Soru boş olamaz")

    col = _collection(collection)
    if col.count() == 0:
        return []

    n = min(top_k, col.count())
    query_embedding = _embedder().encode(
        [question],
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]

    result = col.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma sonuçları liste-içinde-liste döndürür (sorgu başına bir öğe).
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    chunks: list[SourceChunk] = []
    for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
        score = max(0.0, min(1.0, 1.0 - float(dist)))
        chunks.append(
            SourceChunk(
                page=int((meta or {}).get("page", 1)),
                passage=doc,
                score=round(score, 4),
                chunk_id=chunk_id,
            )
        )
    return chunks
