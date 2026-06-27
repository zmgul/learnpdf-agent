"""FastAPI uygulaması — /ingest, /ask endpoint'leri ve statik arayüz.

  POST /ingest : PDF yükle → data/'ya kaydet → ingest_pdf → IngestResponse
  POST /ask    : AskRequest → agent.ask → AskResponse
  GET  /       : static/index.html arayüzünü servis et

Hatalar HTTPException (ErrorResponse gövdesi) ile döner.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.agent import ask as agent_ask
from src.ingest import ingest_pdf
from src.schemas import AskRequest, AskResponse, IngestResponse

# Proje kökü ve sabit dizinler (yalnızca data/ içine PDF kabul edilir).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="research-agent", description="PDF ReAct Document Q&A Agent")


@app.get("/")
def index() -> FileResponse:
    """Tek sayfalık soru-cevap arayüzünü döndürür."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="Arayüz bulunamadı (static/index.html)")
    return FileResponse(index_path)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    collection: str = Form("default"),
) -> IngestResponse:
    """Yüklenen PDF'i data/ dizinine kaydeder ve vektör deposuna alır."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Yalnızca PDF dosyaları kabul edilir")

    os.makedirs(DATA_DIR, exist_ok=True)
    # Yol gezinmesini (path traversal) önlemek için yalnızca dosya adını kullan.
    safe_name = os.path.basename(file.filename)
    dest = os.path.join(DATA_DIR, safe_name)

    try:
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        return ingest_pdf(dest, collection=collection)
    except Exception as exc:  # ingest/IO hatalarını istemciye anlamlı ilet
        raise HTTPException(status_code=500, detail=f"Ingest hatası: {exc}") from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Doğal dil sorusunu ReAct ajanıyla yanıtlar (kaynak gösterir)."""
    try:
        return agent_ask(
            request.question,
            collection=request.collection,
            top_k=request.top_k,
        )
    except RuntimeError as exc:  # eksik API anahtarı vb.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Yanıt hatası: {exc}") from exc
