"""FastAPI uygulaması — /ingest, /ask, /limits endpoint'leri ve statik arayüz.

  POST /ingest : PDF yükle → data/'ya kaydet → ingest_pdf → IngestResponse
  POST /ask    : AskRequest → agent.ask (RAG) → AskResponse
  GET  /limits : günlük soru limiti durumu → LimitStatus
  GET  /       : static/index.html arayüzünü servis et

Kısıtlar (token verimliliği):
  - Soru en fazla 300 karakter (AskRequest); aşılırsa HTTP 400.
  - Günlük toplam soru sayısı MAX_QUESTIONS_PER_DAY (varsayılan 10) ile
    sınırlı; aşılırsa HTTP 429. Basit in-memory sayaç, her gün sıfırlanır.

Hatalar HTTPException ile {"detail": "..."} gövdesi olarak döner.
"""

from __future__ import annotations

import datetime
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from src.agent import ask as agent_ask
from src.ingest import ingest_pdf
from src.schemas import AskRequest, AskResponse, IngestResponse, LimitStatus

load_dotenv()

# Proje kökü ve sabit dizinler (yalnızca data/ içine PDF kabul edilir).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Günlük soru limiti (.env'den, sabit yazma). Basit süreç-içi sayaç.
MAX_QUESTIONS_PER_DAY = int(os.environ.get("MAX_QUESTIONS_PER_DAY", "10"))
_question_counter = {"date": "", "count": 0}

app = FastAPI(title="learnpdf-agent", description="PDF RAG Document Q&A Agent")


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Pydantic doğrulama hatalarını okunabilir tek satırlık 400 mesajına çevirir."""
    detail = "Geçersiz istek."
    for err in exc.errors():
        loc = err.get("loc", ())
        if "question" in loc and "too_long" in err.get("type", ""):
            detail = "Soru en fazla 300 karakter olabilir."
            break
        if "question" in loc and "too_short" in err.get("type", ""):
            detail = "Soru boş olamaz."
            break
    return JSONResponse(status_code=400, content={"detail": detail})


def _limit_status() -> LimitStatus:
    """Bugünkü kullanım durumunu döndürür (gün değişince kullanılan 0'a düşer)."""
    today = datetime.date.today().isoformat()
    used = _question_counter["count"] if _question_counter["date"] == today else 0
    return LimitStatus(
        limit=MAX_QUESTIONS_PER_DAY,
        used=used,
        remaining=max(0, MAX_QUESTIONS_PER_DAY - used),
    )


def _quota_available() -> bool:
    """Bugün kota kaldı mı? (Sayacı ARTIRMAZ; gün değişince sıfırlar.)"""
    today = datetime.date.today().isoformat()
    if _question_counter["date"] != today:
        _question_counter["date"] = today
        _question_counter["count"] = 0
    return _question_counter["count"] < MAX_QUESTIONS_PER_DAY


def _record_question() -> None:
    """Başarılı bir soruyu kotadan düşer (yalnızca yanıt üretildikten sonra)."""
    today = datetime.date.today().isoformat()
    if _question_counter["date"] != today:
        _question_counter["date"] = today
        _question_counter["count"] = 0
    _question_counter["count"] += 1


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


@app.get("/limits", response_model=LimitStatus)
def limits() -> LimitStatus:
    """Günlük soru limiti durumunu döndürür (arayüz buton durumu için)."""
    return _limit_status()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Doğal dil sorusunu RAG akışıyla yanıtlar (kaynak gösterir).

    Soru uzunluğu AskRequest ile 300 karaktere sınırlı (aşımda 400). Günlük
    kota MAX_QUESTIONS_PER_DAY ile sınırlı (aşımda 429).
    """
    # Ek güvenlik: arayüz engellese bile sunucu tarafında da uzunluğu doğrula.
    if len(request.question) > 300:
        raise HTTPException(status_code=400, detail="Soru en fazla 300 karakter olabilir.")

    if not _quota_available():
        raise HTTPException(status_code=429, detail="Günlük soru limitine ulaşıldı.")

    try:
        response = agent_ask(
            request.question,
            collection=request.collection,
            top_k=request.top_k,
        )
    except RuntimeError as exc:  # eksik API anahtarı vb.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Yanıt hatası: {exc}") from exc

    # Kota yalnızca başarılı yanıtta düşülür (hatada kullanıcı hakkını kaybetmesin).
    _record_question()
    return response
