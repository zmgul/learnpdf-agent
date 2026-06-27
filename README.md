# research-agent

PDF üzerinde çalışan **ReAct Document Q&A Agent**. Yüklenen bir PDF'i vektör
deposuna alır ve Claude API aracılığıyla doğal dil sorularını **kaynak göstererek**
(sayfa numarası + pasaj) yanıtlar.

## Yığın

Python 3.11 · FastAPI · Pydantic v2 · Claude API (`claude-sonnet-4-6`) ·
ChromaDB · sentence-transformers · PyMuPDF · Poetry

Mimari: PDF → manuel chunking → embedding → ChromaDB → ReAct ajanı (`retrieve`
aracı + Claude tool-use döngüsü).

## Kurulum

```bash
# 1. Bağımlılıkları kur (Poetry gerekli)
poetry install

# 2. API anahtarını ayarla
cp .env.example .env
# .env içine ANTHROPIC_API_KEY=sk-ant-... değerini gir
```

`.env.example` içeriği:

```
ANTHROPIC_API_KEY=
```

> Anahtar yalnızca ortam değişkeninden okunur; koda asla yazılmaz.

## Çalıştırma

```bash
poetry run uvicorn api.main:app --reload
```

Ardından tarayıcıda <http://127.0.0.1:8000> adresini aç: PDF yükle, soru sor,
yanıtı ve kaynakları gör.

## API

### `POST /ingest`
PDF yükler, `data/` dizinine kaydeder ve vektör deposuna alır.

```bash
curl -F "file=@tez.pdf" -F "collection=default" http://127.0.0.1:8000/ingest
```

Yanıt (`IngestResponse`):

```json
{ "filename": "tez.pdf", "collection": "default", "pages": 42, "chunks": 88 }
```

### `POST /ask`
Doğal dil sorusunu yanıtlar.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Çalışmanın amacı nedir?", "collection": "default", "top_k": 3}'
```

Yanıt (`AskResponse`):

```json
{
  "answer": "...",
  "sources": [{ "page": 3, "passage": "...", "score": 0.81, "chunk_id": "p3-c0" }],
  "iterations": 2
}
```

## Test

```bash
poetry run pytest
```

Testler Claude API çağrısını mock'lar; gerçek anahtar veya ağ erişimi gerektirmez.

## Dizin Yapısı

```
src/        schemas, ingest, retriever, prompts, agent
api/        FastAPI uygulaması (main.py)
static/     plain HTML + fetch() arayüzü
tests/      temel testler
data/        PDF buraya yerleştirilir (repoya eklenmez)
```

## Notlar

- Çok dilli embedding modeli: `paraphrase-multilingual-mpnet-base-v2`
  (Türkçe dahil çok dilli içerikte kaynak çekimi için).
- ChromaDB `./chroma_db/` dizininde persist edilir; `data/` ve `chroma_db/`
  repoya eklenmez.
- Chunking manuel yazılmıştır (LangChain yok); arayüz plain HTML'dir (Jinja2 yok).
