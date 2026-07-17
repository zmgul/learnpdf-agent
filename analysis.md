# PROJE: learnpdf-agent — Sistem Analizi
Sürüm: 0.1

## 1. AMAÇ
Kullanıcının yüklediği herhangi bir PDF'i (kitap, ders notu, döküman) vektör
deposuna alıp Claude API ile doğal dil sorularına kaynak göstererek yanıt veren
genel bir PDF öğrenme aracı.

## 2. TEKNOLOJİ YIĞINI
| Katman         | Teknoloji                      | Sürüm             |
|----------------|-------------------------------|-------------------|
| Dil            | Python                         | 3.11              |
| API Sunucu     | FastAPI                        | 0.111             |
| Veri Doğrulama | Pydantic                       | 2.x               |
| LLM            | Claude (Anthropic API)         | claude-sonnet-4-6 |
| PDF Ayrıştırma | PyMuPDF (fitz)                 | 1.24              |
| Chunking       | Manuel (sıfır bağımlılık)      | —                 |
| Vektör DB      | ChromaDB                       | 0.5               |
| Embedding      | sentence-transformers          | 3.x               |
| Arayüz         | FastAPI + plain HTML + fetch() | —                 |
| Paket Yönetimi | Poetry                         | 2.3.2             |
| Ortam Yönetimi | python-dotenv                  | 1.x               |

## 3. DİZİN HARİTASI
```
learnpdf-agent/
├── src/
│   ├── ingest.py        # PDF → chunk → embed → ChromaDB
│   ├── retriever.py     # ChromaDB bağlam çekme adımı
│   ├── agent.py         # RAG akışı (otomatik retrieve → tek Claude çağrısı)
│   ├── schemas.py       # Pydantic modelleri (request/response)
│   └── prompts.py       # Sistem promptu ve şablonlar
├── api/
│   └── main.py          # FastAPI router'ları (/ingest, /ask)
├── static/
│   └── index.html       # Tek sayfalık soru-cevap arayüzü (plain HTML + fetch)
├── data/
│   └── .gitkeep         # PDF buraya yerleştirilir, repoya eklenmez
├── tests/
│   └── test_agent.py    # Temel soru-cevap testleri
├── analysis.md          # Bu dosya — proje kökeni
├── state.md             # Oturum kontrol noktası
├── .env.example         # ANTHROPIC_API_KEY=...
├── pyproject.toml       # Poetry bağımlılık tanımı
└── README.md
```

## 4. TEMEL DEĞİŞMEZLER
- PDF yalnızca `data/` dizinine yerleştirilir; başka yol kabul edilmez
- Claude API anahtarı yalnızca `.env` üzerinden okunur; kodda hardcode yasak
- Bağlama dayanan her yanıt kaynak göstermeli; bağlamda ilgili içerik
  bulunamazsa yanıt bunu açıkça belirtmeli ve kaynak uydurmamalı
- Vektör DB yalnızca `src/ingest.py` üzerinden doldurulur
- Tüm request/response modelleri `src/schemas.py` içinde Pydantic v2 ile tanımlanır
- Bağımlılık yönetimi yalnızca Poetry; `pip install` doğrudan kullanılmaz

## 5. MEVCUT DURUM
- Çalışıyor  : —
- Eksik/Bozuk: Her şey — proje henüz başlangıç aşamasında
- Sonraki öncelik: `pyproject.toml` oluştur, ardından `src/schemas.py` yaz

## 6. BİLİNEN KISITLAMALAR
- Yüklenen PDF'ler çok dilli olabilir (Türkçe dahil); embedding modeli çok dilli olmalı → `paraphrase-multilingual-mpnet-base-v2` kullan
- Claude API rate limit: ücretsiz tier'da dikkatli ol
- `data/` dizini .gitignore'a eklenir; PDF repoya push edilmez
