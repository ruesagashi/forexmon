# Forexmon Dashboard — Setup Guide

## File yang perlu ditambah ke project

```
forexmon/
├── monitoring/
│   ├── api_server.py     ← FastAPI backend (file baru)
│   └── alerting.py       ← existing
├── dashboard.html         ← taruh di root project (file baru)
└── ...
```

## Install dependencies tambahan

```bash
pip install fastapi uvicorn
```

## Cara Menjalankan

### 1. Jalankan bot dulu (seperti biasa)
```bash
python main.py
```

### 2. Di terminal baru, jalankan API server
```bash
cd /path/to/forexmon
uvicorn monitoring.api_server:app --host 0.0.0.0 --port 8765 --reload
```
.\venv\Scripts\python.exe -m uvicorn monitoring.api_server:app --host 0.0.0.0 --port 8765 --reload

### 3. Buka dashboard
Buka file `dashboard.html` di browser (double-click atau drag ke browser).

Dashboard akan auto-refresh setiap **5 detik** mengambil data real dari SQLite.

---

## API Endpoints

| Endpoint | Deskripsi |
|---|---|
| GET /api/summary | PnL, WinRate, jumlah specialist, dll |
| GET /api/specialists | List specialist aktif + performa |
| GET /api/recent_trades | 8 trade terbaru yang sudah close |
| GET /api/equity_curve | Data equity curve 24 jam terakhir |
| GET /api/regime | Regime terakhir dari Master Brain |
| GET /api/fast_kill_log | Log eliminasi dan promosi specialist |
| GET /api/health | Status server |

---

## Troubleshooting

**Dashboard tidak connect ke API?**
- Pastikan uvicorn sudah running di port 8765
- Cek pojok kanan atas — kalau merah "API DOWN" berarti server mati

**Data kosong / "Waiting for trades..."?**
- Bot belum execute trade apapun
- Jalankan `python populate_specialists.py` dulu untuk isi initial data

**CORS error di browser?**
- Ini normal kalau buka dari file:// langsung
- Solusi: jalankan simple HTTP server:
  ```bash
  python -m http.server 3000
  ```
  Lalu buka: http://localhost:3000/dashboard.html
