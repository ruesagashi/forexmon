# Product Requirements Document
# Adaptive Forex Trading Bot — MT5 Intelligent Agent System

**Versi:** 1.0.0
**Tanggal:** Juli 2025
**Status:** Draft — Siap untuk Development
**Target Platform:** MetaTrader 5 (MT5)
**Bahasa Implementasi:** Python 3.10+

---

## Daftar Isi

1. [Executive Summary](#1-executive-summary)
2. [Arsitektur Sistem](#2-arsitektur-sistem)
3. [Spesifikasi Tiap Komponen](#3-spesifikasi-tiap-komponen)
4. [Tech Stack & Dependencies](#4-tech-stack--dependencies)
5. [Development Phases & Checklist](#5-development-phases--checklist)
6. [Konfigurasi Sistem](#6-konfigurasi-sistem)
7. [Acceptance Criteria](#7-acceptance-criteria--definition-of-done)
8. [Risiko & Mitigasi](#8-risiko--mitigasi)

---

## 1. Executive Summary

Dokumen ini mendefinisikan seluruh kebutuhan teknis, arsitektur sistem, dan roadmap implementasi untuk membangun **Adaptive Forex Trading Bot** berbasis MetaTrader 5 (MT5). Sistem ini dirancang sebagai entitas trading adaptif yang mampu mendeteksi kondisi pasar secara real-time, memilih strategi yang paling relevan, dan terus belajar dari setiap hasil trade — menyerupai cara berpikir trader berpengalaman.

> **🎯 Visi Produk:** Membangun sistem trading otonom yang tidak bergantung pada satu set rules statis, melainkan mampu beradaptasi secara dinamis terhadap perubahan regime pasar forex, dengan mekanisme seleksi strategi yang cepat, efisien, dan self-improving.

### 1.1 Masalah yang Diselesaikan

| Masalah | Dampak | Solusi dalam Sistem Ini |
|---|---|---|
| Rules statis tidak adaptif terhadap perubahan market | Strategi bagus tiba-tiba loss setelah berhari-hari | Master Brain deteksi regime market secara real-time |
| Puluhan ribu kandidat strategi antri forward test | Bottleneck parah, seleksi sangat lambat | Pre-filter ketat + Fast-Kill protocol sebelum forward test |
| MT5 hanya support 50 order aktif | Slot habis oleh strategi performa rendah | Slot Management System dengan ranking dinamis |
| Target WinRate 90%+ tidak realistis | Development tidak pernah selesai | Target realistis: WR 65%+ dengan Profit Factor > 1.5 |
| Strategi WR 80%+ akhirnya degradasi juga | Tidak ada sistem deteksi dini penurunan performa | Performance Monitor dengan Early Warning System |

---

## 2. Arsitektur Sistem

### 2.1 Gambaran Besar

| Komponen | Fungsi Utama | Teknologi |
|---|---|---|
| Master Brain | Deteksi regime pasar (Trending/Ranging/Volatile/Breakout/Reversal) | HMM / K-Means Clustering |
| Specialist Pool | Kumpulan strategi terlatih per regime pasar | Random Forest / XGBoost |
| Slot Manager | Kelola 50 slot MT5, ranking & rotasi strategi | Custom Python Scheduler |
| Risk Manager | Hitung position sizing, SL/TP adaptif berbasis ATR | ATR-based calculation |
| Memory & Learning | Simpan history, perkuat/lemahkan bobot specialist | SQLite + Multi-Armed Bandit |

### 2.2 Alur Sistem Lengkap

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│  MT5 → OHLCV candle → Feature Engineering               │
│  (RSI, MACD, Bollinger, ATR, Volume, Candle Pattern)    │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────▼──────────────────────────────────┐
│                  MASTER BRAIN                           │
│  → Deteksi Regime: TRENDING / RANGING / VOLATILE /      │
│                    BREAKOUT / REVERSAL                  │
│  → Output: Regime Label + Confidence Score (0-100%)     │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────▼──────────────────────────────────┐
│               SPECIALIST POOL                           │
│  Aktifkan Specialist sesuai regime terdeteksi           │
│  Tiap Specialist output: BUY / SELL / HOLD + confidence │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────▼──────────────────────────────────┐
│               SLOT MANAGER                              │
│  Cek kapasitas: 30 slot Approved, 15 slot Probation     │
│  Ranking by: WinRate × ProfitFactor × RecentScore       │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────▼──────────────────────────────────┐
│               RISK MANAGER                              │
│  Position sizing: % risk per trade (default 1%)         │
│  SL/TP: ATR-based, adaptif per volatilitas saat ini     │
└──────────────────────┬──────────────────────────────────┘
                       ↓
                  Eksekusi MT5
                       ↓
┌──────────────────────▼──────────────────────────────────┐
│             MEMORY & LEARNING                           │
│  Catat hasil, update bobot specialist                   │
│  Multi-Armed Bandit: explore vs exploit                 │
│  Re-evaluasi performance tiap 24 jam                    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Spesifikasi Tiap Komponen

### 3.1 Data Pipeline

#### Input Data

| Parameter | Nilai | Keterangan |
|---|---|---|
| Symbol | Configurable (default: EURUSD) | Bisa multi-symbol |
| Timeframe | M5, M15, H1 (multi-TF) | Gunakan kombinasi untuk konfirmasi |
| History length | Minimal 5.000 candle | Untuk training model |
| Update interval | Setiap candle baru close | Event-driven, bukan polling |

#### Feature Engineering — Wajib Diimplementasikan

| Kategori | Fitur | Library |
|---|---|---|
| Trend | EMA 10/20/50/200, ADX, MACD Line & Signal | ta-lib / ta |
| Momentum | RSI 14, Stochastic %K %D, CCI | ta-lib / ta |
| Volatilitas | ATR 14, Bollinger Band Width, Historical Volatility | ta-lib / ta |
| Volume | OBV, Volume SMA, Volume Ratio | ta-lib / ta |
| Candle Pattern | Doji, Engulfing, Hammer, Pin Bar (binary flag) | Custom function |
| Price Action | Higher High/Lower Low, Support/Resistance proximity | Custom function |
| Market Structure | Swing High/Low detection, Trend direction | Custom function |

---

### 3.2 Master Brain — Regime Detector

> **📌 Komponen Paling Kritis:** Jika regime detection salah, semua keputusan downstream akan salah. Komponen ini harus ditest dan divalidasi paling ketat sebelum lanjut ke komponen lain.

#### 5 Regime yang Harus Dideteksi

| Regime | Karakteristik | Indikator Utama | Specialist Aktif |
|---|---|---|---|
| TRENDING UP | Higher High, Higher Low, ADX > 25, EMA slope naik | ADX, EMA alignment, MACD | Trend-Following Specialist |
| TRENDING DOWN | Lower High, Lower Low, ADX > 25, EMA slope turun | ADX, EMA alignment, MACD | Trend-Following Specialist |
| RANGING | Price bouncing di range, ADX < 20, BB menyempit | ADX, BB Width, ATR rendah | Mean-Reversion Specialist |
| BREAKOUT | Price menembus level kunci + volume spike | BB Width expand, Volume surge, ATR naik | Breakout Specialist |
| REVERSAL | Divergence RSI/MACD, candle pattern konfirmasi | RSI divergence, Pin Bar, Engulfing | Reversal Specialist |

#### Implementasi Teknis

- Gunakan **Hidden Markov Model (HMM)** dengan 5 hidden states sebagai pendekatan utama
- Alternatif: K-Means Clustering pada feature vector `[ADX, BB_Width, ATR_ratio, Volume_ratio, EMA_slope]`
- Output: Regime label + Confidence score (0.0 – 1.0)
- Jika confidence < 0.6 → output **UNCERTAIN** → sistem masuk mode HOLD, tidak ada entry baru
- Deteksi regime dijalankan setiap candle H1 close untuk menghindari noise
- Simpan history regime per hari untuk analisis pola seasonal

---

### 3.3 Specialist Pool

#### Struktur Setiap Specialist

| Atribut | Tipe Data | Deskripsi |
|---|---|---|
| specialist_id | UUID string | Identifier unik tiap specialist |
| regime_type | Enum (5 nilai) | Regime dimana specialist ini aktif |
| model | Pickle file | Model ML yang sudah di-train |
| features_used | List of string | Fitur input yang digunakan model |
| winrate | Float 0.0-1.0 | WinRate rolling 50 trade terakhir |
| profit_factor | Float | Gross Profit / Gross Loss |
| status | Enum | PROBATION / APPROVED / SUSPENDED / ELIMINATED |
| created_at | Datetime | Waktu pertama kali dibuat |
| last_trade_at | Datetime | Waktu trade terakhir |
| total_trades | Integer | Total trade yang sudah dieksekusi |
| regime_accuracy | Float | Akurasi saat dipasangkan dengan regime yang tepat |

#### Algoritma Pemilihan Specialist

- Hanya specialist status **APPROVED** yang boleh eksekusi trade di slot Approved
- Specialist **PROBATION** hanya di slot Probation (max 15 slot)
- Jika ada >1 specialist APPROVED untuk satu regime: pilih berdasarkan **Composite Score**

```
Composite Score = (WinRate × 0.4) + (ProfitFactor/3 × 0.3) + (RecentScore × 0.3)
RecentScore     = WinRate dari 10 trade terakhir (bobot lebih tinggi untuk recency)
```

---

### 3.4 Sistem Seleksi & Eliminasi — FAST-KILL PROTOCOL

> **⚠️ Ini adalah solusi utama untuk problem bottleneck.** Sistem ini harus mengeliminasi kandidat buruk SECEPAT mungkin agar slot tidak terisi strategi sampah.

#### Stage 1 — Pre-Filter (sebelum masuk antrian forward test)

Setiap strategi baru dari generator **WAJIB lolos semua filter ini** sebelum masuk antrian:

| Filter | Threshold | Aksi jika Gagal |
|---|---|---|
| Backtest WinRate (10x run) | > 60% | BUANG langsung, jangan antri |
| Profit Factor | > 1.5 | BUANG langsung, jangan antri |
| Monte Carlo Test (1000 simulasi) | Profit di > 65% skenario | BUANG langsung |
| Max Drawdown Backtest | < 15% | BUANG langsung |
| Regime Match Score | > 0.7 (cocok dengan regime tertentu) | BUANG, terlalu generik |
| Minimum Trades in Backtest | > 100 trade | BUANG, sample terlalu kecil |

#### Stage 2 — Fast-Kill di Forward Test

| Checkpoint | Kondisi Eliminasi | Aksi |
|---|---|---|
| Trade ke-1 s/d 3 | Loss semua 3 berturut-turut | ELIMINASI — early signal buruk |
| Trade ke-5 | WinRate < 40% | ELIMINASI — below random chance |
| Trade ke-10 | WinRate < 50% | ELIMINASI — tidak konsisten |
| Trade ke-15 | WinRate < 55% ATAU Drawdown > 5% | ELIMINASI |
| Trade ke-20 | WinRate < 60% ATAU PF < 1.3 | ELIMINASI |
| Trade ke-20 ✅ | WinRate ≥ 60% + PF ≥ 1.3 | LOLOS → Status APPROVED |

#### Stage 3 — Monitoring Specialist APPROVED

| Trigger | Kondisi | Aksi |
|---|---|---|
| Performance Degradasi | WinRate rolling 20 trade turun < 70% | Status → WARNING |
| Performance Kritis | WinRate rolling 20 trade turun < 60% | Status → SUSPENDED, stop entry baru |
| Konfirmasi Eliminasi | Dalam 48 jam status SUSPENDED tidak recovery | Status → ELIMINATED |
| Recovery | Dari status WARNING, WinRate kembali > 75% dalam 10 trade | Status → APPROVED |
| Regime Shift | Regime berubah dari yang specialist ini kuasai | Specialist di-pause otomatis |

---

### 3.5 Slot Manager

#### Alokasi 50 Slot MT5

| Kategori Slot | Jumlah | Kriteria Pengisi | Prioritas |
|---|---|---|---|
| APPROVED slots | 30 slot | Specialist status APPROVED, ranking tertinggi | Utama |
| PROBATION slots | 15 slot | Specialist sedang forward test | Secondary |
| BUFFER slots | 5 slot | Dicadangkan, untuk manual override / emergency | Reserve |

#### Mekanisme Rotasi Slot

- Setiap **6 jam**: re-ranking semua specialist berdasarkan Composite Score
- Specialist APPROVED dengan score terendah dapat di-replace oleh specialist PROBATION yang baru lolos
- Rotasi hanya terjadi saat **tidak ada open trade** dari specialist yang akan dirotasi
- Maximum **3 rotasi per hari** untuk menghindari churn berlebihan

---

### 3.6 Risk Manager

| Parameter | Default Value | Configurable? | Keterangan |
|---|---|---|---|
| Risk per trade | 1% dari balance | Ya (0.5% – 2%) | Max loss per single trade |
| Max drawdown harian | 3% dari balance | Ya | Jika tercapai: stop semua entry hari itu |
| Max drawdown total | 10% dari balance | Ya | Jika tercapai: system pause + alert |
| SL multiplier | 1.5× ATR(14) | Ya | Lebih lebar saat volatilitas tinggi |
| TP multiplier | 2.0× ATR(14) | Ya | Risk/Reward minimum 1:1.33 |
| Max concurrent trades | 30 (dari 30 slot Approved) | Ya | Tidak boleh melebihi slot Approved |
| Cooldown setelah loss beruntun | 3 loss → pause 1 jam | Ya | Hindari revenge trading |

---

### 3.7 Memory & Learning System

#### Database Schema (SQLite)

| Tabel | Kolom Utama | Fungsi |
|---|---|---|
| specialists | id, regime_type, status, model_path, created_at | Master data semua specialist |
| trades | id, specialist_id, symbol, direction, entry, sl, tp, result, pnl, regime_at_entry | History semua trade |
| performance_snapshots | specialist_id, timestamp, winrate, profit_factor, drawdown | Snapshot performa berkala |
| regime_history | timestamp, symbol, timeframe, regime, confidence | Log history regime per waktu |
| system_events | timestamp, event_type, description, metadata | Audit log semua event penting |

#### Multi-Armed Bandit — Explore vs Exploit

- Gunakan algoritma **UCB1 (Upper Confidence Bound)**
- **EXPLOIT (80% waktu):** Aktifkan specialist dengan Composite Score tertinggi
- **EXPLORE (20% waktu):** Beri kesempatan specialist baru untuk membuktikan diri
- Exploration rate otomatis meningkat jika tidak ada strategi baru lolos dalam 7 hari

---

## 4. Tech Stack & Dependencies

| Kategori | Library/Tool | Versi Minimum | Fungsi |
|---|---|---|---|
| MT5 Connector | MetaTrader5 | 5.0.37+ | Koneksi ke platform MT5, ambil data, kirim order |
| Data Processing | pandas, numpy | 2.0+, 1.24+ | Manipulasi data candle dan feature |
| Technical Indicators | ta | 0.10+ | RSI, MACD, Bollinger, ATR, ADX, dll |
| ML — Regime Detection | hmmlearn | 0.3+ | Hidden Markov Model untuk regime detection |
| ML — Regime Detection Alt | scikit-learn | 1.3+ | K-Means clustering, alternatif HMM |
| ML — Specialist Model | xgboost | 1.7+ | Model prediksi per specialist |
| Database | sqlite3 | Built-in Python | Penyimpanan history dan state |
| Scheduling | schedule | 1.2+ | Jalankan task berkala |
| Logging | loguru | 0.7+ | Logging terstruktur dengan rotation |
| Config Management | pydantic | 2.0+ | Validasi config dan settings |
| Notification | python-telegram-bot | 20.0+ | Alert ke Telegram saat event penting |
| Testing | pytest | 7.0+ | Unit test dan integration test |

### 4.1 Struktur Folder Project

```
adaptive_trading_bot/
├── core/
│   ├── master_brain.py          # Regime detection
│   ├── specialist_pool.py       # Manage semua specialist
│   ├── slot_manager.py          # Kelola 50 slot MT5
│   ├── risk_manager.py          # Position sizing & SL/TP
│   └── memory.py                # Database & learning
├── data/
│   ├── pipeline.py              # Ambil & proses data dari MT5
│   └── features.py              # Feature engineering
├── models/
│   ├── regime_detector.py       # Train & predict regime
│   ├── specialist_trainer.py    # Train specialist model
│   └── saved/                   # Pickle files model terlatih
├── execution/
│   ├── mt5_connector.py         # Connect & order ke MT5
│   └── order_manager.py         # Monitor open orders
├── selection/
│   ├── backtest_engine.py       # Backtest + Monte Carlo
│   ├── forward_test.py          # Forward test protocol
│   └── eliminator.py            # Fast-Kill logic
├── monitoring/
│   ├── performance_tracker.py   # Track WR, PF, DD
│   └── alerting.py              # Telegram notifications
├── config/
│   ├── settings.py              # Semua parameter sistem
│   └── symbols.py               # Konfigurasi per symbol
├── tests/                        # Unit & integration tests
├── main.py                       # Entry point
└── requirements.txt
```

---

## 5. Development Phases & Checklist

> **📋 Panduan untuk Developer:** Setiap task harus diselesaikan dan dicek sebelum lanjut ke task berikutnya dalam satu phase. Jangan skip phase — setiap phase adalah fondasi untuk phase berikutnya.

---

### PHASE 1 — Foundation & Data Pipeline
**Estimasi: 5–7 hari**

#### 1.1 Setup Project
- [ ] Buat struktur folder sesuai spesifikasi di Section 4.1
- [ ] Setup virtual environment Python 3.10+
- [ ] Install semua dependencies dari requirements.txt
- [ ] Setup file `config/settings.py` dengan semua parameter default
- [ ] Setup logging dengan loguru (log ke file + console)
- [ ] Install dan verifikasi MetaTrader5 Python library terhubung ke terminal MT5

#### 1.2 MT5 Connector
- [ ] Implementasi `mt5_connector.py` dengan fungsi: `initialize()`, `shutdown()`, `is_connected()`
- [ ] Fungsi `get_candles(symbol, timeframe, count)` → return pandas DataFrame
- [ ] Fungsi `get_account_info()` → balance, equity, margin
- [ ] Fungsi `get_open_positions()` → list semua posisi terbuka
- [ ] Implementasi auto-reconnect jika koneksi terputus
- [ ] Unit test: test koneksi, test ambil data, test disconnect handling

#### 1.3 Feature Engineering
- [ ] Implementasi semua indikator trend: EMA(10,20,50,200), ADX, MACD
- [ ] Implementasi semua indikator momentum: RSI(14), Stochastic, CCI
- [ ] Implementasi indikator volatilitas: ATR(14), Bollinger Band, Historical Vol
- [ ] Implementasi indikator volume: OBV, Volume SMA, Volume Ratio
- [ ] Implementasi candle pattern detector: Doji, Engulfing, Hammer, Pin Bar
- [ ] Implementasi market structure: Swing High/Low, HH/LL detection
- [ ] Fungsi `get_features(df)` → return DataFrame dengan semua fitur
- [ ] Unit test: verifikasi nilai RSI, MACD, ATR dengan kalkulasi manual

#### 1.4 Database Setup
- [ ] Buat semua tabel SQLite sesuai schema di Section 3.7
- [ ] Implementasi fungsi CRUD untuk tabel `specialists`
- [ ] Implementasi fungsi CRUD untuk tabel `trades`
- [ ] Implementasi fungsi insert untuk `performance_snapshots` dan `regime_history`
- [ ] Implementasi fungsi query: `get_specialist_performance(id, last_n_trades)`
- [ ] Unit test: insert dan query semua tabel

---

### PHASE 2 — Master Brain: Regime Detector
**Estimasi: 7–10 hari**

#### 2.1 Data Labeling untuk Training
- [ ] Download minimal 2 tahun data historis untuk symbol utama (EURUSD H1)
- [ ] Implementasi fungsi `label_regime_manual()` untuk ground truth awal
- [ ] Buat visualisasi chart untuk verifikasi label secara visual
- [ ] Target: minimal 500 sample per regime untuk training yang representatif

#### 2.2 Model Regime Detector
- [ ] Implementasi HMM dengan 5 hidden states menggunakan `hmmlearn`
- [ ] Feature vector untuk HMM: `[ADX, BB_Width, ATR_ratio, Volume_ratio, EMA_slope]`
- [ ] Train model pada data historis yang sudah dilabel
- [ ] Implementasi `predict_regime(features)` → `(regime_label, confidence_score)`
- [ ] Simpan model ke `models/saved/regime_detector.pkl`
- [ ] Implementasi alternatif K-Means sebagai fallback jika HMM confidence < 0.5

#### 2.3 Validasi Regime Detector
- [ ] Confusion matrix: akurasi per regime minimal 70%
- [ ] Backtest visual: plot regime label di atas chart candle — apakah masuk akal?
- [ ] Test edge cases: transisi regime, sideways berkepanjangan, news spike
- [ ] Implementasi UNCERTAIN state ketika confidence < 0.6
- [ ] Unit test: test prediksi pada data out-of-sample
- [ ] ✅ **GATE:** Akurasi regime detection minimal 70% sebelum lanjut ke Phase 3

---

### PHASE 3 — Specialist Pool & Backtest Engine
**Estimasi: 10–14 hari**

#### 3.1 Backtest Engine
- [ ] Implementasi `BacktestEngine` class dengan metode `run(strategy, df_historical)`
- [ ] Output backtest: WinRate, ProfitFactor, MaxDrawdown, TotalTrades, SharpeRatio
- [ ] Implementasi Walk-Forward backtest: split data jadi 10 window
- [ ] Implementasi Monte Carlo simulation (1000 iterasi random trade sequence)
- [ ] Implementasi Pre-Filter berdasarkan threshold di Section 3.4 Stage 1
- [ ] Unit test: verifikasi kalkulasi WinRate dan ProfitFactor dengan data dummy

#### 3.2 Specialist Trainer
- [ ] Implementasi `SpecialistTrainer` class
- [ ] Fungsi `generate_specialist(regime_type)` → train XGBoost model pada data regime tersebut
- [ ] Label target: 1 jika candle berikutnya profit (sesuai direction), 0 jika loss
- [ ] Feature selection: gunakan hanya fitur yang relevan per regime
- [ ] Simpan specialist sebagai file `.pkl` dengan metadata
- [ ] Implementasi `predict(features)` → `(direction: BUY/SELL/HOLD, confidence: 0-1)`

#### 3.3 Specialist Pool Manager
- [ ] Implementasi `SpecialistPool` class
- [ ] Fungsi `add_specialist(specialist)` → cek duplikasi, assign ID, status PROBATION
- [ ] Fungsi `get_active_specialists(regime)` → return list specialist APPROVED
- [ ] Fungsi `update_performance(specialist_id, trade_result)` → update WinRate, PF
- [ ] Fungsi `calculate_composite_score(specialist_id)` → return float 0.0-1.0
- [ ] Fungsi `rank_specialists(regime)` → sorted list by Composite Score

---

### PHASE 4 — Seleksi & Fast-Kill Protocol
**Estimasi: 7–10 hari**

#### 4.1 Forward Test Protocol
- [ ] Implementasi `ForwardTester` class
- [ ] Fungsi `start_forward_test(specialist)` → assign ke slot PROBATION di MT5 demo
- [ ] Implementasi checkpoint evaluation sesuai tabel Fast-Kill di Section 3.4 Stage 2
- [ ] Evaluasi dijalankan otomatis setelah setiap trade close
- [ ] Fungsi `promote_to_approved(specialist)` → pindah ke slot APPROVED
- [ ] Fungsi `eliminate(specialist, reason)` → hapus dari slot, log ke database

#### 4.2 Performance Monitor
- [ ] Implementasi `PerformanceMonitor` yang berjalan setiap 1 jam
- [ ] Hitung rolling WinRate dan ProfitFactor untuk setiap specialist APPROVED
- [ ] Implementasi status transition: APPROVED → WARNING → SUSPENDED → ELIMINATED
- [ ] Implementasi recovery detection: SUSPENDED → APPROVED jika recover
- [ ] Simpan snapshot performance ke `performance_snapshots` setiap 6 jam

---

### PHASE 5 — Slot Manager & Risk Manager
**Estimasi: 5–7 hari**

#### 5.1 Slot Manager
- [ ] Implementasi `SlotManager` class yang track 50 slot MT5
- [ ] Fungsi `get_available_slots(slot_type)` → return jumlah slot kosong
- [ ] Implementasi rotasi otomatis setiap 6 jam
- [ ] Log setiap rotasi ke `system_events` database
- [ ] Fungsi `force_close_position(specialist_id)` → tutup semua posisi specialist ini

#### 5.2 Risk Manager
- [ ] Implementasi `RiskManager` class
- [ ] Fungsi `calculate_position_size(balance, risk_pct, sl_pips)` → return lot size
- [ ] Fungsi `calculate_sl_tp(entry_price, direction, atr_value)` → return sl, tp price
- [ ] Implementasi daily drawdown checker: jika DD > 3%, set `no_new_trades = True`
- [ ] Implementasi loss streak detector: 3 loss beruntun → cooldown 1 jam
- [ ] Reset semua daily flags pada awal hari baru (00:00 server time)

---

### PHASE 6 — Memory, Learning & Execution
**Estimasi: 5–7 hari**

#### 6.1 Memory & Learning
- [ ] Implementasi Multi-Armed Bandit UCB1 untuk specialist selection
- [ ] Fungsi `update_weights(specialist_id, reward)` setelah setiap trade close
- [ ] Implementasi exploration mode: setiap 5 trade, 1 trade untuk explore specialist lain
- [ ] Fungsi `get_best_specialist(regime)` menggunakan UCB1 score

#### 6.2 Order Execution
- [ ] Implementasi `OrderManager` class
- [ ] Fungsi `send_order(symbol, direction, lot, sl, tp, specialist_id)` → order ticket
- [ ] Implementasi order comment dengan format: `specialist_id|regime|confidence`
- [ ] Monitor semua open orders setiap 1 menit
- [ ] Saat order close: ambil result, update memory, trigger performance monitor
- [ ] Handle error: requote, no connection, insufficient margin

---

### PHASE 7 — Monitoring, Alerting & Integration Testing
**Estimasi: 5–7 hari**

#### 7.1 Telegram Alerting
- [ ] Setup Telegram Bot dengan BotFather
- [ ] Alert: System startup / shutdown
- [ ] Alert: Specialist APPROVED naik (dengan ringkasan performa)
- [ ] Alert: Specialist masuk status WARNING atau SUSPENDED
- [ ] Alert: Daily drawdown limit tercapai (STOP trading hari ini)
- [ ] Alert: Total drawdown limit kritis (EMERGENCY STOP)
- [ ] Report harian: total PnL, WinRate hari ini, jumlah specialist aktif

#### 7.2 Integration Testing
- [ ] End-to-end test: dari ambil data candle → feature → regime → specialist → order
- [ ] Test koneksi terputus dan reconnect
- [ ] Test daily drawdown limit benar-benar stop trading
- [ ] Test rotasi slot tidak mengganggu posisi yang sedang open
- [ ] Test Fast-Kill protocol eliminasi specialist dengan benar di trade ke-3 loss beruntun
- [ ] Performance test: sistem dapat menyelesaikan 1 siklus dalam < 5 detik per candle

---

### PHASE 8 — Paper Trading & Optimization
**Estimasi: 14–30 hari**

- [ ] Jalankan sistem di akun MT5 DEMO selama minimal 2 minggu tanpa intervensi
- [ ] Monitor: apakah Fast-Kill benar-benar mengeliminasi dengan cepat?
- [ ] Monitor: apakah Slot Manager rotasi dengan benar?
- [ ] Monitor: apakah Regime Detector akurat di live market?
- [ ] Catat semua anomali dan bug di dokumen terpisah
- [ ] Setelah 2 minggu: analisis trade log, fine-tune threshold jika diperlukan
- [ ] ✅ **GATE:** WinRate sistem keseluruhan > 58% dan Profit Factor > 1.3 di demo
- [ ] ✅ **GATE:** Tidak ada crash atau error fatal dalam 7 hari berturut-turut
- [ ] Baru setelah 2 gate ini lolos: pertimbangkan live trading dengan lot minimum

---

## 6. Konfigurasi Sistem

### 6.1 settings.py — Semua Parameter

```python
class TradingConfig:
    # ── MT5 ──────────────────────────────────────
    MT5_LOGIN         = 'YOUR_LOGIN'
    MT5_PASSWORD      = 'YOUR_PASSWORD'
    MT5_SERVER        = 'YOUR_BROKER_SERVER'

    # ── Symbol & Timeframe ────────────────────────
    SYMBOLS           = ['EURUSD', 'GBPUSD']
    PRIMARY_TF        = 'M5'    # Untuk regime detection
    ENTRY_TF          = 'M5'   # Untuk entry timing

    # ── Risk Management ───────────────────────────
    RISK_PER_TRADE    = 0.01    # 1% balance per trade
    MAX_DAILY_DD      = 0.03    # 3% max drawdown per hari
    MAX_TOTAL_DD      = 0.10    # 10% max drawdown total
    SL_ATR_MULT       = 1.5     # SL = 1.5 × ATR
    TP_ATR_MULT       = 2.0     # TP = 2.0 × ATR

    # ── Slot Management ───────────────────────────
    TOTAL_SLOTS       = 50
    APPROVED_SLOTS    = 30
    PROBATION_SLOTS   = 15
    BUFFER_SLOTS      = 5

    # ── Seleksi Threshold ─────────────────────────
    MIN_BACKTEST_WR   = 0.60    # Minimum WR lolos pre-filter
    MIN_PROFIT_FACTOR = 1.50    # Minimum PF lolos pre-filter
    MIN_APPROVED_WR   = 0.60    # Minimum WR untuk APPROVED
    WARNING_WR        = 0.70    # Trigger WARNING jika di bawah ini
    SUSPEND_WR        = 0.60    # Trigger SUSPENDED jika di bawah ini

    # ── Regime Detector ───────────────────────────
    REGIME_CONFIDENCE_MIN = 0.60  # Di bawah ini = UNCERTAIN
    REGIME_UPDATE_TF      = 'H1'  # Update regime tiap candle H1

    # ── Learning ──────────────────────────────────
    EXPLORE_RATIO     = 0.20    # 20% waktu untuk explorasi
    LOSS_STREAK_LIMIT = 3       # Loss beruntun sebelum cooldown
    COOLDOWN_MINUTES  = 60      # Cooldown setelah loss streak

    # ── Telegram ──────────────────────────────────
    TELEGRAM_TOKEN    = 'YOUR_BOT_TOKEN'
    TELEGRAM_CHAT_ID  = 'YOUR_CHAT_ID'
```

---

## 7. Acceptance Criteria & Definition of Done

### 7.1 Per Phase

| Phase | Kriteria Selesai |
|---|---|
| Phase 1 | MT5 terhubung, data candle berhasil diambil, semua fitur terhitung dengan benar (verified unit test) |
| Phase 2 | Regime detector akurasi > 70% di out-of-sample data, UNCERTAIN state berfungsi |
| Phase 3 | Backtest engine menghasilkan output yang benar, specialist trainer bisa train & predict |
| Phase 4 | Fast-Kill mengeliminasi specialist buruk di checkpoint yang tepat (verified di test case) |
| Phase 5 | Slot management tidak pernah melebihi 50 slot, Risk Manager hitung lot size dengan benar |
| Phase 6 | Order masuk ke MT5 dengan SL/TP yang benar, result di-capture dan update memory |
| Phase 7 | Semua alert Telegram terkirim, integration test end-to-end pass |
| Phase 8 | WR > 58% + PF > 1.3 di demo selama min. 2 minggu, tidak ada crash 7 hari berturut |

### 7.2 Non-Functional Requirements

| Requirement | Target |
|---|---|
| Latency per siklus | < 5 detik dari candle close hingga order terkirim |
| Uptime | > 99% selama jam trading aktif (Senin–Jumat) |
| Recovery dari crash | Auto-restart dalam < 60 detik, state restored dari database |
| Log retention | Simpan semua log minimal 90 hari |
| Data backup | Database di-backup otomatis setiap hari |

---

## 8. Risiko & Mitigasi

| Risiko | Probabilitas | Dampak | Mitigasi |
|---|---|---|---|
| Regime detector salah label | Sedang | Tinggi | Validasi visual + minimal 70% akurasi sebelum go-live |
| Overfitting specialist model | Tinggi | Tinggi | Wajib walk-forward backtest + Monte Carlo |
| Koneksi MT5 terputus saat ada order | Rendah | Tinggi | Auto-reconnect + order monitoring post-reconnect |
| Market gap / news spike ekstrem | Rendah | Sangat Tinggi | Max DD limit + pause semua trading saat news besar |
| Terlalu banyak candidate strategi | Tinggi | Sedang | Pre-filter ketat + antrian berprioritas (best backtest duluan) |
| Database corrupt | Sangat Rendah | Tinggi | Daily backup + WAL mode SQLite |

---

*PRD v1.0.0 | Adaptive Forex Trading Bot MT5*
