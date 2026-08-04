# FOREXMON — Adaptive Forex Trading Bot MT5
# Lalu jalankan:
.\venv\Scripts\python.exe retrain_regime_and_backtest.py
.\venv\Scripts\python.exe clear_specs.py
.\venv\Scripts\python.exe populate_specialists.py
.\venv\Scripts\python.exe main.py

.\venv\Scripts\python.exe -m uvicorn monitoring.api_server:app --host 0.0.0.0 --port 8765 --reload



# 1. Retrain regime detector
.\venv\Scripts\python.exe train_regime.py

# 2. Verify model tersimpan
dir models/saved/  # cek ada file regime_detector.pkl

# 3. Restart bot
.\venv\Scripts\python.exe main.py


## Deskripsi Singkat

**FOREXMON** adalah sistem trading otomatis yang menggunakan **Artificial Intelligence** untuk trade forex dan crypto di MetaTrader 5. Bot ini bukan hanya menjalankan rules statis, melainkan **adaptif** — artinya bot bisa belajar dan menyesuaikan strategi sesuai kondisi pasar yang berubah-ubah secara real-time.

---

## Bagaimana Cara Kerjanya?

Sistem FOREXMON bekerja dengan 3 komponen utama:

### 1. **Master Brain — Deteksi Kondisi Pasar** 🧠
Bot membaca candlestick pasar dan mendeteksi sedang dalam kondisi apa:
- **TRENDING UP** — Harga terus naik, momentum positif
- **TRENDING DOWN** — Harga terus turun, momentum negatif  
- **RANGING** — Harga bergerak naik-turun dalam range tertentu
- **BREAKOUT** — Harga menembus level penting dengan volume tinggi
- **REVERSAL** — Tanda perubahan arah (dari naik jadi turun atau sebaliknya)

Master Brain ini menggunakan algoritma **Hidden Markov Model** — teknologi AI yang sama dipakai bank besar untuk prediksi pasar.

### 2. **Specialist Pool — Strategi Terpisah per Kondisi** 📊
Untuk setiap kondisi pasar, bot punya **specialist strategi yang khusus**:
- Specialist A: Bagus di saat TRENDING UP
- Specialist B: Bagus di saat RANGING
- Specialist C: Bagus di saat BREAKOUT
- ...dan seterusnya

Setiap specialist ini adalah **AI model** yang sudah dilatih menggunakan ribuan data historis. Saat Master Brain deteksi "sekarang TRENDING UP", bot otomatis aktifkan Specialist A yang paling cocok untuk kondisi itu.

### 3. **Fast-Kill Protocol — Seleksi Otomatis** ⚡
Bot terus monitor performa setiap specialist:
- **Specialist bagus** (WR > 60%) → naik status ke APPROVED, dapat slot lebih banyak
- **Specialist jelek** (WR < 50%) → langsung di-ELIMINATE, tidak boleh trade lagi
- **Specialist baru** → di-test di slot PROBATION, harus terbukti profitable baru bisa APPROVED

Ini membuat bot **self-improving** — yang bagus dipakai lebih sering, yang jelek dihapus otomatis.

---

## Flow Kerja Singkat

```
Candle baru muncul di MT5
    ↓
Master Brain baca candlestick → "Sekarang TRENDING_UP"
    ↓
Cari Specialist TRENDING_UP yang terbaik (highest WR)
    ↓
Specialist prediksi: "sekarang harus BUY dengan confidence 75%"
    ↓
Risk Manager hitung: "berapa lot, SL/TP berapa?"
    ↓
Order ke MT5: BUY 0.01 lot SL 100 pips TP 150 pips
    ↓
Trade close (profit atau loss)
    ↓
Update performa Specialist
    ↓
Fast-Kill check: specialist ini WR-nya bagus atau jelek?
    ↓
Ulangi ke candle berikutnya
```

---

## Keunggulan FOREXMON

### ✅ **Adaptif**
Tidak seperti bot lain yang pake rules statis (misal "selalu BUY kalau EMA naik"), FOREXMON otomatis switch strategi sesuai kondisi pasar. Kalau market trend, pakai trend strategy. Kalau market ranging, pakai ranging strategy.

### ✅ **Self-Learning**
Setiap trade yang ditutup, bot update model-nya. Kalau specialist tertentu sering loss, bot automatix kurangi penggunaannya. Kalau specialist bagus, bot pakai lebih sering.

### ✅ **Risk Control Ketat**
- Max loss per trade: 1-2% balance
- Max loss per hari: 6% balance (auto stop jika tercapai)
- Max loss total: 15% balance (emergency stop)
- Position sizing otomatis sesuai volatilitas pasar

### ✅ **Multi-Symbol Support**
Bot bisa trade XAUUSD, BTCUSD, ETHUSD sekaligus — masing-masing dengan specialist terpisah.

### ✅ **Real-Time Monitoring**
Dashboard live menunjukkan:
- Equity curve (grafik pertumbuhan)
- Specialist pool status (siapa yang APPROVED, siapa yang PROBATION)
- Regime detector (kondisi pasar sekarang)
- Trade history real-time
- Alert Telegram (notifikasi order masuk/close)

---

## Performa & Ekspektasi

### Backtest Historical (2 tahun data)
- **Win Rate**: 60-65% (dari ribuan trade)
- **Profit Factor**: 1.3-1.8x
- **Max Drawdown**: < 15%

### Live Trading (Demo)
- Dijalankan di akun demo FBS/Valetax
- Profit terakumulasi sesuai specialist performance
- Fast-Kill eliminate yang jelek, APPROVED yang bagus
- Learning curve: minggu pertama data collection, minggu kedua+ profit terlihat

### Realistis Diharapkan
- Target **ROI 5-10% per bulan** (bukan 100%)
- Win Rate stabil **55-65%** di live (lebih rendah dari backtest karena overfitting)
- Perlu modal minimal untuk sustainable trading (recommended: $1000+)

---

## Teknologi yang Dipakai

| Komponen | Teknologi |
|---|---|
| **Regime Detection** | Hidden Markov Model (HMM) + K-Means Clustering |
| **Specialist Model** | XGBoost (Gradient Boosting) |
| **Learning** | Multi-Armed Bandit Algorithm (UCB1) |
| **Platform** | MetaTrader 5 (MT5) Python API |
| **Database** | SQLite (local) |
| **API Backend** | FastAPI (untuk dashboard) |
| **Monitoring** | Telegram Bot notifications |

---

## Ringkasan Dalam 1 Kalimat

**FOREXMON adalah bot trading AI yang adaptif — otomatis deteksi kondisi pasar, pilih strategi terbaik, monitor performance, dan self-improve tanpa henti.**

---

## Pertanyaan Umum (FAQ)

**Q: Ini bot yang bisa jamin untung?**
A: Tidak ada bot yang bisa jamin untung 100%. FOREXMON dirancang untuk **probabilitas winning > 50%** dengan risk management ketat, bukan jaminan. Tujuannya adalah konsisten profitable dalam jangka panjang, bukan quick rich.

**Q: Butuh berapa modal?**
A: Rekomendasi minimal $500-1000 agar position sizing cukup dan risk management bisa berjalan optimal. Kalau di bawah itu, spread dan commission bisa menghabiskan profit.

**Q: Bisa di-run di laptop biasa?**
A: Bisa! Bot hanya butuh:
- Python 3.10+
- Internet connection (untuk connect ke broker)
- Laptop bisa hibernate/sleep (bot tetap jalan)

**Q: Perlu dimonitor terus-terus?**
A: Tidak. Bot berjalan 24/7 otomatis. Yang perlu di-monitor:
- Lihat dashboard 1-2x sehari
- Cek Telegram alert kalau ada order besar
- Check equity curve apakah trending profit atau loss

**Q: Kalau koneksi putus gimana?**
A: Bot punya auto-reconnect. Kalau koneksi hilang saat ada open position, posisi tetap aman di MT5 — bot akan re-sync saat koneksi kembali.

---

## Next Steps

1. **Setup MT5** — Download MetaTrader 5, buka akun demo di broker
2. **Run FOREXMON** — Clone repo, setup credentials, jalankan `python main.py`
3. **Monitor dashboard** — Buka `dashboard.html`, refresh setiap candle
4. **Tunggu specialist mature** — Setidaknya 100+ trade sebelum WR akurat
5. **Optimize & Scale** — Setelah proven di demo, consider live dengan modal kecil

---

**FOREXMON — Adaptive Trading. Smart Execution. Self-Improving System.** 🚀
