"""
config/settings.py
Semua parameter konfigurasi sistem Adaptive Forex Trading Bot.
Adaptasi dari PRD Section 6.1 — Primary Symbol: XAUUSD
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import List


class TradingConfig(BaseSettings):
    """
    Seluruh konfigurasi sistem. Nilai bisa di-override via environment variable
    atau file .env (jika ada).
    """

    # ── MT5 Connection ────────────────────────────────────────────────────────
    MT5_PATH: str = r"C:\Program Files\FBS MetaTrader 5\terminal64.exe"
    MT5_LOGIN: int = 106353764               # ← Nomor akun MT5
    MT5_PASSWORD: str = "zpa[3;5X"   # ← ISI dengan password akun MT5
    MT5_SERVER: str = "FBS-Demo"  # ← ISI dengan nama server broker

    # ── Symbol & Timeframe ────────────────────────────────────────────────────
    SYMBOLS: List[str] = ["XAUUSD"]
    PRIMARY_TF: str = "M5"    # Untuk regime detection
    ENTRY_TF: str = "M5"     # Untuk entry timing
    HISTORY_CANDLES: int = 5000  # Minimal 5000 candle untuk training

    # ── Risk Management ───────────────────────────────────────────────────────
    RISK_PER_TRADE: float = 0.01     # 1% balance per trade
    MAX_DAILY_DD: float = 0.03       # 3% max drawdown per hari
    MAX_TOTAL_DD: float = 0.10       # 10% max drawdown total
    SL_ATR_MULT: float = 1.5         # SL = 1.5 × ATR(14)
    TP_ATR_MULT: float = 2.0         # TP = 2.0 × ATR(14)

    # ── Slot Management ───────────────────────────────────────────────────────
    TOTAL_SLOTS: int = 50
    APPROVED_SLOTS: int = 30
    PROBATION_SLOTS: int = 15
    BUFFER_SLOTS: int = 5
    MAX_CONCURRENT_TRADES: int = 30
    SLOT_ROTATION_HOURS: int = 6     # Re-ranking setiap 6 jam
    MAX_ROTATIONS_PER_DAY: int = 3

    # ── Seleksi & Fast-Kill Threshold ─────────────────────────────────────────
    MIN_BACKTEST_WR: float = 0.60       # Minimum WR lolos pre-filter
    MIN_PROFIT_FACTOR: float = 1.50     # Minimum PF lolos pre-filter
    MIN_MONTE_CARLO_PASS: float = 0.65  # Profit di >65% skenario Monte Carlo
    MAX_BACKTEST_DD: float = 0.15       # Max drawdown backtest 15%
    MIN_REGIME_MATCH: float = 0.70      # Regime match score minimum
    MIN_BACKTEST_TRADES: int = 100      # Minimum trade di backtest

    # Fast-Kill checkpoint thresholds
    FASTKILL_TRADE3_CONDITION: str = "3_CONSECUTIVE_LOSS"
    FASTKILL_TRADE5_WR: float = 0.40
    FASTKILL_TRADE10_WR: float = 0.50
    FASTKILL_TRADE15_WR: float = 0.55
    FASTKILL_TRADE15_DD: float = 0.05
    FASTKILL_TRADE20_WR: float = 0.60
    FASTKILL_TRADE20_PF: float = 1.30
    APPROVED_MIN_WR: float = 0.60       # WR minimum untuk APPROVED
    APPROVED_MIN_PF: float = 1.30

    # ── Performance Monitor (APPROVED specialist) ────────────────────────────
    WARNING_WR: float = 0.70    # Trigger WARNING jika di bawah ini
    SUSPEND_WR: float = 0.60    # Trigger SUSPENDED jika di bawah ini
    SUSPEND_RECOVERY_HOURS: int = 48   # Jam sebelum SUSPENDED → ELIMINATED
    RECOVERY_WR: float = 0.75          # WR untuk recovery ke APPROVED
    RECOVERY_TRADES: int = 10          # Jumlah trade untuk konfirmasi recovery

    # ── Regime Detector ───────────────────────────────────────────────────────
    REGIME_CONFIDENCE_MIN: float = 0.60   # Di bawah ini = UNCERTAIN
    REGIME_UPDATE_TF: str = "H1"          # Update regime tiap candle H1
    REGIME_HMM_STATES: int = 5            # 5 hidden states untuk HMM
    REGIME_FEATURE_VECTOR: List[str] = [
        "adx", "bb_width", "atr_ratio", "volume_ratio", "ema_slope"
    ]

    # ── Composite Score Weights ───────────────────────────────────────────────
    SCORE_WEIGHT_WR: float = 0.4
    SCORE_WEIGHT_PF: float = 0.3
    SCORE_WEIGHT_RECENT: float = 0.3
    RECENT_SCORE_TRADES: int = 10     # WinRate dari N trade terakhir

    # ── Learning (Multi-Armed Bandit) ─────────────────────────────────────────
    EXPLORE_RATIO: float = 0.20        # 20% waktu untuk explorasi
    EXPLORE_EVERY_N_TRADES: int = 5    # Setiap 5 trade, 1 untuk explore
    AUTO_EXPLORE_AFTER_DAYS: int = 7   # Naikkan explore rate jika tidak ada specialist baru

    # ── Loss Streak & Cooldown ────────────────────────────────────────────────
    LOSS_STREAK_LIMIT: int = 3         # Loss beruntun sebelum cooldown
    COOLDOWN_MINUTES: int = 60         # Cooldown setelah loss streak

    # ── Database ──────────────────────────────────────────────────────────────
    DB_PATH: str = "forexmon.db"
    DB_BACKUP_DIR: str = "backups"
    PERFORMANCE_SNAPSHOT_HOURS: int = 6  # Snapshot setiap 6 jam

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_DIR: str = "logs"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "90 days"
    LOG_LEVEL: str = "INFO"

    # ── Telegram Notifications ────────────────────────────────────────────────
    TELEGRAM_TOKEN: str = "YOUR_BOT_TOKEN"   # ← ISI dengan token Telegram bot
    TELEGRAM_CHAT_ID: str = "YOUR_CHAT_ID"   # ← ISI dengan chat ID

    # ── Performance Targets (untuk monitoring) ────────────────────────────────
    TARGET_SYSTEM_WR: float = 0.58      # Target WR keseluruhan sistem
    TARGET_SYSTEM_PF: float = 1.30      # Target PF keseluruhan sistem

    # ── Monitoring Intervals ──────────────────────────────────────────────────
    ORDER_MONITOR_SECONDS: int = 60     # Monitor open orders setiap 1 menit
    PERFORMANCE_MONITOR_HOURS: int = 1  # Cek performa setiap 1 jam
    CYCLE_MAX_SECONDS: int = 5          # Max waktu 1 siklus per candle

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("APPROVED_SLOTS", "PROBATION_SLOTS", "BUFFER_SLOTS")
    @classmethod
    def validate_slot_sum(cls, v: int) -> int:
        return v


# Singleton instance — import ini di semua module
settings = TradingConfig()
