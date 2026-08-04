"""
config/settings.py
Semua parameter konfigurasi sistem Adaptive Forex Trading Bot.
Mode: AGRESIF M5 — High Frequency, Tight Risk Control
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
    MT5_PATH: str = r"C:\Program Files\Valetax International MT5 Terminal\terminal64.exe"
    MT5_LOGIN: int = 0
    MT5_PASSWORD: str = "YOUR_PASSWORD"
    MT5_SERVER: str = "YOUR_BROKER_SERVER"

    # ── Symbol & Timeframe ────────────────────────────────────────────────────
    SYMBOLS: List[str] = ["XAUUSD.vxc"]
    PRIMARY_TF: str = "M5"
    ENTRY_TF: str = "M5"
    HISTORY_CANDLES: int = 5000

    # ── Risk Management — AGRESIF TAPI TERKONTROL ────────────────────────────
    RISK_PER_TRADE: float = 0.05     # 2% balance per trade (agresif)
    MAX_DAILY_DD: float = 1       # 6% max drawdown per hari — STOP jika tercapai
    MAX_TOTAL_DD: float = 1       # 15% max drawdown total — EMERGENCY STOP
    
    # ── SL/TP — RR 1:1.5 untuk M5 agresif ───────────────────────────────────
    # PENTING: TP harus selalu lebih besar dari SL
    # Dengan WR 60%: Expectancy = (0.6 × 1.5) - (0.4 × 1.0) = +0.5 per trade
    SL_ATR_MULT: float = 0.75         # SL ketat = 1.0 × ATR
    TP_ATR_MULT: float = 1.4         # TP = 1.5 × ATR → RR 1:1.5

    # ── Slot Management ───────────────────────────────────────────────────────
    TOTAL_SLOTS: int = 50
    APPROVED_SLOTS: int = 30
    PROBATION_SLOTS: int = 15
    BUFFER_SLOTS: int = 5
    MAX_CONCURRENT_TRADES: int = 30
    SLOT_ROTATION_HOURS: int = 6
    MAX_ROTATIONS_PER_DAY: int = 3

    # ── Seleksi & Fast-Kill Threshold ─────────────────────────────────────────
    MIN_BACKTEST_WR: float = 0.60
    MIN_PROFIT_FACTOR: float = 1.50
    MIN_MONTE_CARLO_PASS: float = 0.65
    MAX_BACKTEST_DD: float = 0.15
    MIN_REGIME_MATCH: float = 0.70
    MIN_BACKTEST_TRADES: int = 100

    # Fast-Kill checkpoint thresholds
    FASTKILL_TRADE3_CONDITION: str = "3_CONSECUTIVE_LOSS"
    FASTKILL_TRADE5_WR: float = 0.40
    FASTKILL_TRADE10_WR: float = 0.50
    FASTKILL_TRADE15_WR: float = 0.55
    FASTKILL_TRADE15_DD: float = 0.05
    FASTKILL_TRADE20_WR: float = 0.60
    FASTKILL_TRADE20_PF: float = 1.30
    APPROVED_MIN_WR: float = 0.60
    APPROVED_MIN_PF: float = 1.30

    # ── Performance Monitor ───────────────────────────────────────────────────
    WARNING_WR: float = 0.70
    SUSPEND_WR: float = 0.60
    SUSPEND_RECOVERY_HOURS: int = 48
    RECOVERY_WR: float = 0.75
    RECOVERY_TRADES: int = 10

    # ── Regime Detector ───────────────────────────────────────────────────────
    REGIME_CONFIDENCE_MIN: float = 0.45
    REGIME_UPDATE_TF: str = "M15"
    REGIME_HMM_STATES: int = 5
    REGIME_FEATURE_VECTOR: List[str] = [
        "adx", "bb_width", "atr_ratio", "volume_ratio", "ema_slope"
    ]

    # ── Composite Score Weights ───────────────────────────────────────────────
    SCORE_WEIGHT_WR: float = 0.4
    SCORE_WEIGHT_PF: float = 0.3
    SCORE_WEIGHT_RECENT: float = 0.3
    RECENT_SCORE_TRADES: int = 10

    # ── Learning (Multi-Armed Bandit) ─────────────────────────────────────────
    EXPLORE_RATIO: float = 0.20
    EXPLORE_EVERY_N_TRADES: int = 5
    AUTO_EXPLORE_AFTER_DAYS: int = 7

    # ── Loss Streak & Cooldown — AGRESIF ─────────────────────────────────────
    LOSS_STREAK_LIMIT: int = 3       # 3 loss beruntun → cooldown
    COOLDOWN_MINUTES: int = 15       # Cooldown 15 menit (lebih pendek untuk M5)

    # ── Database ──────────────────────────────────────────────────────────────
    DB_PATH: str = "forexmon.db"
    DB_BACKUP_DIR: str = "backups"
    PERFORMANCE_SNAPSHOT_HOURS: int = 6

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_DIR: str = "logs"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "90 days"
    LOG_LEVEL: str = "INFO"

    # ── Telegram Notifications ────────────────────────────────────────────────
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # ── Performance Targets ───────────────────────────────────────────────────
    TARGET_SYSTEM_WR: float = 0.58
    TARGET_SYSTEM_PF: float = 1.30

    # ── Monitoring Intervals — M5 AGRESIF ────────────────────────────────────
    ORDER_MONITOR_SECONDS: int = 30  # Cek posisi setiap 30 detik (lebih sering)
    PERFORMANCE_MONITOR_HOURS: int = 1
    CYCLE_MAX_SECONDS: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("APPROVED_SLOTS", "PROBATION_SLOTS", "BUFFER_SLOTS")
    @classmethod
    def validate_slot_sum(cls, v: int) -> int:
        return v


# Singleton instance — import ini di semua module
settings = TradingConfig()