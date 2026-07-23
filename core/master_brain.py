"""
core/master_brain.py
Master Brain — Orchestrator utama untuk regime detection.
Interface antara Data Pipeline dan Specialist Pool.

Responsibilities:
  - Jalankan regime detection setiap candle H1 close
  - Simpan history regime ke database
  - Output: (regime, confidence) untuk dikonsumsi Specialist Pool
  - Handle UNCERTAIN state → sistem masuk mode HOLD
"""

from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from config.settings import settings
from core.memory import db
from data.pipeline import pipeline
from models.regime_detector import Regime, RegimeDetector, regime_detector


class MasterBrain:
    """
    Master Brain mengkoordinasikan deteksi regime dan
    menyediakan signal ke downstream components.
    """

    def __init__(self, detector: RegimeDetector = None):
        self._detector = detector or regime_detector
        self._last_regime: str = Regime.UNCERTAIN
        self._last_confidence: float = 0.0
        self._no_entry_mode: bool = False  # True saat UNCERTAIN

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Detect Regime
    # ─────────────────────────────────────────────────────────────────────────

    def detect_regime(
        self,
        symbol: str = None,
        timeframe: str = None,
        df: pd.DataFrame = None,
    ) -> Tuple[str, float]:
        """
        Deteksi regime pasar saat ini.

        Args:
            symbol: Nama symbol (default: XAUUSD dari settings)
            timeframe: Timeframe untuk deteksi (default: H1 dari settings)
            df: Opsional — gunakan DataFrame yang sudah ada (untuk testing)

        Returns:
            (regime_label, confidence_score)
            regime_label bisa berupa Regime.UNCERTAIN jika confidence < threshold
        """
        symbol = symbol or settings.SYMBOLS[0]
        timeframe = timeframe or settings.REGIME_UPDATE_TF

        # Ambil data jika tidak disediakan
        if df is None:
            df = pipeline.get_data(symbol=symbol, timeframe=timeframe, count=500)
            if df is None or df.empty:
                logger.error("[MasterBrain] Gagal ambil data untuk regime detection.")
                return Regime.UNCERTAIN, 0.0

        # Cek apakah model sudah di-train
        if not self._detector.is_trained:
            # Coba load dari disk
            if not self._detector.load():
                logger.warning(
                    "[MasterBrain] Regime Detector belum di-train! "
                    "Jalankan: python -m models.train_regime"
                )
                return Regime.UNCERTAIN, 0.0

        # Prediksi regime
        regime, confidence = self._detector.predict(df)

        # Update state internal
        self._last_regime = regime
        self._last_confidence = confidence
        self._no_entry_mode = (regime == Regime.UNCERTAIN)

        # Log dan simpan ke database
        if regime != Regime.UNCERTAIN:
            logger.info(
                f"[MasterBrain] Regime: {regime} "
                f"(confidence: {confidence:.2f}) | {symbol} {timeframe}"
            )
            db.log_regime(symbol, timeframe, regime, confidence)
        else:
            logger.info(
                f"[MasterBrain] UNCERTAIN (confidence: {confidence:.2f}) "
                f"→ HOLD mode aktif. Tidak ada entry baru."
            )

        return regime, confidence

    def initialize(self) -> bool:
        """
        Inisialisasi Master Brain: load model dari disk atau training dari data historis.
        Dipanggil saat sistem startup.
        """
        logger.info("[MasterBrain] Inisialisasi...")

        if self._detector.load():
            logger.success("[MasterBrain] Model Regime Detector berhasil dimuat.")
            return True

        logger.warning("[MasterBrain] Model belum ada. Perlu training terlebih dahulu.")
        logger.info("[MasterBrain] Jalankan: python -m models.train_regime")
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # State Accessors
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def current_regime(self) -> str:
        """Regime terakhir yang terdeteksi."""
        return self._last_regime

    @property
    def current_confidence(self) -> float:
        """Confidence score terakhir."""
        return self._last_confidence

    @property
    def is_hold_mode(self) -> bool:
        """True jika sistem dalam mode HOLD (UNCERTAIN regime)."""
        return self._no_entry_mode

    def get_status(self) -> dict:
        """Summary status Master Brain saat ini."""
        return {
            "regime": self._last_regime,
            "confidence": self._last_confidence,
            "hold_mode": self._no_entry_mode,
            "model_ready": self._detector.is_trained,
        }


# Singleton instance
master_brain = MasterBrain()
