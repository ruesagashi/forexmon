"""
data/pipeline.py
Data pipeline: ambil candle dari MT5, jalankan feature engineering,
return DataFrame siap pakai untuk Master Brain & Specialist.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import settings
from execution.mt5_connector import connector
from data.features import get_features, get_regime_features


class DataPipeline:
    """
    Pipeline tunggal untuk:
    1. Ambil data OHLCV dari MT5
    2. Hitung semua fitur teknikal
    3. Return DataFrame siap digunakan komponen lain
    """

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def get_data(
        self,
        symbol: str,
        timeframe: str,
        count: int = None,
        use_cache: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Ambil data lengkap (OHLCV + semua fitur) untuk symbol dan timeframe tertentu.

        Args:
            symbol: Nama symbol (contoh: 'XAUUSD')
            timeframe: Timeframe string (contoh: 'H1', 'M15')
            count: Jumlah candle yang diminta (default dari settings)
            use_cache: Gunakan cache jika ada (untuk testing)

        Returns:
            DataFrame dengan OHLCV + semua fitur teknikal, atau None jika gagal.
        """
        count = count or settings.HISTORY_CANDLES
        cache_key = f"{symbol}_{timeframe}"

        if use_cache and cache_key in self._cache:
            logger.debug(f"[Pipeline] Menggunakan cache untuk {cache_key}")
            return self._cache[cache_key]

        logger.info(f"[Pipeline] Mengambil {count} candle {symbol} {timeframe}...")

        # Step 1: Ambil raw candle dari MT5
        df_raw = connector.get_candles(symbol=symbol, timeframe=timeframe, count=count)
        if df_raw is None:
            logger.error(f"[Pipeline] Gagal ambil candle {symbol} {timeframe}.")
            return None

        # Step 2: Hitung semua fitur
        df_features = get_features(df_raw)
        if df_features is None or df_features.empty:
            logger.error(f"[Pipeline] Feature engineering gagal untuk {symbol} {timeframe}.")
            return None

        if use_cache:
            self._cache[cache_key] = df_features

        logger.info(
            f"[Pipeline] OK — {symbol} {timeframe}: "
            f"{len(df_features)} candle, {len(df_features.columns)} fitur. "
            f"Range: {df_features.index[0]} → {df_features.index[-1]}"
        )
        return df_features

    def get_latest_features(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[pd.Series]:
        """
        Ambil fitur dari candle terbaru saja (untuk prediksi real-time).

        Returns:
            pd.Series dengan semua fitur dari candle terakhir.
        """
        df = self.get_data(symbol=symbol, timeframe=timeframe, count=300)
        if df is None or df.empty:
            return None
        return df.iloc[-1]

    def get_regime_input(
        self,
        symbol: str,
        timeframe: str = None,
    ) -> Optional[pd.DataFrame]:
        """
        Ambil subset fitur khusus untuk Master Brain (Regime Detector).
        Timeframe default: H1 (sesuai PRD).
        """
        tf = timeframe or settings.REGIME_UPDATE_TF
        df = self.get_data(symbol=symbol, timeframe=tf)
        if df is None:
            return None
        return get_regime_features(df)

    def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: list[str] = None,
        count: int = 300,
    ) -> dict[str, pd.DataFrame]:
        """
        Ambil data untuk multiple timeframe sekaligus.
        Berguna untuk konfirmasi sinyal antar timeframe.

        Returns:
            Dict {timeframe: DataFrame}
        """
        timeframes = timeframes or [settings.PRIMARY_TF, settings.ENTRY_TF]
        result = {}

        for tf in timeframes:
            df = self.get_data(symbol=symbol, timeframe=tf, count=count)
            if df is not None:
                result[tf] = df
            else:
                logger.warning(f"[Pipeline] Gagal ambil data {symbol} {tf}.")

        return result

    def clear_cache(self) -> None:
        """Bersihkan cache data."""
        self._cache.clear()
        logger.debug("[Pipeline] Cache dibersihkan.")


# Singleton instance
pipeline = DataPipeline()
