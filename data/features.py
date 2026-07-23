"""
data/features.py
Feature engineering — semua indikator teknikal untuk sistem trading.
Mengimplementasikan semua fitur sesuai PRD Section 3.1.

Kategori:
  - Trend: EMA(10/20/50/200), ADX, MACD
  - Momentum: RSI(14), Stochastic(%K %D), CCI
  - Volatilitas: ATR(14), Bollinger Band Width, Historical Volatility
  - Volume: OBV, Volume SMA, Volume Ratio
  - Candle Pattern: Doji, Engulfing, Hammer, Pin Bar (binary flag)
  - Market Structure: Swing High/Low, HH/LL detection
"""

import numpy as np
import pandas as pd
import ta
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# TREND INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA(10/20/50/200), ADX(14), MACD Line & Signal.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA
    df["ema_10"] = ta.trend.EMAIndicator(close, window=10).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()

    # EMA Slope (perbedaan EMA_50 saat ini vs 5 candle lalu, normalized)
    df["ema_slope"] = (df["ema_50"] - df["ema_50"].shift(5)) / df["ema_50"].shift(5) * 100

    # ADX
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_indicator.adx()
    df["adx_pos"] = adx_indicator.adx_pos()  # +DI
    df["adx_neg"] = adx_indicator.adx_neg()  # -DI

    # MACD
    macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_hist"] = macd_indicator.macd_diff()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI(14), Stochastic(%K %D), CCI(20).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # CCI
    df["cci"] = ta.trend.CCIIndicator(high, low, close, window=20).cci()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    ATR(14), Bollinger Band Width, Historical Volatility (20-period).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr_indicator.average_true_range()

    # ATR ratio = ATR / Close (normalized volatility)
    df["atr_ratio"] = df["atr"] / close

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]  # Relative width
    df["bb_pct"] = bb.bollinger_pband()  # Price position dalam band (0-1)

    # Historical Volatility (20-period log return std, annualized)
    log_returns = np.log(close / close.shift(1))
    df["hist_vol"] = log_returns.rolling(window=20).std() * np.sqrt(252)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBV, Volume SMA(20), Volume Ratio.
    Untuk XAUUSD gunakan tick_volume karena real_volume mungkin 0.
    """
    close = df["close"]

    # Gunakan tick_volume sebagai proxy volume
    volume = df["tick_volume"].replace(0, np.nan).ffill()

    # OBV
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # Volume SMA(20)
    df["volume_sma"] = volume.rolling(window=20).mean()

    # Volume Ratio = volume saat ini vs SMA(20)
    df["volume_ratio"] = volume / df["volume_sma"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE PATTERN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _body_size(df: pd.DataFrame) -> pd.Series:
    return abs(df["close"] - df["open"])

def _range_size(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]

def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)

def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deteksi candle pattern: Doji, Bullish/Bearish Engulfing, Hammer, Pin Bar.
    Semua output adalah binary flag (0/1).
    """
    body = _body_size(df)
    candle_range = _range_size(df)
    upper_shadow = _upper_shadow(df)
    lower_shadow = _lower_shadow(df)

    # Hindari division by zero
    candle_range_safe = candle_range.replace(0, np.nan)

    # ── Doji ──────────────────────────────────────────────────────────────────
    # Body sangat kecil dibanding range total (< 10% dari range)
    df["pattern_doji"] = (body / candle_range_safe < 0.1).astype(int)

    # ── Hammer ────────────────────────────────────────────────────────────────
    # Lower shadow panjang (>= 2× body), upper shadow kecil, close > open (bullish)
    hammer_cond = (
        (lower_shadow >= 2 * body) &
        (upper_shadow <= 0.3 * body) &
        (df["close"] > df["open"])
    )
    df["pattern_hammer"] = hammer_cond.astype(int)

    # ── Pin Bar (Bearish) ─────────────────────────────────────────────────────
    # Upper shadow panjang (>= 2× body), lower shadow kecil, close < open
    pin_bar_cond = (
        (upper_shadow >= 2 * body) &
        (lower_shadow <= 0.3 * body) &
        (df["close"] < df["open"])
    )
    df["pattern_pin_bar"] = pin_bar_cond.astype(int)

    # ── Bullish Engulfing ─────────────────────────────────────────────────────
    # Candle bullish yang bodynya "menelan" body candle bearish sebelumnya
    prev_bearish = df["close"].shift(1) < df["open"].shift(1)
    curr_bullish = df["close"] > df["open"]
    engulf_bull = (
        curr_bullish &
        prev_bearish &
        (df["open"] <= df["close"].shift(1)) &
        (df["close"] >= df["open"].shift(1))
    )
    df["pattern_engulf_bull"] = engulf_bull.astype(int)

    # ── Bearish Engulfing ─────────────────────────────────────────────────────
    prev_bullish = df["close"].shift(1) > df["open"].shift(1)
    curr_bearish = df["close"] < df["open"]
    engulf_bear = (
        curr_bearish &
        prev_bullish &
        (df["open"] >= df["close"].shift(1)) &
        (df["close"] <= df["open"].shift(1))
    )
    df["pattern_engulf_bear"] = engulf_bear.astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MARKET STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def _find_swing_highs(high: pd.Series, window: int = 5) -> pd.Series:
    """
    Deteksi swing high: titik high yang lebih tinggi dari N candle di kiri dan kanan.
    """
    swing_high = pd.Series(0, index=high.index)
    for i in range(window, len(high) - window):
        center = high.iloc[i]
        left = high.iloc[i - window:i]
        right = high.iloc[i + 1:i + window + 1]
        if center > left.max() and center > right.max():
            swing_high.iloc[i] = 1
    return swing_high


def _find_swing_lows(low: pd.Series, window: int = 5) -> pd.Series:
    """
    Deteksi swing low: titik low yang lebih rendah dari N candle di kiri dan kanan.
    """
    swing_low = pd.Series(0, index=low.index)
    for i in range(window, len(low) - window):
        center = low.iloc[i]
        left = low.iloc[i - window:i]
        right = low.iloc[i + 1:i + window + 1]
        if center < left.min() and center < right.min():
            swing_low.iloc[i] = 1
    return swing_low


def add_market_structure(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Deteksi:
    - Swing High / Swing Low
    - Higher High (HH): swing high lebih tinggi dari swing high sebelumnya
    - Lower Low (LL): swing low lebih rendah dari swing low sebelumnya
    - Trend direction berdasarkan HH/HL atau LH/LL
    """
    high = df["high"]
    low = df["low"]

    # Swing points
    df["swing_high"] = _find_swing_highs(high, window)
    df["swing_low"] = _find_swing_lows(low, window)

    # Higher High / Lower Low detection
    # Bandingkan swing high saat ini dengan swing high sebelumnya
    swing_high_vals = df.loc[df["swing_high"] == 1, "high"]
    swing_low_vals = df.loc[df["swing_low"] == 1, "low"]

    # HH: swing high saat ini > swing high sebelumnya
    df["higher_high"] = 0
    if len(swing_high_vals) >= 2:
        for idx in range(1, len(swing_high_vals)):
            if swing_high_vals.iloc[idx] > swing_high_vals.iloc[idx - 1]:
                df.at[swing_high_vals.index[idx], "higher_high"] = 1

    # LL: swing low saat ini < swing low sebelumnya
    df["lower_low"] = 0
    if len(swing_low_vals) >= 2:
        for idx in range(1, len(swing_low_vals)):
            if swing_low_vals.iloc[idx] < swing_low_vals.iloc[idx - 1]:
                df.at[swing_low_vals.index[idx], "lower_low"] = 1

    # Simple trend direction: 1 = Uptrend, -1 = Downtrend, 0 = Sideways
    # Berdasarkan posisi close terhadap EMA 50 dan EMA 200
    df["trend_direction"] = 0
    if "ema_50" in df.columns and "ema_200" in df.columns:
        df.loc[
            (df["close"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]),
            "trend_direction"
        ] = 1
        df.loc[
            (df["close"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]),
            "trend_direction"
        ] = -1

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_features(df: pd.DataFrame, swing_window: int = 5) -> pd.DataFrame:
    """
    Main function: tambahkan semua fitur ke DataFrame OHLCV.

    Args:
        df: DataFrame dengan kolom open, high, low, close, tick_volume
        swing_window: Window untuk deteksi swing high/low

    Returns:
        DataFrame yang sudah diperkaya dengan semua fitur.
        Baris awal yang NaN (akibat rolling window) di-drop otomatis.
    """
    if df is None or len(df) < 200:
        logger.warning(f"[Features] Data terlalu sedikit: {len(df) if df is not None else 0} baris. Min: 200.")
        return df

    logger.debug(f"[Features] Menghitung fitur untuk {len(df)} candle...")

    df = df.copy()

    # Tambahkan semua indikator secara berurutan
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    df = add_candle_patterns(df)
    df = add_market_structure(df, window=swing_window)

    # Drop baris dengan NaN (akibat rolling window EMA200, dll)
    initial_len = len(df)
    df.dropna(subset=["ema_200", "rsi", "atr", "macd"], inplace=True)
    dropped = initial_len - len(df)

    if dropped > 0:
        logger.debug(f"[Features] Drop {dropped} baris NaN (warm-up period). Sisa: {len(df)} baris.")

    logger.debug(f"[Features] Selesai. Total fitur: {len(df.columns)} kolom.")
    return df


def get_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset fitur khusus untuk input ke Regime Detector (Master Brain).
    Sesuai PRD: [ADX, BB_Width, ATR_ratio, Volume_ratio, EMA_slope]
    """
    required = ["adx", "bb_width", "atr_ratio", "volume_ratio", "ema_slope"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.error(f"[Features] Kolom berikut tidak ditemukan untuk regime detection: {missing}")
        return pd.DataFrame()

    return df[required].copy()
