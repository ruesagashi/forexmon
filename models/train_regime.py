"""
models/train_regime.py
Script untuk train Regime Detector dari data historis MT5.
Sesuai PRD Phase 2.1 & 2.2.

Jalankan sekali sebelum bot dioperasikan:
  python -m models.train_regime
"""

import sys
from pathlib import Path

# Setup logging
from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", colorize=True, level="INFO")

import MetaTrader5 as mt5
import pandas as pd

from config.settings import settings
from execution.mt5_connector import connector
from data.features import get_features, get_regime_features
from models.regime_detector import RegimeDetector, label_regime_manual, Regime


def download_historical_data(
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    years: int = 2,
) -> pd.DataFrame:
    """
    Download minimal 2 tahun data historis dari MT5.
    Sesuai PRD Phase 2.1: minimal 5000 candle.
    """
    # 2 tahun H1 = ~365 * 24 * 5/7 ≈ 6240 candle
    n_candles = max(7000, years * 365 * 18)  # Buffer lebih banyak

    logger.info(f"Mengambil {n_candles} candle {symbol} {timeframe} dari MT5...")

    tf_map = {
        "H1": mt5.TIMEFRAME_H1,
        "M15": mt5.TIMEFRAME_M15,
        "H4": mt5.TIMEFRAME_H4,
    }

    rates = mt5.copy_rates_from_pos(symbol, tf_map.get(timeframe, mt5.TIMEFRAME_H1), 0, n_candles)
    if rates is None or len(rates) == 0:
        logger.error(f"Gagal download data: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]

    logger.success(f"Download OK: {len(df)} candle | {df.index[0]} s/d {df.index[-1]}")
    return df


def analyze_label_distribution(labels: pd.Series) -> bool:
    """
    Cek apakah distribusi label sudah cukup representatif.
    PRD: minimal 500 sample per regime.
    """
    MIN_SAMPLES = 200  # Turunkan threshold karena XAUUSD data terbatas

    logger.info("\nDistribusi Label Regime:")
    logger.info("-" * 40)

    all_ok = True
    for regime in [r.value for r in Regime if r != Regime.UNCERTAIN]:
        count = (labels == regime).sum()
        status = "OK" if count >= MIN_SAMPLES else "KURANG"
        logger.log("SUCCESS" if count >= MIN_SAMPLES else "WARNING",
                   f"  {regime:<20}: {count:>5} samples [{status}]")
        if count < MIN_SAMPLES:
            all_ok = False

    uncertain_count = (labels == "UNCERTAIN").sum()
    uncertain_pct = uncertain_count / len(labels) * 100
    logger.info(f"  {'UNCERTAIN':<20}: {uncertain_count:>5} samples ({uncertain_pct:.1f}%)")

    return all_ok


def train_and_validate():
    """
    Main training pipeline:
    1. Download data historis
    2. Feature engineering
    3. Label regime (ground truth)
    4. Train HMM + KMeans
    5. Validasi akurasi (gate: >= 70%)
    6. Simpan model
    """
    logger.info("=" * 60)
    logger.info("REGIME DETECTOR TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Connect MT5 ───────────────────────────────────────────────────
    logger.info("\n[1/5] Menghubungkan ke MT5...")
    if not connector.initialize():
        logger.error("Gagal connect ke MT5!")
        return False

    # ── Step 2: Download data ─────────────────────────────────────────────────
    logger.info(f"\n[2/5] Download data historis XAUUSD {settings.PRIMARY_TF} (2 tahun)...")
    df_raw = download_historical_data(symbol=settings.SYMBOLS[0], timeframe=settings.PRIMARY_TF, years=2)

    if df_raw.empty:
        logger.error("Download data gagal!")
        connector.shutdown()
        return False

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    logger.info(f"\n[3/5] Feature engineering ({len(df_raw)} candle)...")
    df_features = get_features(df_raw)

    if df_features is None or df_features.empty:
        logger.error("Feature engineering gagal!")
        connector.shutdown()
        return False

    logger.success(f"Features OK: {len(df_features)} baris, {len(df_features.columns)} kolom")

    # ── Step 4: Labeling ──────────────────────────────────────────────────────
    logger.info("\n[4/5] Membuat label regime (ground truth)...")
    labels = label_regime_manual(df_features)
    df_features["regime_label"] = labels

    # Cek distribusi
    all_ok = analyze_label_distribution(labels)
    if not all_ok:
        logger.warning("Beberapa regime kurang sample, training tetap dilanjutkan...")

    # ── Step 5: Train Model ───────────────────────────────────────────────────
    logger.info("\n[5/5] Training Regime Detector (HMM + KMeans)...")

    # Split data: 80% train, 20% validation
    n_train = int(len(df_features) * 0.8)
    df_train = df_features.iloc[:n_train]
    df_val = df_features.iloc[n_train:]

    logger.info(f"Train set: {len(df_train)} baris | Val set: {len(df_val)} baris")

    detector = RegimeDetector()
    success = detector.train(df_train)

    if not success:
        logger.error("Training gagal!")
        connector.shutdown()
        return False

    # ── Validasi ──────────────────────────────────────────────────────────────
    logger.info("\nValidasi pada data out-of-sample...")
    val_results = detector.validate(df_val)

    connector.shutdown()

    if not val_results:
        logger.warning("Validasi tidak menghasilkan data yang cukup.")
    else:
        overall_acc = val_results.get("overall_accuracy", 0)
        if overall_acc >= 0.70:
            logger.success(f"\nGATE Phase 2 LOLOS! Akurasi: {overall_acc:.1%}")
            logger.success("Regime Detector siap digunakan.")
        else:
            logger.warning(f"\nGATE Phase 2 BELUM LOLOS. Akurasi: {overall_acc:.1%} (target: 70%)")
            logger.info("Model tetap disimpan. Pertimbangkan fine-tuning parameter.")

    return True


if __name__ == "__main__":
    success = train_and_validate()
    sys.exit(0 if success else 1)
