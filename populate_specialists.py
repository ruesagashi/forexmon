"""
populate_specialists.py
Script untuk generate initial Specialist Pool menggunakan data historis MT5.
Dijalankan sebelum memulai Phase 8 (Paper Trading) agar database tidak kosong.
"""

import sys
from pathlib import Path
from loguru import logger

# Setup logging ke console
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

def main():
    logger.info("Memulai inisialisasi awal Specialist Pool...")
    
    from execution.mt5_connector import connector
    if not connector.initialize():
        logger.error("Gagal connect ke MT5. Pastikan MT5 terbuka.")
        return
        
    from config.settings import settings
    from data.features import get_features
    from models.regime_detector import label_regime_manual, Regime
    from models.specialist import SpecialistTrainer
    from core.pool_manager import pool_manager
    from selection.eliminator import run_pre_filter
    import random
    
    for symbol in settings.SYMBOLS:
        logger.info(f"Mengambil 50000 candle {symbol} {settings.PRIMARY_TF} dari MT5...")
        df_raw = connector.get_candles(symbol, settings.PRIMARY_TF, count=50000)
        if df_raw is None or df_raw.empty:
            logger.error(f"Gagal ambil data {symbol}.")
            continue
            
        logger.info(f"Ekstraksi fitur teknikal {symbol}...")
        df_f = get_features(df_raw)
        df_f = df_f.dropna().copy()
        
        logger.info(f"Melabeli Regime historis {symbol}...")
        df_f["regime"] = label_regime_manual(df_f)
        
        base_features = [c for c in df_f.columns if c not in ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread', 'regime', 'target', 'tp_price', 'sl_price']]
        
        for regime in Regime:
            if regime == Regime.UNCERTAIN:
                continue
                
            logger.info(f"--- Training Specialist untuk {regime.name} ({symbol}) ---")
            df_regime = df_f[df_f["regime"] == regime.value].copy()
            
            if len(df_regime) < 150:
                logger.warning(f"Data historis {regime.name} ({symbol}) terlalu sedikit ({len(df_regime)} baris). Skip.")
                continue
                
            for i in range(3):
                # Acak fitur agar punya variasi, ambil 80% fitur
                k = int(len(base_features) * 0.8)
                subset_features = random.sample(base_features, k)
                
                trainer = SpecialistTrainer(subset_features)
                logger.info(f"Training Specialist {i+1}/3 untuk {regime.name} ({symbol}) dengan {len(subset_features)} fitur...")
                
                try:
                    spec = trainer.generate_specialist(regime, symbol, df_f, df_f["regime"])
                    if spec:
                        # Jalankan Pre-Filter Stage 1 (Backtest & Monte Carlo)
                        is_passed, metrics = run_pre_filter(df_regime, spec)
                        
                        if is_passed:
                            spec.status = "PROBATION"
                            pool_manager.add_specialist(spec)
                            logger.success(f"Specialist {spec.id} ({symbol}) ditambahkan ke Pool!")
                        else:
                            logger.warning(f"Specialist {spec.id} ({symbol}) gagal Pre-Filter (MC/Backtest).")
                    else:
                        logger.warning(f"Gagal train specialist {symbol} (tidak ada edge).")
                except Exception as e:
                    logger.error(f"Error train specialist: {e}")

    connector.shutdown()
    logger.success("Inisialisasi Pool Selesai. Siap untuk Live Trading Phase 8!")

if __name__ == "__main__":
    main()
