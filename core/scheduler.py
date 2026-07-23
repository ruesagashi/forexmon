"""
core/scheduler.py
Main Loop Scheduler untuk Live Trading.
Bertugas menarik data candle secara berkala dan men-trigger execution_engine.
"""

import time
from datetime import datetime
from loguru import logger

from execution.mt5_connector import connector
from core.execution import execution_engine
from config.settings import settings


class Scheduler:
    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "H1"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.last_candle_time = None
        self.is_running = False

    def start(self):
        """Memulai infinite loop live trading."""
        logger.info(f"=== Memulai LIVE TRADING Scheduler untuk {self.symbol} {self.timeframe} ===")
        self.is_running = True
        
        while self.is_running:
            try:
                self._run_cycle()
            except KeyboardInterrupt:
                logger.info("Scheduler dihentikan oleh user (KeyboardInterrupt).")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Error pada Scheduler: {e}")
                
            # Tunggu X detik sebelum cek lagi (sesuai PRD)
            time.sleep(settings.ORDER_MONITOR_SECONDS)

    def stop(self):
        self.is_running = False
        logger.info("=== Scheduler dihentikan ===")

    def _run_cycle(self):
        """Mengecek apakah ada candle baru. Jika ada, jalankan execution cycle."""
        if not connector.ensure_connected():
            return
            
        # Ambil 500 candle terakhir (perlu history untuk indikator & regime)
        df = connector.get_candles(self.symbol, self.timeframe, count=500)
        
        if df is None or df.empty:
            logger.warning("[Scheduler] Gagal mengambil data candle dari MT5.")
            return
            
        current_candle_time = df.index[-1]
        
        # Inisialisasi last_candle_time jika baru pertama kali run
        if self.last_candle_time is None:
            self.last_candle_time = current_candle_time
            logger.info(f"[Scheduler] Inisialisasi. Candle terakhir di {current_candle_time}")
            
        # Jika ada candle baru yang CLOSE (bukan tick baru dalam candle yang sama)
        if current_candle_time > self.last_candle_time:
            logger.success(f"[Scheduler] NEW CANDLE DETECTED: {current_candle_time} (Prev: {self.last_candle_time})")
            self.last_candle_time = current_candle_time
            
            # TODO: Idealnya kita perlu mengekstrak features di sini sebelum dipassing.
            # Sementara ini kita asumsikan ExecutionEngine / MasterBrain yang akan handle feature extraction.
            from data.pipeline import extract_features_pipeline
            df_features = extract_features_pipeline(df)
            
            if not df_features.empty:
                # Menjalankan logika AI dan Eksekusi
                execution_engine.run_cycle(df_features)

scheduler = Scheduler(symbol=settings.SYMBOLS[0], timeframe=settings.PRIMARY_TF)
