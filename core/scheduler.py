"""
core/scheduler.py
Main Loop Scheduler untuk Live Trading.
Bertugas menarik data candle secara berkala dan men-trigger execution_engine.
"""

import time
from datetime import datetime
from loguru import logger
import MetaTrader5 as mt5

from execution.mt5_connector import connector
from core.execution import execution_engine
from config.settings import settings


class Scheduler:
    def __init__(self, symbols: list = None, timeframe: str = "H1"):
        self.symbols = symbols or settings.SYMBOLS
        self.timeframe = timeframe
        self.last_candle_time = {sym: None for sym in self.symbols}
        self.is_running = False
        
        self.last_populate_check = datetime.now()
        self.populate_interval_hours = 24

    def start(self):
        """Memulai infinite loop live trading."""
        logger.info(f"=== Memulai LIVE TRADING Scheduler untuk {len(self.symbols)} symbol ({self.timeframe}) ===")
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
                
            try:
                self._monitor_orders()
            except Exception as e:
                logger.error(f"Error pada Order Monitor: {e}")
                
            # Tunggu X detik sebelum cek lagi (sesuai PRD)
            time.sleep(settings.ORDER_MONITOR_SECONDS)

    def stop(self):
        self.is_running = False
        logger.info("=== Scheduler dihentikan ===")
        
    def _monitor_orders(self):
        """Memonitor posisi terbuka untuk mendeteksi yang sudah tertutup (SL/TP)."""
        if not connector.ensure_connected():
            return
            
        from core.memory import db
        from core.pool_manager import pool_manager
        
        # Ambil semua trade yang masih "open" di database
        open_db_trades = db.get_open_trades()
        
        if not open_db_trades:
            return
            
        # Ambil posisi terbuka di MT5
        active_mt5_tickets = [p['ticket'] for p in connector.get_open_positions()]
        
        for trade in open_db_trades:
            ticket = trade['ticket']
            if ticket not in active_mt5_tickets:
                # Trade sudah tertutup di MT5!
                # TODO: Retrieve history deal to get exact PnL. For now, estimate based on SL/TP logic.
                # In real scenario: use mt5.history_deals_get
                history = mt5.history_deals_get(position=ticket)
                if history:
                    deal = history[-1] # Usually the closing deal
                    pnl = deal.profit
                    exit_price = deal.price
                else:
                    # Fallback if no history (e.g. testing)
                    pnl = 0.0 
                    exit_price = 0.0
                    
                result = "WIN" if pnl > 0 else "LOSS"
                
                db.close_trade(
                    trade_id=trade['id'],
                    exit_price=exit_price,
                    pnl=pnl,
                    result=result
                )
                logger.info(f"[Monitor] Trade {ticket} closed. Result: {result}, PnL: {pnl:.2f}")
                
                from monitoring.alerting import telegram_alerter
                telegram_alerter.send_alert(f"🏁 <b>TRADE CLOSED</b>\nSym: {trade['symbol']}\nResult: {result}\nPnL: {pnl:.2f}\nSpec: {trade['specialist_id']}")
                
                
                # Update metrics dan trigger evaluasi
                db.update_specialist_metrics(trade['specialist_id'])
                pool_manager.evaluate_fast_kill(trade['specialist_id'])

    def _run_cycle(self):
        """Mengecek apakah ada candle baru untuk setiap symbol."""
        if not connector.ensure_connected():
            return
            
        if (datetime.now() - self.last_populate_check).total_seconds() > self.populate_interval_hours * 3600:
            self.check_and_repopulate_if_needed()
            self.last_populate_check = datetime.now()
            
        for symbol in self.symbols:
            # Ambil 500 candle terakhir (perlu history untuk indikator & regime)
            df = connector.get_candles(symbol, self.timeframe, count=500)
            
            if df is None or df.empty:
                logger.warning(f"[Scheduler] Gagal mengambil data candle {symbol} dari MT5.")
                continue
                
            current_candle_time = df.index[-1]
            
            # Inisialisasi last_candle_time jika baru pertama kali run
            if self.last_candle_time[symbol] is None:
                self.last_candle_time[symbol] = current_candle_time
                logger.info(f"[Scheduler] Inisialisasi {symbol}. Candle terakhir di {current_candle_time}")
                
            # Jika ada candle baru yang CLOSE (bukan tick baru dalam candle yang sama)
            if current_candle_time > self.last_candle_time[symbol]:
                logger.success(f"[Scheduler] NEW CANDLE DETECTED for {symbol}: {current_candle_time} (Prev: {self.last_candle_time[symbol]})")
                self.last_candle_time[symbol] = current_candle_time
                
                # Ekstrak features
                from data.features import get_features
                df_features = get_features(df)
                
                if not df_features.empty:
                    # Menjalankan logika AI dan Eksekusi
                    execution_engine.run_cycle(symbol, self.timeframe, df_features)

    def check_and_repopulate_if_needed(self):
        """Mengecek apakah perlu generate specialist baru (auto-repopulate)."""
        logger.info("[Scheduler] Mengecek kebutuhan auto-repopulate specialist...")
        from core.memory import db
        from models.regime_detector import Regime
        from config.settings import settings
        from data.features import get_features
        from models.regime_detector import label_regime_manual
        from models.specialist import SpecialistTrainer
        from core.pool_manager import pool_manager
        from selection.eliminator import run_pre_filter
        import random
        
        for symbol in self.symbols:
            df_raw = None
            df_f = None
            base_features = []
            
            for regime in Regime:
                if regime == Regime.UNCERTAIN:
                    continue
                    
                approved_count = db.count_specialists(status="APPROVED", regime_type=regime.value)
                if approved_count < 2:
                    logger.warning(f"[Scheduler] {regime.name} ({symbol}) hanya memiliki {approved_count} APPROVED specialist. Generating 3 baru...")
                    
                    # Fetch and prepare data lazily
                    if df_raw is None:
                        logger.info(f"[Scheduler] Mendownload history untuk training repopulate {symbol}...")
                        df_raw = connector.get_candles(symbol, self.timeframe, count=50000)
                        if df_raw is None or df_raw.empty:
                            logger.error(f"[Scheduler] Gagal ambil data history {symbol} untuk repopulate.")
                            break
                        df_f = get_features(df_raw).dropna().copy()
                        df_f["regime"] = label_regime_manual(df_f)
                        base_features = [c for c in df_f.columns if c not in ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread', 'regime', 'target', 'tp_price', 'sl_price']]
                    
                    df_regime = df_f[df_f["regime"] == regime.value].copy()
                    if len(df_regime) < 300:
                        logger.warning(f"[Scheduler] Data {regime.name} terlalu sedikit ({len(df_regime)}). Skip repopulate.")
                        continue
                        
                    for i in range(3):
                        k = int(len(base_features) * 0.8)
                        subset_features = random.sample(base_features, k)
                        trainer = SpecialistTrainer(subset_features)
                        
                        try:
                            spec = trainer.generate_specialist(regime, symbol, df_f, df_f["regime"])
                            if spec:
                                is_passed, metrics = run_pre_filter(df_regime, spec)
                                if is_passed:
                                    spec.status = "PROBATION"
                                    pool_manager.add_specialist(spec)
                                    logger.success(f"[Scheduler] Auto-Repopulate: {spec.id} ditambahkan (PROBATION).")
                                else:
                                    logger.warning(f"[Scheduler] Auto-Repopulate: {spec.id} gagal Pre-Filter.")
                        except Exception as e:
                            logger.error(f"[Scheduler] Auto-Repopulate error: {e}")

scheduler = Scheduler(symbols=settings.SYMBOLS, timeframe=settings.PRIMARY_TF)
