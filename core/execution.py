"""
core/execution.py
Execution Engine yang mengatur siklus pengambilan keputusan trading:
1. Analisa Market (Regime)
2. Seleksi Specialist
3. Evaluasi Risk & Slot
4. Eksekusi Order
"""

from loguru import logger
import pandas as pd

from core.master_brain import MasterBrain
from core.pool_manager import pool_manager
from core.slot_manager import slot_manager
from core.risk_manager import risk_manager
from execution.mt5_connector import connector
from config.settings import settings


class ExecutionEngine:
    def __init__(self):
        self.master_brain = MasterBrain()

    def run_cycle(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """
        Jalankan satu siklus eksekusi.
        Dipanggil setiap kali ada bar/candle baru.
        """
        logger.info(f"--- Memulai Execution Cycle untuk {symbol} ---")

        # 1. Cek koneksi MT5
        if not connector.ensure_connected():
            logger.error("[Execution] MT5 tidak terkoneksi. Abort cycle.")
            return

        # 2. Risk Manager: Cek apakah hari ini boleh trade
        if not risk_manager.check_daily_limit():
            logger.warning("[Execution] Trading dihentikan oleh Risk Manager (Drawdown/Cooldown).")
            return

        # 2.5 Cek posisi terbuka untuk symbol ini
        open_positions = connector.get_open_positions(symbol=symbol)
        if len(open_positions) > 0:
            logger.info(f"[Execution] Masih ada {len(open_positions)} posisi terbuka untuk {symbol}. Menunggu close sebelum entry baru.")
            return

        # 3. Master Brain: Deteksi Regime
        regime, confidence = self.master_brain.detect_regime(df=df)
        
        if self.master_brain.is_hold_mode:
            logger.info("[Execution] Master Brain menginstruksikan HOLD (Regime UNCERTAIN).")
            return

        # 4. Specialist Selection
        specialist = pool_manager.get_best_specialist_for_regime(regime)
        if not specialist:
            logger.info(f"[Execution] Tidak ada Specialist aktif untuk regime {regime}.")
            return
            
        # 5. Prediksi arah trade dari Specialist
        # Ambil baris terakhir (current context)
        current_row = df.iloc[-1:]
        signal, confidence = specialist.predict(current_row)
        
        if signal == 0:
            logger.info(f"[Execution] Specialist {specialist.id} ({regime}) memutuskan HOLD.")
            return
            
        # 6. Slot Manager: Cek apakah ada slot
        if not slot_manager.has_available_slot(specialist.status):
            logger.warning(f"[Execution] Tidak ada slot MT5 tersisa untuk status {specialist.status}.")
            return

        # 7. Eksekusi Order
        self._execute_trade(symbol, timeframe, specialist, signal, df.iloc[-1])

    def _execute_trade(self, symbol: str, timeframe: str, specialist, signal: int, current_bar: pd.Series):
        """Kalkulasi risk dan kirim order ke MT5."""
        
        direction = "BUY" if signal == 1 else "SELL"
        atr = current_bar.get("atr", 1.0)
        
        sl_dist_points = atr * settings.SL_ATR_MULT
        tp_dist_points = atr * settings.TP_ATR_MULT
        
        tick = connector.get_symbol_info(symbol)
        if not tick:
            return
            
        ask = tick["ask"]
        bid = tick["bid"]
        
        if direction == "BUY":
            price = ask
            sl = price - sl_dist_points
            tp = price + tp_dist_points
        else:
            price = bid
            sl = price + sl_dist_points
            tp = price - tp_dist_points
            
        # Hitung Lot Size dari Risk Manager
        lot_size = risk_manager.calculate_lot_size(sl_dist_points)
        if lot_size <= 0:
            logger.warning("[Execution] Lot size 0, order dibatalkan.")
            return
            
        comment = f"spec_{specialist.id}"
        
        logger.info(f"[Execution] Eksekusi {direction} {lot_size} lot di {price:.2f} (SL: {sl:.2f}, TP: {tp:.2f}) by {specialist.id}")
        
        result = connector.send_order(
            symbol=symbol,
            order_type=direction,
            volume=lot_size,
            price=price,
            sl=sl,
            tp=tp,
            comment=comment
        )
        
        if result:
            logger.success(f"[Execution] Order berhasil! Ticket: {result['ticket']}")
            from core.memory import db
            db.add_trade(
                specialist_id=specialist.id,
                symbol=symbol,
                direction=direction,
                entry_price=price,
                sl=sl,
                tp=tp,
                lot_size=lot_size,
                regime_at_entry=specialist.regime.value,
                confidence=0.0,
                ticket=result["ticket"]
            )
            
            from monitoring.alerting import telegram_alerter
            msg = f"🟢 <b>NEW TRADE</b>\nSym: {symbol}\nDir: {direction}\nLot: {lot_size}\nPrice: {price:.2f}\nSpec: {specialist.id}\nRegime: {specialist.regime.value}"
            telegram_alerter.send_alert(msg)
            # Slot otomatis terpakai karena posisi terbuka bertambah (MT5 API sync)

execution_engine = ExecutionEngine()
