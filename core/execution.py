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

        # Hitung current_dd dari DB
        from datetime import datetime
        from core.memory import db
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        rows = db._get_conn().execute("SELECT pnl FROM trades WHERE close_at >= ?", (today_start,)).fetchall()
        daily_pnl = sum(r["pnl"] for r in rows if r["pnl"] is not None)
        
        account = connector.get_account_info()
        balance = account["balance"] if account else 1000.0
        
        current_dd = 0.0
        if daily_pnl < 0 and balance > 0:
            current_dd = abs(daily_pnl) / balance

        # 2. Risk Manager: Cek apakah hari ini boleh trade
        if not risk_manager.check_daily_limit(current_dd):
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
        specialist = pool_manager.get_best_specialist_for_regime(regime, symbol=symbol)
        if not specialist:
            logger.info(f"[Execution] Tidak ada Specialist aktif untuk regime {regime}.")
            return
            
        # 5. Prediksi arah trade dari Specialist
        # Ambil baris terakhir (current context)
        current_row = df.iloc[-1:]
        signal, confidence = specialist.predict(current_row)
        
        logger.debug(f"[Execution] Specialist predict: action={signal}, confidence={confidence}")
        
        if signal == 1:
            logger.debug(f"[Execution] Action is BUY (1), confidence {confidence} >= 0.45?")
            if confidence >= 0.45:
                logger.debug(f"[Execution] EXECUTE BUY")
            else:
                logger.debug(f"[Execution] HOLD - confidence too low")
        elif signal == -1:
            logger.debug(f"[Execution] Action is SELL (-1), confidence {confidence} >= 0.45?")
            if confidence >= 0.45:
                logger.debug(f"[Execution] EXECUTE SELL")
            else:
                logger.debug(f"[Execution] HOLD - confidence too low")
        elif signal == 0:
            logger.debug(f"[Execution] Action is HOLD (0), confidence {confidence}")
            
        if signal == 0:
            logger.info(f"[Execution] Specialist {specialist.id} ({regime}) memutuskan HOLD.")
            return
            
        # 6. Slot Manager: Cek apakah ada slot
        if not slot_manager.has_available_slot(specialist.status):
            logger.warning(f"[Execution] Tidak ada slot MT5 tersisa untuk status {specialist.status}.")
            return

        # Validation: Check for counter mismatch
        spec_db = pool_manager.db.get_specialist(specialist.id) if hasattr(pool_manager, 'db') else None
        if not spec_db:
            from core.memory import db
            spec_db = db.get_specialist(specialist.id)
            
        if spec_db and spec_db.get('total_trades', 0) == 0:
            from core.memory import db
            live_trades = db.get_specialist_trades(specialist.id)
            if len(live_trades) > 0:
                logger.error("Counter mismatch detected!")

        # 7. Eksekusi Order
        self._execute_trade(symbol, timeframe, specialist, signal, df.iloc[-1])

    def _execute_trade(self, symbol: str, timeframe: str, specialist, signal: int, current_bar: pd.Series):
        """Kalkulasi risk dan kirim order ke MT5."""
        
        direction = "BUY" if signal == 1 else "SELL"
        atr = current_bar.get("atr", 1.0)
        
        tick = connector.get_symbol_info(symbol)
        if not tick:
            return
            
        ask = tick["ask"]
        bid = tick["bid"]
        
        price = ask if direction == "BUY" else bid

        # Hitung SL/TP menggunakan RiskManager
        sl, tp = risk_manager.calculate_sl_tp(
            entry_price=price, 
            direction=direction, 
            atr=atr, 
            regime=specialist.regime.value
        )
            
        # Hitung Lot Size dari Risk Manager
        account = connector.get_account_info()
        balance = account["balance"] if account else 1000.0
        
        # Konversi absolute distance ke pip (asumsi XAUUSD pip = 0.1 atau 0.01)
        # Sesuai formula user: lot_size = risk_amount / (sl_pips * 0.1)
        # Jika risk $50, entry 2000, sl 1990 ($10 dist). 
        # Real loss 1 lot XAUUSD = $10 * 100 oz = $1000.
        # Agar rumus (sl_pips * 0.1) = $1000, maka sl_pips harus 10000.
        sl_abs_dist = abs(price - sl)
        sl_pips = sl_abs_dist * 1000 
        
        lot_size = risk_manager.calculate_position_size(
            balance=balance, 
            risk_pct=settings.RISK_PER_TRADE, 
            sl_pips=sl_pips
        )
        
        # Penyesuaian lot size mengikuti aturan broker (MT5 volume step)
        vol_step = tick.get("volume_step", 0.01)
        vol_min = tick.get("volume_min", 0.01)
        vol_max = tick.get("volume_max", 100.0)
        
        steps = int(lot_size / vol_step)
        lot_size = steps * vol_step
        lot_size = max(vol_min, min(vol_max, lot_size))
        lot_size = round(lot_size, 2)  # Hindari floating point precision issue (misal 0.07000001)
        
        if lot_size <= 0:
            logger.warning("[Execution] Lot size <= 0, order dibatalkan.")
            return
            
        # Validasi akhir
        if not risk_manager.validate_trade(lot_size, price, sl, tp):
            return
            
        comment = f"spec_{specialist.id}"
        
        logger.info(f"[Execution] Eksekusi {direction} {lot_size:.2f} lot di {price:.2f} (SL: {sl:.2f}, TP: {tp:.2f}) by {specialist.id}")
        
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
            db.open_trade(
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
