"""
selection/backtester.py
Fungsi untuk menjalankan backtest historical pada specialist dan mengupdate statusnya.
"""

from loguru import logger
import pandas as pd
from typing import Dict

from core.memory import db
from models.specialist import Specialist
from selection.eliminator import BacktestEngine
from execution.mt5_connector import connector
from data.features import get_features
from config.settings import settings

def backtest_specialist_on_history(specialist: Specialist, df_history: pd.DataFrame) -> dict:
    """
    Menjalankan backtest pada data history untuk 1 specialist.
    Mengembalikan dict hasil evaluasi.
    """
    logger.info(f"[Backtester] Menjalankan backtest untuk Specialist {specialist.id} ({specialist.symbol})...")
    
    # Hitung features
    from models.regime_detector import label_regime_manual
    df_f = get_features(df_history.copy()).dropna()
    df_f["regime"] = label_regime_manual(df_f)
    
    # Filter dataset hanya untuk regime specialist ini (seperti saat training!)
    df_f = df_f[df_f["regime"] == specialist.regime.value].copy()

    if len(df_f) < 100:
        logger.warning(f"[Backtester] Data history (setelah features) terlalu sedikit untuk {specialist.id}")
        return {"winrate": 0.0, "profit_factor": 0.0, "total_trades": 0, "max_drawdown": 0.0}
        
    engine = BacktestEngine(df_f, specialist)
    results = engine.run()
    return results

def backtest_all_specialists():
    """
    Loop semua specialist di database, jalankan backtest dengan data history 6 bulan terakhir.
    Update status ke APPROVED jika WR > 60% dan PF > 1.3
    Update status ke ELIMINATED jika WR < 50%
    """
    logger.info("[Backtester] Memulai backtest massal untuk semua specialist...")
    
    conn = db._get_conn()
    rows = conn.execute("SELECT * FROM specialists").fetchall()
    
    if not rows:
        logger.info("[Backtester] Tidak ada specialist di database.")
        return
        
    connector.initialize()
    
    # Ambil data history (6 bulan ~ 55000 candle untuk M5)
    # Cache per symbol untuk optimasi agar tidak download berkali-kali
    history_cache = {}
    
    approved_count = 0
    eliminated_count = 0
    
    for row in rows:
        spec_dict = dict(row)
        spec_id = spec_dict["id"]
        symbol = spec_dict["symbol"]
        model_path = spec_dict["model_path"]
        
        try:
            specialist = Specialist.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load {spec_id}: {e}")
            continue
            
        if symbol not in history_cache:
            logger.info(f"[Backtester] Mendownload history 6 bulan (55000 candle) untuk {symbol}...")
            df_hist = connector.get_candles(symbol, settings.PRIMARY_TF, count=55000)
            if df_hist is None or df_hist.empty:
                logger.error(f"[Backtester] Gagal download data history {symbol}")
                continue
            history_cache[symbol] = df_hist
            
        df_history = history_cache[symbol]
        results = backtest_specialist_on_history(specialist, df_history)
        
        wr = results.get("winrate", 0.0)
        pf = results.get("profit_factor", 0.0)
        total_trades = results.get("total_trades", 0)
        max_dd = results.get("max_drawdown", 0.0)
        
        logger.info(f"[Backtester] {spec_id} | Trades: {total_trades} | WR: {wr:.2%} | PF: {pf:.2f} | Max DD: {max_dd:.2%}")
        
        # Update metrics di DB
        db.update_specialist_performance(spec_id, wr, pf, total_trades)
        
        # Logika Update Status sesuai aturan
        if total_trades > 0:
            if wr >= 0.60 and pf >= 1.30:
                db.update_specialist_status(spec_id, "APPROVED")
                logger.success(f"{spec_id} → APPROVED (WR {wr:.1%})")
                approved_count += 1
            elif wr < 0.50:
                db.update_specialist_status(spec_id, "ELIMINATED")
                logger.warning(f"{spec_id} → ELIMINATED (WR {wr:.1%})")
                eliminated_count += 1
            else:
                logger.info(f"[Backtester] {spec_id} nanggung. Tetap {spec_dict['status']}.")
        
    connector.shutdown()
    logger.info("[Backtester] Selesai melakukan mass backtest!")
    print(f"Approved: {approved_count}, Eliminated: {eliminated_count}")

if __name__ == "__main__":
    backtest_all_specialists()
