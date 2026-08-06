"""
main.py
Entry point — Adaptive Forex Trading Bot (MT5)
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)
logger.add(
    LOG_DIR / "forexmon_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="90 days",
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
)


def run_live():
    from core.scheduler import scheduler
    from core.memory import db
    
    logger.info("=" * 60)
    logger.info("  ADAPTIVE FOREX TRADING BOT — MT5")
    logger.info("  Mode: LIVE TRADING")
    logger.info("=" * 60)
    
    db.log_event("SYSTEM_START", "Live trading engine started.")
    
    try:
        from core.pool_manager import pool_manager
        pool_manager.retroactive_re_evaluate_all()
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Dihentikan oleh user.")
    finally:
        scheduler.stop()


def run_test():
    """Jalankan Foundation Test (Phase 1)"""
    logger.info("=" * 60)
    logger.info("  ADAPTIVE FOREX TRADING BOT — MT5")
    logger.info("  Phase 1: Foundation Test")
    logger.info("=" * 60)

    from execution.mt5_connector import connector
    if not connector.initialize():
        logger.error("Gagal connect ke MT5.")
        return
        
    logger.success("MT5 Connected!")
    
    df_raw = connector.get_candles("XAUUSD", "H1", count=300)
    if df_raw is not None:
        logger.success(f"Data XAUUSD H1 OK: {len(df_raw)} candles.")
        from data.features import get_features
        df_f = get_features(df_raw)
        if df_f is not None and not df_f.empty:
            logger.success("Feature Engineering OK.")
            
    account = connector.get_account_info()
    if account:
        logger.success(f"Account Balance: {account['balance']}")
        
    connector.shutdown()
    logger.success("Foundation Test Selesai!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forexmon Adaptive MT5 Bot")
    parser.add_argument("--mode", type=str, choices=["live", "test"], default="live", help="Mode eksekusi")
    args = parser.parse_args()

    if args.mode == "live":
        run_live()
    elif args.mode == "test":
        run_test()
