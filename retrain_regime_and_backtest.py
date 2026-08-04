import sys
from loguru import logger
import pandas as pd
from config.settings import settings
from execution.mt5_connector import connector
from data.features import get_features
from models.regime_detector import RegimeDetector, label_regime_manual
from selection.backtester import backtest_all_specialists

def main():
    connector.initialize()
    logger.info("Fetching 50k candles to retrain regime detector...")
    # Get any major symbol to train the regime detector, e.g., XAUUSD.vxc
    df_raw = connector.get_candles('XAUUSD.vxc', settings.PRIMARY_TF, count=90000)
    
    if df_raw is None or df_raw.empty:
        logger.error("Failed to fetch candles.")
        return
        
    logger.info("Generating features...")
    df_f = get_features(df_raw).dropna().copy()
    
    logger.info("Labeling regimes manually with the updated rule...")
    df_f["regime"] = label_regime_manual(df_f)
    
    logger.info("Training RegimeDetector...")
    detector = RegimeDetector()
    success = detector.train(df_f)
    if success:
        logger.success("RegimeDetector trained and saved successfully.")
    else:
        logger.error("Failed to train RegimeDetector.")
        return
        
    logger.info("Re-backtesting all specialists...")
    backtest_all_specialists()
    
    connector.shutdown()
    logger.success("All done!")

if __name__ == "__main__":
    main()
