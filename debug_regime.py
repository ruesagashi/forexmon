import sys
from loguru import logger
from config.settings import settings
from execution.mt5_connector import connector
from data.features import get_features
from models.regime_detector import RegimeDetector

def main():
    connector.initialize()
    df_raw = connector.get_candles('XAUUSD.vxc', settings.PRIMARY_TF, count=500)
    if df_raw is None or df_raw.empty:
        logger.error("Failed to get candles.")
        return
        
    df_features = get_features(df_raw)
    
    detector = RegimeDetector()
    # Harus di load dulu
    detector.load()
    
    regime, confidence = detector.predict(df_features)
    print(f"\n[RESULT] Final Detected Regime: {regime}, Confidence: {confidence:.2f}")
    
    connector.shutdown()

if __name__ == "__main__":
    main()
