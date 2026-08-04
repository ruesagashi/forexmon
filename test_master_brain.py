import sys
sys.path.insert(0, '.')
from config.settings import settings
from execution.mt5_connector import connector
from core.master_brain import master_brain

connector.initialize()

# Test detect
regime, confidence = master_brain.detect_regime('XAUUSD.vxc', 'M15')

print(f"Regime: {regime}")
print(f"Confidence: {confidence:.2f}")

connector.shutdown()
