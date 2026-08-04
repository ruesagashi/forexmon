import os
model_path = "models/saved/regime_detector.pkl"
if os.path.exists(model_path):
    size = os.path.getsize(model_path)
    print(f"✓ Model exists: {model_path}")
    print(f"✓ File size: {size} bytes")
else:
    print(f"✗ Model NOT found: {model_path}")

from models.regime_detector import RegimeDetector
detector = RegimeDetector()
result = detector.load()
print(f"Load result: {result}")
