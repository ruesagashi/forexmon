import sys
sys.path.insert(0, '.')
from models.specialist import Specialist
import pickle

spec = Specialist.load('models/saved/specialists/spec_RANGING_5615d238.pkl')
print(f'ID: {spec.id}')
print(f'Symbol: {spec.symbol}')
print(f'Features: {len(spec.feature_cols)} features')
print(f'Model classes: {spec.model.classes_}')

# Cek distribusi prediksi di beberapa sample
from execution.mt5_connector import connector
from data.features import get_features
connector.initialize()
df = connector.get_candles('XAUUSD.vxc', 'M5', count=500)
df_f = get_features(df).dropna()
X = df_f[spec.feature_cols].values
preds = spec.model.predict(X)
probas = spec.model.predict_proba(X)

import numpy as np
unique, counts = np.unique(preds, return_counts=True)
print(f'Distribusi prediksi: {dict(zip(unique, counts))}')
print(f'Avg max confidence: {probas.max(axis=1).mean():.3f}')
connector.shutdown()
