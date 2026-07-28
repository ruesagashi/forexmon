"""
models/specialist.py
Spesialis model untuk trading (XGBoost).
Sesuai PRD Section 3.2.
"""

import uuid
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from loguru import logger
from typing import Tuple, Dict, Any, List

from config.settings import settings
from models.regime_detector import Regime

SPECIALIST_DIR = Path("models/saved/specialists")
SPECIALIST_DIR.mkdir(parents=True, exist_ok=True)


class Specialist:
    """
    Instance dari sebuah strategi ML yang menargetkan regime tertentu.
    Mengandung XGBoost model, metadata, dan performance metrics.
    """

    def __init__(self, regime: Regime, feature_cols: List[str], symbol: str = "XAUUSD"):
        self.id = str(uuid.uuid4())[:8]  # Short UUID
        self.regime = regime
        self.symbol = symbol
        self.status = "PROBATION"
        self.feature_cols = feature_cols
        
        # XGBoost Multi-class model (0: HOLD, 1: BUY, 2: SELL)
        self.model = None
        
        # Performance Tracking
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.trades_count = 0
        self.composite_score = 0.0

    def predict(self, df_row: pd.DataFrame) -> Tuple[int, float]:
        """
        Prediksi arah trade untuk row terkini.
        Args:
            df_row: DataFrame berisi 1 baris terkini.
        Returns:
            (signal, confidence): signal (1: BUY, -1: SELL, 0: HOLD), confidence (0.0-1.0)
        """
        if self.model is None:
            return 0, 0.0

        missing = [c for c in self.feature_cols if c not in df_row.columns]
        if missing:
            logger.error(f"[Specialist {self.id}] Missing features: {missing}")
            return 0, 0.0

        X = df_row[self.feature_cols].values
        
        # Predict probability for classes: [0, 1, 2]
        probas = self.model.predict_proba(X)[0]
        
        pred_class = int(np.argmax(probas))
        confidence = float(probas[pred_class])
        
        # Mapping dari [0, 1, 2] ke [0, 1, -1]
        signal = 0
        if pred_class == 1:
            signal = 1
        elif pred_class == 2:
            signal = -1
            
        # Terapkan threshold confidence (misal 0.6)
        if confidence < 0.6:
            return 0, confidence
            
        return signal, confidence

    def update_metrics(self, win_rate: float, profit_factor: float, trades_count: int, recent_wr: float = None):
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.trades_count = trades_count
        if recent_wr is None:
            recent_wr = win_rate
        
        # PRD: Score = (WR * 0.4) + (Normalized_PF * 0.3) + (Recent_WR * 0.3)
        self.composite_score = (win_rate * 0.4) + (min(profit_factor, 3.0)/3.0 * 0.3) + (recent_wr * 0.3)

    def save(self) -> str:
        filepath = SPECIALIST_DIR / f"spec_{self.regime.value}_{self.id}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({
                "id": self.id,
                "regime": self.regime,
                "symbol": self.symbol,
                "status": self.status,
                "feature_cols": self.feature_cols,
                "model": self.model,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "trades_count": self.trades_count,
                "composite_score": self.composite_score
            }, f)
        return str(filepath)

    @classmethod
    def load(cls, filepath: str) -> "Specialist":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        spec = cls(regime=data["regime"], feature_cols=data["feature_cols"], symbol=data.get("symbol", "XAUUSD"))
        spec.id = data["id"]
        spec.status = data["status"]
        spec.model = data["model"]
        spec.win_rate = data["win_rate"]
        spec.profit_factor = data["profit_factor"]
        spec.trades_count = data["trades_count"]
        spec.composite_score = data.get("composite_score", 0.0)
        return spec


class SpecialistTrainer:
    """
    Membangun data target label dan melatih Specialist (XGBoost).
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def generate_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Label:
        0: HOLD (hit SL atau close di luar batas waktu)
        1: BUY (hit TP sebelum SL)
        2: SELL (hit TP sebelum SL)
        """
        labels = np.zeros(len(df), dtype=int)
        
        # Convert df columns to numpy for speed
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values
        
        sl_mult = settings.SL_ATR_MULT
        tp_mult = settings.TP_ATR_MULT
        
        # Max horizon to look for TP/SL (e.g. 24 candles)
        horizon = 10
        
        n = len(df)
        for i in range(n - 1):
            entry_idx = i + 1  # Kita entry di open candle berikutnya
            entry_price = opens[entry_idx]
            atr = atrs[i]  # ATR dari candle i
            
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult
            
            # Target untuk BUY
            buy_tp = entry_price + tp_dist
            buy_sl = entry_price - sl_dist
            
            # Target untuk SELL
            sell_tp = entry_price - tp_dist
            sell_sl = entry_price + sl_dist
            
            buy_result = 0  # 1 if hit TP first, -1 if hit SL
            sell_result = 0
            
            for j in range(entry_idx, min(n, entry_idx + horizon)):
                curr_h = highs[j]
                curr_l = lows[j]
                
                # Check BUY
                if buy_result == 0:
                    if curr_l <= buy_sl and curr_h >= buy_tp:
                        buy_result = -1  # Konservatif: anggap kena SL kalau 1 candle tembus SL & TP
                    elif curr_l <= buy_sl:
                        buy_result = -1
                    elif curr_h >= buy_tp:
                        buy_result = 1
                
                # Check SELL
                if sell_result == 0:
                    if curr_h >= sell_sl and curr_l <= sell_tp:
                        sell_result = -1  # Konservatif
                    elif curr_h >= sell_sl:
                        sell_result = -1
                    elif curr_l <= sell_tp:
                        sell_result = 1
                        
                if buy_result != 0 and sell_result != 0:
                    break
                    
            if buy_result == 1 and sell_result == 1:
                labels[i] = 0  # Ambiguous — skip, jadikan HOLD
            elif buy_result == 1:
                labels[i] = 1  # BUY jelas
            elif sell_result == 1:
                labels[i] = 2  # SELL jelas
            # else 0 (HOLD: keduanya kena SL atau timeout)
            
        return labels

    def generate_specialist(self, regime: Regime, symbol: str, df_historical: pd.DataFrame, df_labels: pd.Series) -> "Specialist":
        """
        Train specialist baru untuk suatu regime.
        Args:
            regime: Target regime
            symbol: Target symbol
            df_historical: DataFrame fitur
            df_labels: Series dari Master Brain yang melabeli regime tiap row
        """
        logger.info(f"[SpecialistTrainer] Mempersiapkan data untuk {regime.name} ({symbol})...")
        
        # Filter data hanya untuk regime ini
        mask = df_labels == regime.name
        df_regime = df_historical[mask].copy()
        
        if len(df_regime) < 1000:
            logger.warning(f"Data untuk {regime.name} terlalu sedikit ({len(df_regime)}). Butuh minimal 1000.")
            return None
            
        # Generate target labels berdasarkan rules TP/SL
        y_labels = self.generate_labels(df_regime)
        
        X = df_regime[self.feature_cols].values
        y = y_labels
        
        # Cek class balance
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        logger.info(f"[SpecialistTrainer] Label balance untuk {regime.name}: {class_counts}")
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info(f"[SpecialistTrainer] Training model untuk {regime.name}...")
        
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        
        model.fit(X, y, sample_weight=sample_weights)
        
        # Evaluasi In-sample accuracy
        y_pred = model.predict(X)
        acc = (y_pred == y).mean()
        logger.info(f"[SpecialistTrainer] Training selesai. In-sample Accuracy: {acc:.2%}")
        
        # Buat objek Specialist
        spec = Specialist(regime=regime, feature_cols=self.feature_cols, symbol=symbol)
        spec.model = model
        
        return spec
