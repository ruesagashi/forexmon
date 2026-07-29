"""
models/regime_detector.py
Master Brain — Deteksi Regime Pasar menggunakan Hidden Markov Model (HMM).
Sesuai PRD Section 3.2.

5 Regime:
  TRENDING_UP   : HH, HL, ADX > 25, EMA slope naik
  TRENDING_DOWN : LH, LL, ADX > 25, EMA slope turun
  RANGING       : ADX < 20, BB menyempit, ATR rendah
  BREAKOUT      : BB Width expand, Volume surge, ATR naik
  REVERSAL      : RSI divergence, candle pattern konfirmasi

Output: (regime_label, confidence_score)
Jika confidence < 0.6 → UNCERTAIN → sistem HOLD
"""

import os
import pickle
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config.settings import settings

# ─────────────────────────────────────────────────────────────────────────────
# Regime Enum
# ─────────────────────────────────────────────────────────────────────────────

class Regime(str, Enum):
    TRENDING_UP   = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING       = "RANGING"
    BREAKOUT      = "BREAKOUT"
    REVERSAL      = "REVERSAL"
    UNCERTAIN     = "UNCERTAIN"


# Mapping HMM state index → Regime (ditentukan setelah training berdasarkan mean feature)
# Urutan ini akan di-kalibrasi otomatis saat training
REGIME_LIST = [
    Regime.TRENDING_UP,
    Regime.TRENDING_DOWN,
    Regime.RANGING,
    Regime.BREAKOUT,
    Regime.REVERSAL,
]

MODEL_PATH = Path("models/saved/regime_detector.pkl")
SCALER_PATH = Path("models/saved/regime_scaler.pkl")
STATE_MAP_PATH = Path("models/saved/regime_state_map.pkl")
RF_MODEL_PATH = Path("models/saved/regime_rf.pkl")
LABEL_ENCODER_PATH = Path("models/saved/regime_label_encoder.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Buat Label Regime secara Rule-Based (untuk ground truth awal)
# ─────────────────────────────────────────────────────────────────────────────

def label_regime_manual(df: pd.DataFrame) -> pd.Series:
    """
    Buat label regime berdasarkan rules teknikal.
    Digunakan sebagai ground truth untuk training awal.

    Rules:
      TRENDING_UP   : ADX > 25 AND EMA slope > 0 AND close > EMA50 AND close > EMA200
      TRENDING_DOWN : ADX > 25 AND EMA slope < 0 AND close < EMA50 AND close < EMA200
      RANGING       : ADX < 20 AND BB width < BB_width_median
      BREAKOUT      : BB width > BB_width_75pct AND volume_ratio > 1.5 AND ATR_ratio > ATR_ratio_median
      REVERSAL      : RSI divergence proxy (RSI < 30 atau > 70) AND candle pattern
      Sisanya       : UNCERTAIN
    """
    required = ["adx", "ema_slope", "close", "ema_50", "ema_200",
                "bb_width", "volume_ratio", "atr_ratio", "rsi",
                "pattern_hammer", "pattern_pin_bar", "pattern_engulf_bull", "pattern_engulf_bear"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"[Labeling] Kolom hilang: {missing}")
        return pd.Series("UNCERTAIN", index=df.index)

    labels = pd.Series("UNCERTAIN", index=df.index)
    bb_width_med = df["bb_width"].median()
    bb_width_75 = df["bb_width"].quantile(0.75)
    atr_ratio_med = df["atr_ratio"].median()

    # TRENDING UP
    trending_up = (
        (df["adx"] > 25) &
        (df["ema_slope"] > 0.05) &
        (df["close"] > df["ema_50"]) &
        (df["close"] > df["ema_200"])
    )

    # TRENDING DOWN
    trending_down = (
        (df["adx"] > 25) &
        (df["ema_slope"] < -0.05) &
        (df["close"] < df["ema_50"]) &
        (df["close"] < df["ema_200"])
    )

    # RANGING
    ranging = (
        (df["adx"] < 20) &
        (df["bb_width"] < bb_width_med) &
        (abs(df["ema_slope"]) < 0.05) &
        (df["atr_ratio"] < atr_ratio_med)
    )

    # BREAKOUT
    breakout = (
        (df["bb_width"] > bb_width_75) &
        (df["volume_ratio"] > 1.5) &
        (df["atr_ratio"] > atr_ratio_med)
    )

    # REVERSAL — RSI extreme + candle pattern konfirmasi
    reversal = (
        (
            (df["rsi"] < 30) & (df["pattern_hammer"] == 1)
        ) | (
            (df["rsi"] > 70) & (df["pattern_pin_bar"] == 1)
        ) | (
            (df["rsi"] < 35) & (df["pattern_engulf_bull"] == 1)
        ) | (
            (df["rsi"] > 65) & (df["pattern_engulf_bear"] == 1)
        )
    )

    # Terapkan dengan prioritas (Reversal > Breakout > Trending > Ranging)
    labels[ranging] = Regime.RANGING
    labels[trending_up] = Regime.TRENDING_UP
    labels[trending_down] = Regime.TRENDING_DOWN
    labels[breakout] = Regime.BREAKOUT
    labels[reversal] = Regime.REVERSAL

    # Log distribusi label
    dist = labels.value_counts()
    logger.info(f"[Labeling] Distribusi regime:\n{dist.to_string()}")
    uncertain_pct = (labels == "UNCERTAIN").mean() * 100
    logger.info(f"[Labeling] UNCERTAIN: {uncertain_pct:.1f}%")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Kalibrasi State → Regime
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_state_to_regime(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    feature_names: list,
) -> dict:
    """
    Tentukan mapping HMM state index → Regime berdasarkan mean feature per state.

    Logic:
      - State dengan ADX tinggi & EMA slope positif → TRENDING_UP
      - State dengan ADX tinggi & EMA slope negatif → TRENDING_DOWN
      - State dengan ADX rendah & BB width rendah   → RANGING
      - State dengan BB width tinggi & volume tinggi → BREAKOUT
      - State sisanya                                 → REVERSAL
    """
    means = model.means_  # shape: (n_states, n_features)
    adx_idx = feature_names.index("adx")
    bb_idx = feature_names.index("bb_width")
    atr_idx = feature_names.index("atr_ratio")
    vol_idx = feature_names.index("volume_ratio")
    slope_idx = feature_names.index("ema_slope")

    n_states = means.shape[0]
    state_map = {}
    used_regimes = set()

    # Score setiap state untuk setiap regime
    scores = {}
    for s in range(n_states):
        m = means[s]
        scores[s] = {
            Regime.TRENDING_UP:   m[adx_idx] + max(m[slope_idx], 0) * 10,
            Regime.TRENDING_DOWN: m[adx_idx] + max(-m[slope_idx], 0) * 10,
            Regime.RANGING:       -m[adx_idx] - m[bb_idx] * 5,
            Regime.BREAKOUT:      m[bb_idx] * 5 + m[vol_idx] * 3,
            Regime.REVERSAL:      m[atr_idx] * 5 - m[adx_idx],
        }

    # Greedy assignment: state dengan score tertinggi per regime
    regime_priority = [Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.RANGING, Regime.BREAKOUT, Regime.REVERSAL]
    remaining_states = list(range(n_states))

    for regime in regime_priority:
        if not remaining_states:
            break
        best_state = max(remaining_states, key=lambda s: scores[s][regime])
        state_map[best_state] = regime
        remaining_states.remove(best_state)
        used_regimes.add(regime)

    # Sisa state (jika n_states > 5) → UNCERTAIN
    for s in remaining_states:
        state_map[s] = Regime.UNCERTAIN

    logger.info(f"[HMM] State mapping: {state_map}")
    return state_map


# ─────────────────────────────────────────────────────────────────────────────
# RegimeDetector Class
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Master Brain — deteksi regime market menggunakan HMM + fallback K-Means.

    Usage:
        detector = RegimeDetector()
        detector.train(df_features)                 # Training
        regime, confidence = detector.predict(df)  # Prediksi
    """

    FEATURE_COLS = settings.REGIME_FEATURE_VECTOR  # [adx, bb_width, atr_ratio, volume_ratio, ema_slope]
    N_STATES = settings.REGIME_HMM_STATES           # 5

    # Fitur tambahan untuk RF (lebih banyak dari HMM)
    RF_FEATURE_COLS = [
        "adx", "adx_pos", "adx_neg", "ema_slope",
        "bb_width", "atr_ratio", "volume_ratio",
        "rsi", "macd_hist", "cci",
        "higher_high", "lower_low", "trend_direction",
    ]

    def __init__(self):
        self.hmm_model: Optional[hmm.GaussianHMM] = None
        self.kmeans_model: Optional[KMeans] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.rf_scaler: Optional[StandardScaler] = None
        self.state_map: dict = {}
        self._is_trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, lengths: list = None) -> bool:
        """
        Train HMM + K-Means pada data fitur historis.

        Args:
            df: DataFrame dengan fitur (harus ada kolom FEATURE_COLS)
            lengths: List panjang sequence untuk HMM (jika data dari multiple symbol)

        Returns:
            True jika training berhasil.
        """
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            logger.error(f"[RegimeDetector] Fitur tidak ditemukan: {missing}")
            return False

        X_raw = df[self.FEATURE_COLS].dropna().values
        if len(X_raw) < 200:
            logger.error(f"[RegimeDetector] Data terlalu sedikit untuk training: {len(X_raw)} baris")
            return False

        logger.info(f"[RegimeDetector] Training dengan {len(X_raw)} sample, {len(self.FEATURE_COLS)} fitur...")

        # Normalisasi fitur
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        # ── Train HMM ─────────────────────────────────────────────────────────
        if lengths is None:
            lengths = [len(X_scaled)]

        try:
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.N_STATES,
                covariance_type="full",
                n_iter=200,
                tol=1e-4,
                random_state=42,
                verbose=False,
            )
            self.hmm_model.fit(X_scaled, lengths)
            logger.success(f"[HMM] Training selesai. Score: {self.hmm_model.score(X_scaled, lengths):.2f}")

            # Kalibrasi state → regime
            self.state_map = _calibrate_state_to_regime(
                self.hmm_model, self.scaler, self.FEATURE_COLS
            )

        except Exception as e:
            logger.error(f"[HMM] Training gagal: {e}")
            self.hmm_model = None

        # ── Train K-Means (sebagai fallback) ──────────────────────────────────
        try:
            self.kmeans_model = KMeans(
                n_clusters=self.N_STATES,
                random_state=42,
                n_init=10,
            )
            self.kmeans_model.fit(X_scaled)
            logger.success("[KMeans] Fallback model training selesai.")
        except Exception as e:
            logger.error(f"[KMeans] Training gagal: {e}")
            self.kmeans_model = None

        # ── Train RandomForest Supervised (PRIMARY predictor) ─────────────────
        # RF dilatih pada label manual (ground truth rules) → akurasi tinggi
        from models.regime_detector import label_regime_manual
        try:
            labels = label_regime_manual(df)
            # Filter hanya non-UNCERTAIN
            valid_mask = labels != "UNCERTAIN"
            # Gunakan lebih banyak fitur untuk RF
            rf_cols = [c for c in self.RF_FEATURE_COLS if c in df.columns]
            X_rf_raw = df.loc[valid_mask, rf_cols].values
            y_rf = labels[valid_mask].values

            if len(X_rf_raw) >= 100:
                self.rf_scaler = StandardScaler()
                X_rf_scaled = self.rf_scaler.fit_transform(X_rf_raw)

                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y_rf)

                self.rf_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                )
                self.rf_model.fit(X_rf_scaled, y_encoded)

                # Quick train-set accuracy check
                train_acc = self.rf_model.score(X_rf_scaled, y_encoded)
                logger.success(f"[RF] Training selesai. Train accuracy: {train_acc:.1%} ({len(X_rf_raw)} samples)")
            else:
                logger.warning("[RF] Data terlalu sedikit untuk RF, skip.")

        except Exception as e:
            logger.error(f"[RF] Training gagal: {e}")
            self.rf_model = None

        if self.hmm_model is None and self.kmeans_model is None and self.rf_model is None:
            return False

        self._is_trained = True
        self.save()
        return True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Prediksi regime dari DataFrame fitur.
        Priority: RandomForest (primary) > HMM (secondary) > KMeans (fallback)

        Returns:
            (regime_label, confidence_score)
            Jika confidence < REGIME_CONFIDENCE_MIN -> UNCERTAIN
        """
        if not self._is_trained:
            logger.warning("[RegimeDetector] Model belum di-train.")
            return Regime.UNCERTAIN, 0.0

        if df is None or df.empty:
            return Regime.UNCERTAIN, 0.0

        # -- PRIMARY: RandomForest Supervised ----------------------------------
        if self.rf_model is not None and self.rf_scaler is not None:
            rf_cols = [c for c in self.RF_FEATURE_COLS if c in df.columns]
            X_rf_raw = df[rf_cols].dropna().values
            if len(X_rf_raw) > 0:
                try:
                    X_last = X_rf_raw[-1:]
                    X_scaled_rf = self.rf_scaler.transform(X_last)
                    proba = self.rf_model.predict_proba(X_scaled_rf)[0]
                    pred_class = int(proba.argmax())
                    confidence = float(proba[pred_class])
                    regime_label = self.label_encoder.inverse_transform([pred_class])[0]
                    try:
                        regime = Regime(regime_label)
                    except ValueError:
                        regime = Regime.UNCERTAIN
                    if regime == Regime.RANGING and confidence < 0.75:
                        print(f"[DEBUG] RF confidence too low for RANGING: {confidence:.2f}")
                        return Regime.UNCERTAIN, confidence
                    elif confidence < settings.REGIME_CONFIDENCE_MIN:
                        print(f"[DEBUG] RF confidence too low: {confidence:.2f} for {regime_label}")
                        return Regime.UNCERTAIN, confidence
                        
                    print(f"[DEBUG] RF Pred: {regime_label}, Confidence: {confidence:.2f}, Probas: {proba}")
                    logger.debug(f"[RF] Regime: {regime} (conf: {confidence:.2f})")
                    return regime, confidence
                except Exception as e:
                    logger.warning(f"[RF] Prediksi gagal: {e}, fallback ke HMM.")

        # -- SECONDARY: HMM ---------------------------------------------------
        hmm_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        if len(hmm_cols) == len(self.FEATURE_COLS) and self.hmm_model is not None and self.scaler is not None:
            X_raw = df[self.FEATURE_COLS].dropna().values
            if len(X_raw) > 0:
                X_seq = X_raw[-50:] if len(X_raw) > 50 else X_raw
                X_scaled = self.scaler.transform(X_seq)
                try:
                    regime, confidence = self._predict_hmm(X_scaled)
                    if confidence >= settings.REGIME_CONFIDENCE_MIN:
                        return regime, confidence
                except Exception as e:
                    logger.warning(f"[HMM] Prediksi gagal: {e}")

        # -- FALLBACK: KMeans -------------------------------------------------
        if self.kmeans_model is not None and self.scaler is not None:
            kmeans_cols = [c for c in self.FEATURE_COLS if c in df.columns]
            if len(kmeans_cols) == len(self.FEATURE_COLS):
                X_raw = df[self.FEATURE_COLS].dropna().values
                if len(X_raw) > 0:
                    X_scaled = self.scaler.transform(X_raw[-1:])
                    regime, confidence = self._predict_kmeans(X_scaled)
                    if confidence >= settings.REGIME_CONFIDENCE_MIN:
                        return regime, confidence

        return Regime.UNCERTAIN, 0.0

    def _predict_hmm(self, X_scaled: np.ndarray) -> Tuple[str, float]:
        """
        Prediksi menggunakan HMM.
        Confidence dihitung dari posterior probability state terakhir.
        """
        states = self.hmm_model.predict(X_scaled)
        last_state = int(states[-1])

        # Posterior probability untuk state terakhir
        posteriors = self.hmm_model.predict_proba(X_scaled)
        last_posterior = posteriors[-1]
        confidence = float(last_posterior[last_state])

        regime = self.state_map.get(last_state, Regime.UNCERTAIN)
        print(f"[DEBUG] HMM State: {last_state}, Regime mapped: {regime}, Confidence: {confidence:.2f}")
        return regime, confidence

    def _predict_kmeans(self, X_scaled: np.ndarray) -> Tuple[str, float]:
        """
        Prediksi menggunakan K-Means sebagai fallback.
        Confidence dihitung dari jarak ke centroid (dinormalisasi).
        """
        cluster = int(self.kmeans_model.predict(X_scaled[-1:])[0])

        # Hitung confidence dari jarak ke centroid
        distances = self.kmeans_model.transform(X_scaled[-1:])
        min_dist = float(distances[0, cluster])
        max_dist = float(distances[0].max())

        if max_dist > 0:
            confidence = 1.0 - (min_dist / (max_dist + 1e-8))
        else:
            confidence = 0.5

        # Map cluster → regime menggunakan state_map jika ada, else langsung index
        regime = self.state_map.get(cluster, REGIME_LIST[cluster % len(REGIME_LIST)])
        return regime, float(confidence)

    def predict_single(self, features: dict) -> Tuple[str, float]:
        """
        Prediksi dari single feature dict (untuk real-time 1 candle).
        Menggabungkan dengan rolling context dari predict() bila tersedia.
        """
        feature_arr = np.array([[features.get(f, 0.0) for f in self.FEATURE_COLS]])
        X_scaled = self.scaler.transform(feature_arr)

        if self.hmm_model is not None:
            try:
                cluster = int(self.hmm_model.predict(X_scaled)[0])
                posteriors = self.hmm_model.predict_proba(X_scaled)
                confidence = float(posteriors[0, cluster])
                regime = self.state_map.get(cluster, Regime.UNCERTAIN)
                if confidence >= settings.REGIME_CONFIDENCE_MIN:
                    return regime, confidence
            except Exception:
                pass

        if self.kmeans_model is not None:
            return self._predict_kmeans(X_scaled)

        return Regime.UNCERTAIN, 0.0

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Validasi akurasi regime detector terhadap label manual (ground truth).
        Sesuai PRD Section 2.3: akurasi per regime minimal 70%.

        Pendekatan: prediksi batch seluruh sequence sekaligus lalu bandingkan
        dengan ground truth label — lebih akurat dan tidak ada index mismatch.

        Returns:
            Dict dengan overall_accuracy, per_regime_accuracy, coverage
        """
        if not self._is_trained:
            logger.error("[RegimeDetector] Model belum di-train, tidak bisa validasi.")
            return {}

        from models.regime_detector import label_regime_manual

        # Buat label ground truth
        true_labels = label_regime_manual(df)

        # Filter hanya baris yang TIDAK UNCERTAIN di ground truth
        valid_mask = true_labels != "UNCERTAIN"
        df_valid = df[valid_mask].copy()
        true_valid = true_labels[valid_mask]

        if len(df_valid) < 50:
            logger.warning("[RegimeDetector] Data validasi terlalu sedikit setelah filter UNCERTAIN.")
            return {}

        logger.info(f"[RegimeDetector] Validasi {len(df_valid)} samples...")

        # -- Prediksi Batch dengan RF (primary) atau HMM (fallback) -----------
        pred_regimes = []
        pred_confs = []

        if self.rf_model is not None and self.rf_scaler is not None:
            # RF: prediksi semua baris sekaligus (vectorized)
            rf_cols = [c for c in self.RF_FEATURE_COLS if c in df.columns]
            X_rf_raw = df[rf_cols].values
            X_rf_scaled = self.rf_scaler.transform(X_rf_raw)
            all_probas = self.rf_model.predict_proba(X_rf_scaled)

            for i in range(len(df)):
                proba = all_probas[i]
                pred_class = int(proba.argmax())
                conf = float(proba[pred_class])
                regime_label = self.label_encoder.inverse_transform([pred_class])[0]
                try:
                    regime_pred = Regime(regime_label)
                except ValueError:
                    regime_pred = Regime.UNCERTAIN
                if conf < settings.REGIME_CONFIDENCE_MIN:
                    regime_pred = Regime.UNCERTAIN
                pred_regimes.append(str(regime_pred.value) if hasattr(regime_pred, 'value') else str(regime_pred))
                pred_confs.append(conf)

        else:
            # Fallback: HMM batch prediction
            missing = [c for c in self.FEATURE_COLS if c not in df.columns]
            if missing:
                logger.error(f"[RegimeDetector] Fitur tidak ada: {missing}")
                return {}
            X_raw = df[self.FEATURE_COLS].values
            X_scaled = self.scaler.transform(X_raw)
            all_states = self.hmm_model.predict(X_scaled)
            all_posteriors = self.hmm_model.predict_proba(X_scaled)
            for i, state in enumerate(all_states):
                regime_pred = self.state_map.get(int(state), Regime.UNCERTAIN)
                conf = float(all_posteriors[i, state])
                if conf < settings.REGIME_CONFIDENCE_MIN:
                    regime_pred = Regime.UNCERTAIN
                pred_regimes.append(str(regime_pred.value) if hasattr(regime_pred, 'value') else str(regime_pred))
                pred_confs.append(conf)

        pred_series = pd.Series(pred_regimes, index=df.index)

        # Filter ke baris yang sama dengan ground truth valid
        pred_valid = pred_series[valid_mask]

        # ── Overall Accuracy (exclude UNCERTAIN predictions) ──────────────────
        non_uncertain_mask = pred_valid != Regime.UNCERTAIN
        pred_decisive = pred_valid[non_uncertain_mask]
        true_decisive = true_valid[non_uncertain_mask].astype(str)

        if len(pred_decisive) == 0:
            logger.warning("[RegimeDetector] Semua prediksi UNCERTAIN, tidak bisa hitung akurasi.")
            return {}

        correct = (pred_decisive == true_decisive).sum()
        total = len(pred_decisive)
        overall_acc = correct / total
        coverage = len(pred_decisive) / len(pred_valid)

        # ── Per Regime Accuracy ───────────────────────────────────────────────
        per_regime = {}
        for regime in REGIME_LIST:
            regime_str = regime.value if hasattr(regime, 'value') else str(regime)
            true_mask_r = true_decisive == regime_str
            if true_mask_r.sum() == 0:
                per_regime[regime_str] = {"accuracy": 0.0, "total": 0}
                continue
            r_true = true_decisive[true_mask_r]
            r_pred = pred_decisive[true_mask_r]
            r_correct = (r_pred == r_true).sum()
            per_regime[regime_str] = {
                "accuracy": r_correct / len(r_true),
                "total": int(len(r_true)),
            }

        results = {
            "overall_accuracy": overall_acc,
            "total_samples": total,
            "coverage": coverage,
            "per_regime": per_regime,
        }

        logger.info(f"[RegimeDetector] Validasi selesai:")
        logger.info(f"  Coverage    : {coverage:.1%} (prediksi decisive)")
        logger.info(f"  Accuracy    : {overall_acc:.1%}")
        for r, v in per_regime.items():
            if v["total"] > 0:
                status = "OK" if v["accuracy"] >= 0.70 else "PERLU IMPROVEMENT"
                logger.info(f"  {r:<20}: {v['accuracy']:.1%} ({v['total']} samples) [{status}]")

        gate_pass = overall_acc >= 0.70
        logger.log(
            "SUCCESS" if gate_pass else "WARNING",
            f"[RegimeDetector] GATE Phase 2: {'LOLOS' if gate_pass else 'BELUM LOLOS'} "
            f"(target >= 70%, actual: {overall_acc:.1%})"
        )

        return results

    # ── Save & Load ───────────────────────────────────────────────────────────

    def save(self) -> bool:
        """Simpan model ke disk."""
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.hmm_model, f)

            with open(SCALER_PATH, "wb") as f:
                pickle.dump(self.scaler, f)

            with open(STATE_MAP_PATH, "wb") as f:
                pickle.dump(self.state_map, f)

            if self.rf_model is not None:
                with open(RF_MODEL_PATH, "wb") as f:
                    pickle.dump((self.rf_model, self.rf_scaler, self.label_encoder), f)

            logger.success(f"[RegimeDetector] Model disimpan ke {MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"[RegimeDetector] Gagal simpan model: {e}")
            return False

    def load(self) -> bool:
        """Muat model dari disk."""
        try:
            if not MODEL_PATH.exists():
                logger.warning(f"[RegimeDetector] Model tidak ditemukan di {MODEL_PATH}")
                return False

            with open(MODEL_PATH, "rb") as f:
                self.hmm_model = pickle.load(f)

            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)

            with open(STATE_MAP_PATH, "rb") as f:
                self.state_map = pickle.load(f)

            # Load RF model jika ada
            if RF_MODEL_PATH.exists():
                with open(RF_MODEL_PATH, "rb") as f:
                    self.rf_model, self.rf_scaler, self.label_encoder = pickle.load(f)
                logger.success("[RegimeDetector] RF model dimuat.")

            self._is_trained = True
            logger.success(f"[RegimeDetector] Model dimuat dari {MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"[RegimeDetector] Gagal muat model: {e}")
            return False

    @property
    def is_trained(self) -> bool:
        return self._is_trained


# Singleton
regime_detector = RegimeDetector()
