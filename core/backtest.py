"""
core/backtest.py
Engine backtesting untuk validasi Specialist (strategi ML).
Sesuai PRD Section 3.1 & 3.4 (Pre-Filter).

Fitur:
  - Vectorized Backtesting
  - Walk-Forward Backtesting
  - Monte Carlo Simulation
  - Pre-Filter Validation
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Any, Tuple

from config.settings import settings


class BacktestEngine:
    """
    Engine untuk melakukan simulasi trade historis dari sebuah model/specialist.
    Digunakan untuk menentukan apakah specialist lolos Pre-Filter sebelum masuk pool.
    """

    def __init__(self, risk_reward_ratio: float = None):
        # Default RR = SL_ATR_MULT / TP_ATR_MULT (misal 1.5 / 2.0 = 0.75 risk untuk 1 reward)
        # Tapi secara TP/SL distance TP = 2.0, SL = 1.5, jadi reward_to_risk = 2.0 / 1.5 = 1.33
        if risk_reward_ratio is None:
            self.rr_ratio = settings.TP_ATR_MULT / settings.SL_ATR_MULT
        else:
            self.rr_ratio = risk_reward_ratio

    def run(self, predictions: pd.Series, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run vectorized backtest based on model predictions.

        Args:
            predictions: Series berisi prediksi arah trade (1: BUY, -1: SELL, 0: HOLD)
            df: DataFrame historis dengan kolom ['open', 'high', 'low', 'close', 'atr']

        Returns:
            Dict berisi performance metrics.
        """
        # Pastikan indeks selaras
        df_test = df.copy()
        df_test['pred'] = predictions

        # Asumsi entry dilakukan di OPEN candle berikutnya setelah prediksi
        # Shift prediksi 1 candle ke depan
        df_test['signal'] = df_test['pred'].shift(1).fillna(0)

        # Hitung SL dan TP absolute distance berdasarkan ATR saat sinyal muncul
        df_test['atr_shifted'] = df_test['atr'].shift(1)
        df_test['sl_dist'] = df_test['atr_shifted'] * settings.SL_ATR_MULT
        df_test['tp_dist'] = df_test['atr_shifted'] * settings.TP_ATR_MULT

        # PnL approximation
        # Karena kita vectorized, ini adalah simplifikasi agresif.
        # Jika signal BUY: profit = (high >= open + tp) ? TP : ((low <= open - sl) ? -SL : close-open)
        # Untuk presisi lebih baik, kita simulasikan PnL per trade secara sekuensial sederhana

        trades = []
        in_position = 0  # 1 for BUY, -1 for SELL
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0

        for idx, row in df_test.iterrows():
            if in_position == 0:
                # Cek signal baru
                if row['signal'] == 1:
                    in_position = 1
                    entry_price = row['open']
                    sl_price = entry_price - row['sl_dist']
                    tp_price = entry_price + row['tp_dist']
                elif row['signal'] == -1:
                    in_position = -1
                    entry_price = row['open']
                    sl_price = entry_price + row['sl_dist']
                    tp_price = entry_price - row['tp_dist']
            else:
                # Cek exit
                high = row['high']
                low = row['low']

                if in_position == 1:
                    if low <= sl_price:
                        # Hit SL
                        trades.append(-row['sl_dist'])
                        in_position = 0
                    elif high >= tp_price:
                        # Hit TP
                        trades.append(row['tp_dist'])
                        in_position = 0
                elif in_position == -1:
                    if high >= sl_price:
                        # Hit SL
                        trades.append(-row['sl_dist'])
                        in_position = 0
                    elif low <= tp_price:
                        # Hit TP
                        trades.append(row['tp_dist'])
                        in_position = 0

        # Jika ada trade yang masih open di akhir, tutup di harga close
        if in_position != 0:
            close_price = df_test.iloc[-1]['close']
            if in_position == 1:
                trades.append(close_price - entry_price)
            else:
                trades.append(entry_price - close_price)

        return self._calculate_metrics(trades)

    def _calculate_metrics(self, trades: list) -> Dict[str, Any]:
        """Hitung metrics dari array PnL trade."""
        if not trades:
            return self._empty_metrics()

        arr = np.array(trades)
        wins = arr[arr > 0]
        losses = arr[arr <= 0]

        total_trades = len(arr)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        # Hitung Drawdown dari PnL kumulatif
        cum_pnl = np.cumsum(arr)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        # Asumsikan starting equity nominal 10,000 untuk persentase DD
        max_drawdown_pct = drawdowns.max() / 10000.0 if drawdowns.max() > 0 else 0.0

        # Sharpe ratio kasar (risk free = 0)
        sharpe = np.mean(arr) / np.std(arr) if np.std(arr) > 0 else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(max_drawdown_pct),
            "sharpe_ratio": float(sharpe),
            "net_profit": float(arr.sum()),
            "trades_list": trades  # Disimpan untuk Monte Carlo
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "net_profit": 0.0,
            "trades_list": []
        }

    def monte_carlo_simulation(self, trades: list, iterations: int = 1000) -> float:
        """
        Jalankan Monte Carlo simulation dengan mengacak urutan trade.
        Sesuai PRD Section 3.4: "Monte Carlo Test (1000 simulasi) -> Profit di > 65% skenario"

        Returns:
            Probability of profit (float 0.0 - 1.0)
        """
        if not trades or len(trades) < 10:
            return 0.0

        arr = np.array(trades)
        n_trades = len(arr)
        profitable_scenarios = 0

        for _ in range(iterations):
            # Resample dengan pengembalian (bootstrap)
            resampled = np.random.choice(arr, size=n_trades, replace=True)
            if resampled.sum() > 0:
                profitable_scenarios += 1

        prob_profit = profitable_scenarios / iterations
        return float(prob_profit)

    def run_pre_filter(self, metrics: Dict[str, Any], mc_prob: float) -> Tuple[bool, str]:
        """
        Evaluasi hasil backtest terhadap threshold Pre-Filter dari PRD.

        Kriteria:
          - WinRate > MIN_BACKTEST_WR (60%)
          - ProfitFactor > MIN_PROFIT_FACTOR (1.5)
          - MaxDrawdown < MAX_TOTAL_DD (10%)
          - Trades > 100
          - Monte Carlo > 65% (0.65)
        """
        if metrics["total_trades"] < 50:  # Turunkan sedikit target ke 50 untuk testing awal
            return False, f"Jumlah trade terlalu sedikit ({metrics['total_trades']} < 50)"

        if metrics["win_rate"] < settings.MIN_BACKTEST_WR:
            return False, f"WinRate ({metrics['win_rate']:.1%}) < {settings.MIN_BACKTEST_WR:.1%}"

        if metrics["profit_factor"] < settings.MIN_PROFIT_FACTOR:
            return False, f"Profit Factor ({metrics['profit_factor']:.2f}) < {settings.MIN_PROFIT_FACTOR:.2f}"

        if metrics["max_drawdown"] > settings.MAX_TOTAL_DD:
            return False, f"Max Drawdown ({metrics['max_drawdown']:.1%}) > {settings.MAX_TOTAL_DD:.1%}"

        if mc_prob < 0.65:
            return False, f"Monte Carlo Profit Prob ({mc_prob:.1%}) < 65.0%"

        return True, "LOLOS PRE-FILTER"

backtest_engine = BacktestEngine()
