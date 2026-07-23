"""
tests/test_phase3.py
Unit tests untuk Phase 3: Specialist Pool & Backtest Engine.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from core.backtest import backtest_engine
from models.specialist import Specialist, SpecialistTrainer
from core.pool_manager import SpecialistPoolManager
from models.regime_detector import Regime


class TestBacktestEngine:
    def test_calculate_metrics(self):
        # Dummy trades (PnL array)
        trades = [10.0, 15.0, -5.0, -10.0, 20.0, -5.0]
        # Wins = 10+15+20 = 45. Losses = -5-10-5 = -20
        # WinRate = 3/6 = 50%
        # PF = 45 / 20 = 2.25
        # Net Profit = 25
        
        metrics = backtest_engine._calculate_metrics(trades)
        
        assert metrics["total_trades"] == 6
        assert metrics["win_rate"] == 0.5
        assert metrics["profit_factor"] == 2.25
        assert metrics["net_profit"] == 25.0

    def test_monte_carlo(self):
        # Dengan WR tinggi, Monte Carlo profit prob harus mendekati 1.0
        good_trades = [10.0] * 80 + [-5.0] * 20
        prob = backtest_engine.monte_carlo_simulation(good_trades, iterations=100)
        assert prob > 0.8  # Sangat mungkin untung

        # Dengan WR rendah, probabilitas rendah
        bad_trades = [10.0] * 20 + [-5.0] * 80
        prob = backtest_engine.monte_carlo_simulation(bad_trades, iterations=100)
        assert prob < 0.2

    def test_pre_filter(self):
        good_metrics = {
            "total_trades": 150,
            "win_rate": 0.65,
            "profit_factor": 1.6,
            "max_drawdown": 0.05
        }
        passed, msg = backtest_engine.run_pre_filter(good_metrics, mc_prob=0.70)
        assert passed is True

        bad_wr = good_metrics.copy()
        bad_wr["win_rate"] = 0.50
        passed, msg = backtest_engine.run_pre_filter(bad_wr, mc_prob=0.70)
        assert passed is False
        assert "WinRate" in msg


class TestSpecialistTrainer:
    def test_generate_labels(self):
        # Create dummy df
        df = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],
            'high': [105, 105, 105, 105, 105],
            'low': [95, 95, 95, 95, 95],
            'close': [100, 100, 100, 100, 100],
            'atr': [1.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        with patch('config.settings.settings.SL_ATR_MULT', 2.0), \
             patch('config.settings.settings.TP_ATR_MULT', 4.0):
            # TP = 4, SL = 2
            # Entry di 100. High = 105 -> (TP hit!). Low = 95 -> SL hit? SL is 98. So SL hits first if we just check high/low.
            # wait, our label logic checks SL/TP sequentially.
            trainer = SpecialistTrainer(['atr'])
            labels = trainer.generate_labels(df)
            assert len(labels) == 5

    def test_specialist_predict(self):
        spec = Specialist(Regime.TRENDING_UP, ['f1', 'f2'])
        # without model
        signal, conf = spec.predict(pd.DataFrame({'f1': [1], 'f2': [2]}))
        assert signal == 0
        assert conf == 0.0

        # with mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.1, 0.8, 0.1]]  # Class 1 (BUY) max
        spec.model = mock_model
        signal, conf = spec.predict(pd.DataFrame({'f1': [1], 'f2': [2]}))
        assert signal == 1
        assert conf == 0.8


class TestPoolManager:
    @patch('core.memory.db.get_specialist')
    @patch('core.memory.db._get_conn')
    @patch('core.memory.db.update_specialist_status')
    def test_fast_kill_probation_fail(self, mock_update, mock_get_conn, mock_get_spec):
        pm = SpecialistPoolManager()
        mock_get_spec.return_value = {"status": "PROBATION", "total_trades": 5}
        
        # Mock DB trades: 5 trades, 1 win, 4 losses -> WR 20%
        mock_conn = MagicMock()
        mock_conn.execute().fetchall.return_value = [
            {"pnl": 10}, {"pnl": -5}, {"pnl": -5}, {"pnl": -5}, {"pnl": -5}
        ]
        mock_get_conn.return_value = mock_conn

        pm.evaluate_fast_kill("spec_1")
        
        # WR < 40% di trade 5 -> ELIMINATED
        mock_update.assert_called_with("spec_1", "ELIMINATED")

    @patch('core.memory.db.get_specialist')
    @patch('core.memory.db._get_conn')
    @patch('core.memory.db.update_specialist_status')
    def test_fast_kill_probation_pass(self, mock_update, mock_get_conn, mock_get_spec):
        pm = SpecialistPoolManager()
        mock_get_spec.return_value = {"status": "PROBATION", "total_trades": 20}
        
        # 20 trades: 14 wins, 6 losses -> WR 70%, PF > 1.3
        trades = [{"pnl": 10}] * 14 + [{"pnl": -5}] * 6
        mock_conn = MagicMock()
        mock_conn.execute().fetchall.return_value = trades
        mock_get_conn.return_value = mock_conn

        pm.evaluate_fast_kill("spec_1")
        
        # WR > 60%, PF = 140 / 30 = 4.6 -> APPROVED
        mock_update.assert_called_with("spec_1", "APPROVED")
