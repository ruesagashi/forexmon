"""
tests/test_phase67.py
Unit tests untuk Phase 6 (UCB1) & Phase 7 (Scheduler).
"""

import pytest
from unittest.mock import patch, MagicMock

from core.pool_manager import SpecialistPoolManager
from core.scheduler import Scheduler

class TestPhase6UCB1:
    @patch('core.pool_manager.db.get_specialists_by_regime')
    @patch('core.pool_manager.SpecialistPoolManager.get_specialist_object')
    def test_ucb1_selection(self, mock_get_obj, mock_get_db):
        pm = SpecialistPoolManager()
        
        # Spec 1: Banyak trade (100), WR lumayan (60%)
        # Spec 2: Sedikit trade (10), WR tinggi (80%)
        # Spec 3: Sedang trade (50), WR tinggi (75%)
        
        mock_get_db.return_value = [
            {"id": "spec1", "total_trades": 100, "winrate": 0.60},
            {"id": "spec2", "total_trades": 10, "winrate": 0.80},
            {"id": "spec3", "total_trades": 50, "winrate": 0.75},
        ]
        
        # Total N = 160.
        # spec1: 0.60 + 0.5 * sqrt(ln(160) / 100) = 0.60 + 0.5 * sqrt(5.07 / 100) = 0.60 + 0.5 * 0.225 = 0.712
        # spec2: 0.80 + 0.5 * sqrt(5.07 / 10) = 0.80 + 0.5 * 0.712 = 1.156
        # spec3: 0.75 + 0.5 * sqrt(5.07 / 50) = 0.75 + 0.5 * 0.318 = 0.909
        
        # Yang menang harusnya spec2 karena WR tinggi dan trades sedikit (explore)
        mock_get_obj.return_value = "OBJ_SPEC2"
        
        best = pm.get_best_specialist_for_regime("TRENDING_UP")
        
        mock_get_obj.assert_called_once_with("spec2")
        assert best == "OBJ_SPEC2"


class TestPhase7Scheduler:
    @patch('core.scheduler.connector')
    @patch('core.scheduler.execution_engine')
    @patch('core.scheduler.time.sleep', side_effect=InterruptedError) # Break infinite loop
    def test_scheduler_cycle(self, mock_sleep, mock_engine, mock_connector):
        sched = Scheduler()
        
        mock_connector.ensure_connected.return_value = True
        
        # Mock dataframe dengan index time
        import pandas as pd
        from datetime import datetime, timedelta
        
        now = datetime.now()
        df1 = pd.DataFrame({'close': [1,2,3]}, index=[now - timedelta(hours=2), now - timedelta(hours=1), now])
        mock_connector.get_candles.return_value = df1
        
        try:
            sched.start()
        except InterruptedError:
            pass
            
        assert sched.last_candle_time == now
        # Call pertama cuma inisialisasi last_candle_time, belum panggil run_cycle
        assert mock_engine.run_cycle.call_count == 0
