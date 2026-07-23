"""
tests/test_phase45.py
Unit tests untuk Phase 4 & 5 (Slot Manager & Risk Manager).
"""

import pytest
from unittest.mock import patch, MagicMock

from core.slot_manager import SlotManager
from core.risk_manager import RiskManager
from config.settings import settings


class TestSlotManager:
    @patch('core.slot_manager.connector.get_open_positions')
    @patch('core.slot_manager.pool_manager.get_specialist_object')
    def test_get_open_slots(self, mock_get_spec, mock_get_positions):
        sm = SlotManager()
        
        # Simulasi 3 posisi open
        # Pos 1: APPROVED, Pos 2: PROBATION, Pos 3: Manual (unknown)
        mock_get_positions.return_value = [
            {"comment": "spec_111"},
            {"comment": "spec_222"},
            {"comment": "manual_trade"}
        ]
        
        # Mock spesialis
        def mock_spec(spec_id):
            spec = MagicMock()
            if spec_id == "111":
                spec.status = "APPROVED"
            elif spec_id == "222":
                spec.status = "PROBATION"
            return spec
            
        mock_get_spec.side_effect = mock_spec
        
        slots = sm.get_open_slots()
        
        assert slots["APPROVED"] == 29   # 30 - 1
        assert slots["PROBATION"] == 14  # 15 - 1
        assert slots["TOTAL_USED"] == 3

    @patch('core.slot_manager.SlotManager.get_open_slots')
    def test_has_available_slot(self, mock_slots):
        sm = SlotManager()
        
        # Test saat APPROVED habis tapi BUFFER ada
        mock_slots.return_value = {
            "APPROVED": 0,
            "PROBATION": 10,
            "BUFFER": 5
        }
        assert sm.has_available_slot("APPROVED") is True
        
        # Test saat semua habis
        mock_slots.return_value = {
            "APPROVED": 0,
            "PROBATION": 0,
            "BUFFER": 0
        }
        assert sm.has_available_slot("APPROVED") is False
        assert sm.has_available_slot("PROBATION") is False


class TestRiskManager:
    @patch('core.risk_manager.connector.get_account_info')
    @patch('core.risk_manager.connector.get_symbol_info')
    def test_calculate_lot_size(self, mock_symbol, mock_account):
        rm = RiskManager()
        
        # Balance $10,000. Risk 1% = $100.
        mock_account.return_value = {"balance": 10000.0}
        mock_symbol.return_value = {
            "trade_contract_size": 100.0,
            "volume_step": 0.01,
            "volume_min": 0.01,
            "volume_max": 100.0
        }
        
        # SL distance = 10 points (misal $10 pergerakan gold)
        # Loss per 1 lot = 10 * 100 = $1000.
        # Target loss = $100.
        # Lot = 100 / 1000 = 0.10 lot
        lot = rm.calculate_lot_size(sl_distance_points=10.0)
        assert lot == 0.10
        
        # SL distance = 5 points -> loss/lot = 500 -> Lot = 100 / 500 = 0.20
        lot2 = rm.calculate_lot_size(5.0)
        assert lot2 == 0.20

    @patch('core.risk_manager.db._get_conn')
    @patch('core.risk_manager.connector.get_account_info')
    def test_check_daily_limit_passed(self, mock_account, mock_get_conn):
        rm = RiskManager()
        mock_account.return_value = {"balance": 10000.0}
        
        # Simulasi db mengembalikan daily PnL -100 (-1%)
        mock_conn = MagicMock()
        mock_conn.execute().fetchall.return_value = [{"pnl": -100.0}]
        mock_get_conn.return_value = mock_conn
        
        assert rm.check_daily_limit() is True

    @patch('core.risk_manager.db._get_conn')
    @patch('core.risk_manager.connector.get_account_info')
    def test_check_daily_limit_failed(self, mock_account, mock_get_conn):
        rm = RiskManager()
        mock_account.return_value = {"balance": 10000.0}
        
        # Simulasi db mengembalikan daily PnL -400 (-4%, melebih max 3%)
        mock_conn = MagicMock()
        mock_conn.execute().fetchall.return_value = [{"pnl": -400.0}]
        mock_get_conn.return_value = mock_conn
        
        assert rm.check_daily_limit() is False
