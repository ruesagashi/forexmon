"""
core/risk_manager.py
Manajemen risiko untuk menghitung Lot Size, SL/TP absolut, dan proteksi Drawdown harian.
Sesuai PRD Section 3.6.
"""

from datetime import datetime, timedelta
from loguru import logger

from execution.mt5_connector import connector
from core.memory import db
from config.settings import settings


class RiskManager:
    def __init__(self):
        self.cooldown_until = None

    def check_daily_limit(self) -> bool:
        """
        Cek apakah kita sudah menyentuh Max Daily Drawdown (3%).
        Atau apakah sedang dalam cooldown karena loss streak.
        Returns:
            True jika aman trading, False jika dilarang trading hari ini.
        """
        # Cek Cooldown loss streak
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            logger.warning(f"[RiskManager] Sedang COOLDOWN sampai {self.cooldown_until}. Tidak boleh trade.")
            return False

        # Cek Daily Drawdown dari eksekusi trade hari ini di DB
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        conn = db._get_conn()
        rows = conn.execute(
            "SELECT pnl FROM trades WHERE close_at >= ?", (today_start,)
        ).fetchall()
        
        daily_pnl = sum(r["pnl"] for r in rows if r["pnl"] is not None)
        
        # Ambil balance akun dari MT5
        account = connector.get_account_info()
        if not account:
            return False  # Safety fallback
            
        balance = account["balance"]
        
        # Jika PnL minus > 3% dari balance, freeze
        max_loss_allowed = balance * settings.MAX_DAILY_DD
        
        if daily_pnl < 0 and abs(daily_pnl) >= max_loss_allowed:
            logger.error(f"[RiskManager] DAILY DRAWDOWN LIMIT TERLAMPAUI! PnL: {daily_pnl:.2f}, Limit: -{max_loss_allowed:.2f}")
            return False
            
        return True

    def calculate_lot_size(self, symbol: str, sl_distance_points: float) -> float:
        """
        Hitung ukuran lot berdasarkan Risk % dan jarak SL.
        Asumsi XAUUSD standard contract size = 100
        1 pip = 0.01 harga, 1 point = 0.01 (tergantung broker, let's pull from MT5)
        """
        if sl_distance_points <= 0:
            return 0.01  # fallback min lot
            
        account = connector.get_account_info()
        symbol_info = connector.get_symbol_info(symbol)
        
        if not account or not symbol_info:
            return 0.01

        balance = account["balance"]
        risk_amount = balance * settings.RISK_PER_TRADE
        
        # Kalkulasi pergerakan 1 lot.
        # XAUUSD: 1 lot = 100 oz. Pergerakan harga $1 = $100 profit/loss untuk 1 lot.
        # sl_distance_points di sini adalah selisih harga absolut, misal open 2400, sl 2390 -> distance = 10.
        # Loss per 1 lot = distance * contract_size
        contract_size = symbol_info.get("trade_contract_size", 100.0)
        loss_per_1_lot = sl_distance_points * contract_size
        
        if loss_per_1_lot <= 0:
            return 0.01
            
        calculated_lot = risk_amount / loss_per_1_lot
        
        # Rounding down to nearest volume_step
        vol_step = symbol_info.get("volume_step", 0.01)
        vol_min = symbol_info.get("volume_min", 0.01)
        vol_max = symbol_info.get("volume_max", 100.0)
        
        # Pembulatan konservatif ke bawah
        steps = int(calculated_lot / vol_step)
        final_lot = steps * vol_step
        
        # Clamp ke min/max
        final_lot = max(vol_min, min(vol_max, final_lot))
        
        return round(final_lot, 2)

    def trigger_loss_streak_cooldown(self):
        """Memicu cooldown 1 jam jika terjadi loss streak."""
        self.cooldown_until = datetime.utcnow() + timedelta(hours=1)
        logger.error(f"[RiskManager] LOSS STREAK TERDETEKSI! Cooldown 1 jam aktif.")

risk_manager = RiskManager()
