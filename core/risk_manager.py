"""
core/risk_manager.py
Manage position sizing, SL/TP calculation, dan monitoring.
SL/TP sekarang AWARE dengan REGIME untuk optimal RR ratio.
"""

from loguru import logger
from config.settings import settings


class RiskManager:
    """
    Kelola risk per trade dengan strict rules:
    - Max 1-2% risk per trade
    - SL/TP berdasarkan regime
    - Auto-adjust kalau balance besar/kecil
    """

    def __init__(self):
        self.total_risk_today = 0.0
        self.trades_today = 0

    def calculate_position_size(self, balance: float, risk_pct: float, sl_pips: float) -> float:
        """
        Hitung lot size otomatis berdasarkan risk, balance, dan SL distance.
        """
        if sl_pips == 0:
            return 0.01  # Fallback minimum

        risk_amount = balance * risk_pct
        pips_value = 0.1  # 1 pip = $0.1 untuk XAUUSD
        
        lot_size = risk_amount / (sl_pips * pips_value)
        
        # Enforce min/max
        min_lot = 0.01
        max_lot = 0.1
        
        lot_size = max(min_lot, min(lot_size, max_lot))
        
        logger.debug(f"[RiskManager] Position size: {lot_size:.3f} lot (risk=${risk_amount:.2f}, SL={sl_pips:.1f}pips)")
        
        return lot_size

    def calculate_sl_tp(self, entry_price: float, direction: str, atr: float, regime: str = None, stop_loss: float = None, take_profit: float = None) -> tuple:
        """
        Hitung SL dan TP berdasarkan ATR, direction, dan REGIME.
        Setiap regime punya optimal SL/TP ratio untuk maximize expectancy.
        
        REGIME SETTINGS:
        - REVERSAL: tight SL (0.8), tight TP (0.8) → RR 1:1, quick close
        - TRENDING: medium SL (1.0), lebar TP (1.5) → RR 1:1.5
        - RANGING: tight SL (0.6), medium TP (0.9) → RR 1:1.5, low volatility
        - BREAKOUT: tight SL (0.6), sangat lebar TP (2.5) → RR 1:4.17, big move potential
        """
        
        # Kalau sudah set custom SL/TP, pakai itu
        if stop_loss and take_profit:
            return stop_loss, take_profit

        # Regime-specific SL/TP multipliers (optimal untuk M5)
        regime_settings = {
            "REVERSAL": {"sl": 0.8, "tp": 0.8},           # RR 1:1
            "TRENDING_UP": {"sl": 1.0, "tp": 1.5},        # RR 1:1.5
            "TRENDING_DOWN": {"sl": 1.0, "tp": 1.5},      # RR 1:1.5
            "RANGING": {"sl": 0.6, "tp": 0.9},            # RR 1:1.5
            "BREAKOUT": {"sl": 0.6, "tp": 2.5},           # RR 1:4.17
        }
        
        # Get settings untuk regime ini
        if regime and regime in regime_settings:
            multipliers = regime_settings[regime]
            sl_mult = multipliers["sl"]
            tp_mult = multipliers["tp"]
            logger.debug(f"[RiskManager] {regime}: SL={sl_mult}×ATR, TP={tp_mult}×ATR (RR 1:{tp_mult/sl_mult:.2f})")
        else:
            # Fallback ke settings.py (balanced untuk semua regime)
            sl_mult = settings.SL_ATR_MULT
            tp_mult = settings.TP_ATR_MULT
            logger.debug(f"[RiskManager] No regime match, using default: SL={sl_mult}×ATR, TP={tp_mult}×ATR")

        # Calculate distances
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        if direction.upper() == "BUY":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # SELL
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        return sl, tp

    def check_daily_limit(self, current_dd: float) -> bool:
        """
        Check kalau DD hari ini sudah capai limit.
        """
        if current_dd > settings.MAX_DAILY_DD:
            logger.warning(f"[RiskManager] Daily DD {current_dd:.1%} exceed limit {settings.MAX_DAILY_DD:.1%}. STOP trading!")
            return False
        
        if current_dd > settings.MAX_TOTAL_DD:
            logger.error(f"[RiskManager] EMERGENCY! Total DD {current_dd:.1%} exceed {settings.MAX_TOTAL_DD:.1%}. Emergency stop!")
            return False
        
        return True

    def validate_trade(self, lot_size: float, entry_price: float, sl: float, tp: float) -> bool:
        """
        Validate sebelum entry.
        """
        # Check lot size
        if lot_size < 0.01 or lot_size > 0.1:
            logger.error(f"[RiskManager] Invalid lot size: {lot_size}")
            return False
        
        # Check SL != TP
        if abs(sl - tp) < 1:
            logger.error(f"[RiskManager] SL dan TP terlalu dekat")
            return False
        
        # Check entry valid
        if entry_price == 0:
            logger.error(f"[RiskManager] Invalid entry price")
            return False
        
        return True

# Singleton instance
risk_manager = RiskManager()
