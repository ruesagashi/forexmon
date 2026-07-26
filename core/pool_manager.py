"""
core/pool_manager.py
Mengelola Specialist Pool dan Fast-Kill Protocol.
Sesuai PRD Section 3.3, 3.4 & 3.5.
"""

from loguru import logger
from typing import List, Optional

from core.memory import db
from models.specialist import Specialist
from config.settings import settings


class SpecialistPoolManager:
    """
    Mengatur siklus hidup Specialist: dari PROBATION, evaluasi per trade (Fast-Kill),
    hingga APPROVED atau ELIMINATED.
    """

    def __init__(self):
        # In-memory cache untuk objek Specialist yang sedang di-load (agar tidak perlu load pickle berulang)
        self.loaded_specialists = {}

    def add_specialist(self, specialist: Specialist) -> bool:
        """
        Daftarkan specialist baru ke dalam database sebagai PROBATION.
        """
        success = db.add_specialist(
            specialist_id=specialist.id,
            regime_type=specialist.regime.name,
            symbol=specialist.symbol,
            model_path=specialist.save(),
            features_used=specialist.feature_cols,
            metadata={"composite_score": specialist.composite_score}
        )
        if success:
            self.loaded_specialists[specialist.id] = specialist
        return success

    def get_specialist_object(self, specialist_id: str) -> Optional[Specialist]:
        """Ambil objek ML Specialist berdasarkan ID (dengan in-memory caching)."""
        if specialist_id in self.loaded_specialists:
            return self.loaded_specialists[specialist_id]
            
        spec_db = db.get_specialist(specialist_id)
        if not spec_db or not spec_db["model_path"]:
            return None
            
        try:
            spec = Specialist.load(spec_db["model_path"])
            # Sinkronisasi status/metrics dengan DB
            spec.status = spec_db["status"]
            spec.win_rate = spec_db["winrate"]
            spec.profit_factor = spec_db["profit_factor"]
            spec.trades_count = spec_db["total_trades"]
            
            self.loaded_specialists[specialist_id] = spec
            return spec
        except Exception as e:
            logger.error(f"[PoolManager] Gagal meload specialist {specialist_id}: {e}")
            return None

    def get_best_specialist_for_regime(self, regime_name: str, symbol: str) -> Optional[Specialist]:
        """
        Ambil specialist APPROVED terbaik untuk regime saat ini menggunakan algoritma UCB1 (Multi-Armed Bandit).
        Jika tidak ada APPROVED, coba ambil dari PROBATION (highest winrate).
        """
        # Prioritas 1: APPROVED dengan UCB1
        approved = db.get_specialists_by_regime(regime_name, status="APPROVED", symbol=symbol)
        if approved:
            import math
            # N = total trades dari semua approved di regime ini
            total_N = sum(max(1, spec["total_trades"]) for spec in approved)
            c = 0.5  # Konstanta UCB1
            
            best_spec = None
            best_score = -999.0
            
            for spec in approved:
                n_i = max(1, spec["total_trades"])
                wr_i = spec["winrate"]
                
                # UCB1 Formula: WR + c * sqrt(ln(N) / n_i)
                ucb1_score = wr_i + c * math.sqrt(math.log(total_N) / n_i)
                
                if ucb1_score > best_score:
                    best_score = ucb1_score
                    best_spec = spec
                    
            if best_spec:
                return self.get_specialist_object(best_spec["id"])
            
        # Prioritas 2: PROBATION (Explore Murni)
        probation = db.get_specialists_by_regime(regime_name, status="PROBATION", symbol=symbol)
        if probation:
            best_id = max(probation, key=lambda x: x["winrate"])["id"]
            return self.get_specialist_object(best_id)
            
        return None

    def evaluate_fast_kill(self, specialist_id: str):
        """
        Jalankan Fast-Kill Protocol (Stage 2 & 3) untuk sebuah specialist setelah trade selesai.
        Akan membaca riwayat trade-nya di DB, lalu update status jika perlu.
        """
        spec = db.get_specialist(specialist_id)
        if not spec:
            return

        status = spec["status"]
        if status in ["ELIMINATED", "SUSPENDED"]:
            return  # Tidak perlu evaluasi jika sudah mati/suspend

        # Ambil trade history
        trades = db.get_recent_trades(specialist_id, limit=20)
        
        if not trades:
            return
        # Urutkan dari terlama ke terbaru dalam sample ini
        trades.reverse()
        
        total_trades = spec["total_trades"]  # Total keseluruhan
        wins = sum(1 for t in trades if t["pnl"] > 0)
        recent_wr = wins / len(trades) if trades else 0.0
        
        # 1. Evaluasi Stage 2: PROBATION -> APPROVED / ELIMINATED
        if status == "PROBATION":
            self._evaluate_probation(specialist_id, total_trades, trades)
            
        # 2. Evaluasi Stage 3: APPROVED -> WARNING / SUSPENDED
        elif status in ["APPROVED", "WARNING"]:
            self._evaluate_approved(specialist_id, status, trades, recent_wr)

    def _evaluate_probation(self, specialist_id: str, total_trades: int, trades: list):
        """Evaluasi Stage 2 Fast-Kill"""
        if total_trades < 3:
            return
            
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = wins / len(trades)
        
        # Trade ke-1 s/d 3: Loss semua 3 berturut-turut
        if 3 <= total_trades < 5 and wins == 0:
            # Check if last 3 trades are losses
            last_3_wins = sum(1 for t in trades[-3:] if t["pnl"] > 0)
            if last_3_wins == 0:
                db.update_specialist_status(specialist_id, "ELIMINATED")
                logger.warning(f"[Fast-Kill] {specialist_id} ELIMINATED: 3 loss beruntun awal.")
                from monitoring.alerting import telegram_alerter
                telegram_alerter.send_alert(f"💀 <b>SPECIALIST KILLED</b>\nID: {specialist_id}\nReason: 3 consecutive initial losses.")
                return
            
        # Trade ke-5: WR < 40%
        if 5 <= total_trades < 10 and wr < 0.40:
            db.update_specialist_status(specialist_id, "ELIMINATED")
            logger.warning(f"[Fast-Kill] {specialist_id} ELIMINATED: WR {wr:.1%} < 40% di trade 5.")
            return
            
        # Trade ke-10: WR < 50%
        if 10 <= total_trades < 15 and wr < 0.50:
            db.update_specialist_status(specialist_id, "ELIMINATED")
            logger.warning(f"[Fast-Kill] {specialist_id} ELIMINATED: WR {wr:.1%} < 50% di trade 10.")
            return
            
        # Trade ke-15: WR < 55%
        if 15 <= total_trades < 20 and wr < 0.55:
            db.update_specialist_status(specialist_id, "ELIMINATED")
            logger.warning(f"[Fast-Kill] {specialist_id} ELIMINATED: WR {wr:.1%} < 55% di trade 15.")
            return
            
        # Trade ke-20: Penentuan Lulus
        if total_trades >= 20:
            # Hitung Profit Factor dari trades ini
            gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 99.0
            
            if wr < 0.60 or pf < 1.3:
                db.update_specialist_status(specialist_id, "ELIMINATED")
                logger.warning(f"[Fast-Kill] {specialist_id} ELIMINATED di trade 20: WR {wr:.1%} atau PF {pf:.2f} di bawah target.")
            else:
                db.update_specialist_status(specialist_id, "APPROVED")
                logger.success(f"[Fast-Kill] {specialist_id} LULUS PROBATION! Status -> APPROVED.")

    def _evaluate_approved(self, specialist_id: str, current_status: str, recent_20_trades: list, recent_wr: float):
        """Evaluasi Stage 3 Monitoring"""
        if len(recent_20_trades) < 20:
            return  # Tunggu sampai ada window 20 trade
            
        if current_status == "APPROVED":
            if recent_wr < 0.60:
                db.update_specialist_status(specialist_id, "SUSPENDED")
                logger.error(f"[PoolManager] {specialist_id} SUSPENDED: Recent WR drop ke {recent_wr:.1%}")
                from monitoring.alerting import telegram_alerter
                telegram_alerter.send_alert(f"⚠️ <b>SPECIALIST SUSPENDED</b>\nID: {specialist_id}\nReason: WR drops below 60% ({recent_wr:.1%})")
            elif recent_wr < 0.70:
                db.update_specialist_status(specialist_id, "WARNING")
                logger.warning(f"[PoolManager] {specialist_id} WARNING: Recent WR drop ke {recent_wr:.1%}")
                
        elif current_status == "WARNING":
            if recent_wr > 0.75:
                db.update_specialist_status(specialist_id, "APPROVED")
                logger.success(f"[PoolManager] {specialist_id} RECOVERED -> APPROVED. Recent WR {recent_wr:.1%}")
            elif recent_wr < 0.60:
                db.update_specialist_status(specialist_id, "SUSPENDED")
                logger.error(f"[PoolManager] {specialist_id} SUSPENDED: WR drop ke {recent_wr:.1%}")

pool_manager = SpecialistPoolManager()
