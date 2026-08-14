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
        self.last_selected_specialist = None
        self.trade_counts_per_regime = {}
        self.last_regime_specialist = {}

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
        logger.debug(f"[PoolManager] Looking for {regime_name} specialist...")
        if regime_name not in self.trade_counts_per_regime:
            self.trade_counts_per_regime[regime_name] = 0
            
        force_regime_rotation = False
        if self.trade_counts_per_regime[regime_name] >= 5:
            force_regime_rotation = True
            self.trade_counts_per_regime[regime_name] = 0
            
        # Prioritas 1: APPROVED dengan UCB1
        approved = db.get_specialists_by_regime(regime_name, status="APPROVED", symbol=symbol)
        logger.debug(f"[PoolManager] Found {len(approved) if approved else 0} candidates")
        if approved:
            import math
            # N = total trades dari semua approved di regime ini
            total_N = sum(max(1, spec["total_trades"]) for spec in approved)
            c = 0.15  # Konstanta UCB1 (lebih exploitation, less exploration)
            
            scored_specs = []
            for spec in approved:
                n_i = max(1, spec["total_trades"])
                wr_i = spec["winrate"]
                
                # 3. Add safety check — jangan pick specialist dengan:
                if wr_i < 0.30 or n_i < 5:
                    continue
                    
                # 1. Naikin minimum trades sebelum explore aggressif:
                if n_i < 30:
                    if wr_i < 0.40:
                        continue
                
                # UCB1 Formula: WR + c * sqrt(ln(N) / n_i)
                ucb1_score = wr_i + c * math.sqrt(math.log(total_N) / n_i)
                scored_specs.append((ucb1_score, spec))
                
            logger.debug(f"[PoolManager] After filtering: {len(scored_specs)} specialists")
                
            # Sort by highest UCB1 score
            scored_specs.sort(key=lambda x: x[0], reverse=True)
            
            best_spec = None
            best_score = -999.0
            
            for score, spec in scored_specs:
                spec_id = spec["id"]
                
                # Cooldown 1: Jangan pick specialist sama 2x berturut secara global
                if spec_id == self.last_selected_specialist:
                    continue
                    
                # Cooldown 2: Rotasi per 5 trade di regime yang sama
                if force_regime_rotation and spec_id == self.last_regime_specialist.get(regime_name):
                    continue
                    
                best_spec = spec
                best_score = score
                break
                
            # Fallback: jika semua difilter (misal hanya ada 1 specialist), ambil yang terbaik saja
            if not best_spec and scored_specs:
                best_score, best_spec = scored_specs[0]
                
            if best_spec:
                spec_id = best_spec["id"]
                self.last_selected_specialist = spec_id
                self.last_regime_specialist[regime_name] = spec_id
                self.trade_counts_per_regime[regime_name] += 1
                
                logger.debug(f"[PoolManager] Selected: {spec_id} (WR={best_spec['winrate']:.1%})")
                return self.get_specialist_object(spec_id)
            else:
                logger.error(f"[PoolManager] NO SPECIALIST FOUND for {regime_name}!")
            
        # Prioritas 2: PROBATION (Explore Murni)
        probation = db.get_specialists_by_regime(regime_name, status="PROBATION", symbol=symbol)
        if probation:
            best_spec = max(probation, key=lambda x: x["winrate"])
            best_id = best_spec["id"]
            
            self.last_selected_specialist = best_id
            self.last_regime_specialist[regime_name] = best_id
            self.trade_counts_per_regime[regime_name] += 1
            
            logger.debug(f"[PoolManager] Selected: {best_id} (WR={best_spec['winrate']:.1%})")
            return self.get_specialist_object(best_id)
            
        logger.error(f"[PoolManager] NO SPECIALIST FOUND for {regime_name}!")
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
            
        # Trade ke-5: WR < 35% (Stricter threshold)
        if 5 <= total_trades < 10 and wr < 0.35:
            db.update_specialist_status(specialist_id, "ELIMINATED")
            logger.warning(f"[Fast-Kill] {specialist_id} ELIMINATED: WR {wr:.1%} < 35% di trade 5.")
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

    def _evaluate_approved(self, spec: dict, recent_20_trades: list, recent_wr: float):
        """Evaluasi Stage 3 Monitoring & Performance Decay"""
        specialist_id = spec["id"]
        current_status = spec["status"]
        overall_wr = spec["winrate"]
        
        # 1. Performance Decay Detection (Track last 10 trades)
        if len(recent_20_trades) >= 10:
            recent_10 = recent_20_trades[:10]
            recent_10_wins = sum(1 for t in recent_10 if t["pnl"] > 0)
            recent_10_wr = recent_10_wins / 10.0
            
            if current_status == "APPROVED":
                if overall_wr - recent_10_wr > 0.15:
                    db.update_specialist_status(specialist_id, "WARNING", reason="WR degradation detected")
                    logger.warning(f"[PoolManager] {specialist_id} WR degradation detected: Overall {overall_wr:.1%} vs Recent 10 {recent_10_wr:.1%}")
                    current_status = "WARNING"
                elif recent_10_wr < 0.55:
                    # 3. Add APPROVED threshold ketat
                    db.update_specialist_status(specialist_id, "WARNING", reason="Approved specialist underperforming: WR < 55% in recent 10")
                    logger.warning(f"[PoolManager] {specialist_id} Approved specialist underperforming: WR {recent_10_wr:.1%} < requirement")
                    current_status = "WARNING"
            
            elif current_status == "WARNING":
                # Check 5 next trades to see if it kept dropping.
                if len(recent_20_trades) >= 5:
                    recent_5 = recent_20_trades[:5]
                    recent_5_wins = sum(1 for t in recent_5 if t["pnl"] > 0)
                    recent_5_wr = recent_5_wins / 5.0
                    
                    if recent_5_wr < recent_10_wr:
                        db.update_specialist_status(specialist_id, "SUSPENDED", reason="Continued WR decline in WARNING state")
                        logger.error(f"[PoolManager] {specialist_id} SUSPENDED: Continued decline. Recent 5 WR {recent_5_wr:.1%} < Recent 10 WR {recent_10_wr:.1%}")
                        return

        if len(recent_20_trades) < 20:
            return  # Tunggu sampai ada window 20 trade
            
        if current_status == "APPROVED":
            if recent_wr < 0.60:
                db.update_specialist_status(specialist_id, "SUSPENDED", reason=f"WR drops below 60% ({recent_wr:.1%})")
                logger.error(f"[PoolManager] {specialist_id} SUSPENDED: Recent WR drop ke {recent_wr:.1%}")
                from monitoring.alerting import telegram_alerter
                telegram_alerter.send_alert(f"⚠️ <b>SPECIALIST SUSPENDED</b>\nID: {specialist_id}\nReason: WR drops below 60% ({recent_wr:.1%})")
            elif recent_wr < 0.70:
                db.update_specialist_status(specialist_id, "WARNING", reason=f"WR drops below 70% ({recent_wr:.1%})")
                logger.warning(f"[PoolManager] {specialist_id} WARNING: Recent WR drop ke {recent_wr:.1%}")
                
        elif current_status == "WARNING":
            if recent_wr > 0.75:
                db.update_specialist_status(specialist_id, "APPROVED", reason="Recovered WR > 75%")
                logger.success(f"[PoolManager] {specialist_id} RECOVERED -> APPROVED. Recent WR {recent_wr:.1%}")
            elif recent_wr < 0.60:
                db.update_specialist_status(specialist_id, "SUSPENDED", reason=f"WR drop ke {recent_wr:.1%}")
                logger.error(f"[PoolManager] {specialist_id} SUSPENDED: WR drop ke {recent_wr:.1%}")

    def generate_daily_report(self):
        """
        Generate Enhanced Daily Report
        """
        from datetime import datetime
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        
        trades_today = db.get_trades_today()
        total_trades = len(trades_today)
        approved_count = db.count_specialists("APPROVED")
        
        events_today = db.get_events_today()
        # Cari PROBATION via log
        new_specialists = sum(1 for e in events_today if e["event_type"] == "STATUS_CHANGE" and "PROBATION" in (e.get("description") or ""))
        eliminated_today = sum(1 for e in events_today if e["event_type"] == "STATUS_CHANGE" and "ELIMINATED" in (e.get("description") or ""))
        
        warning_specs = db.get_specialists_by_status("WARNING")
        warning_count = len(warning_specs)
        
        # Fetch reason from recent events for the warnings
        warning_details_list = []
        for s in warning_specs:
            spec_id = s["id"]
            # To get exact reason, we can display WR
            warning_details_list.append(f"{spec_id[:8]} (WR: {s['winrate']:.1%})")
        warning_details = ", ".join(warning_details_list)
        
        # Calculate top and worst performer today based on all approved specialists
        approved_specs = db.get_specialists_by_status("APPROVED")
        valid_performers = [s for s in approved_specs if s["total_trades"] >= 20]
        
        top_performer = max(valid_performers, key=lambda x: x["winrate"]) if valid_performers else None
        worst_performer = min(valid_performers, key=lambda x: x["winrate"]) if valid_performers else None
        
        # Regime distribution
        regimes = {}
        for t in trades_today:
            r = t["regime_at_entry"]
            regimes[r] = regimes.get(r, 0) + 1
            
        regime_dist = []
        for r, count in regimes.items():
            pct = (count / total_trades) * 100 if total_trades > 0 else 0
            regime_dist.append(f"{r} {pct:.0f}%")
        regime_str = ", ".join(regime_dist) if regime_dist else "None"
        
        logger.info(f"=== [PoolManager] Daily Specialist Report ({today_str}) ===")
        logger.info(f"- Total trades: {total_trades}")
        logger.info(f"- Approved count: {approved_count}")
        logger.info(f"- New specialist generated: {new_specialists}")
        logger.info(f"- Eliminated today: {eliminated_today}")
        logger.info(f"- WARNING status: {warning_count} - {warning_details}")
        if top_performer:
            logger.info(f"- Top performer: {top_performer['id'][:8]} (WR {top_performer['winrate']:.0%}, {top_performer['total_trades']} trades)")
        if worst_performer:
            logger.info(f"- Worst performer: {worst_performer['id'][:8]} (WR {worst_performer['winrate']:.0%}, {worst_performer['total_trades']} trades)")
        logger.info(f"- Regime distribution: {regime_str}")
        logger.info("=========================================================")

    def monitor_decay_warnings(self):
        """
        Check WARNING specialists and log their recent 5 trades WR trend.
        Call this every 5 minutes from a scheduler.
        """
        warnings = db.get_specialists_by_status("WARNING")
        for spec in warnings:
            spec_id = spec["id"]
            recent_trades = db.get_recent_trades(spec_id, limit=10)
            if len(recent_trades) >= 5:
                recent_5 = recent_trades[:5]
                wins_5 = sum(1 for t in recent_5 if t["pnl"] > 0)
                wr_5 = wins_5 / 5.0
                
                trend = "STABLE"
                if len(recent_trades) == 10:
                    prev_5 = recent_trades[5:10]
                    wins_prev = sum(1 for t in prev_5 if t["pnl"] > 0)
                    wr_prev = wins_prev / 5.0
            return
        
        # 1. Evaluasi Stage 2: PROBATION -> APPROVED / ELIMINATED
        if status == "PROBATION":
            self._evaluate_probation(specialist_id, total_trades, trades)
            
        # 2. Evaluasi Stage 3: APPROVED -> WARNING / SUSPENDED
        elif status in ["APPROVED", "WARNING"]:
            self._evaluate_approved(spec, trades, recent_wr)

    def _evaluate_probation(self, specialist_id: str, total_trades: int, trades: list):
        """Evaluasi Stage 2 Fast-Kill"""
        if total_trades < 3:
            return
            
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = wins / len(trades)
        
        # Hitung Profit Factor dari trades ini
        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 99.0
        
        # Safety Check (Override Eliminasi)
        if wr > 0.75 or pf > 1.5:
            if total_trades >= 20:
                db.update_specialist_status(specialist_id, "APPROVED")
                logger.success(f"[Fast-Kill] {specialist_id} LULUS PROBATION! (Safety net: WR {wr:.1%} / PF {pf:.2f})")
            return
        
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
            
        # Trade ke-5: WR < 35% (Stricter threshold)
        if 5 <= total_trades < 10 and wr < 0.35:
            db.update_specialist_status(specialist_id, "ELIMINATED")
            logger.warning(f"[Fast-Kill] {specialist_id} ELIMINATED: WR {wr:.1%} < 35% di trade 5.")
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

    def _evaluate_approved(self, spec: dict, recent_20_trades: list, recent_wr: float):
        """Evaluasi Stage 3 Monitoring & Performance Decay"""
        specialist_id = spec["id"]
        current_status = spec["status"]
        overall_wr = spec["winrate"]
        
        # 1. Performance Decay Detection (Track last 10 trades)
        if len(recent_20_trades) >= 10:
            recent_10 = recent_20_trades[:10]
            recent_10_wins = sum(1 for t in recent_10 if t["pnl"] > 0)
            recent_10_wr = recent_10_wins / 10.0
            
            if current_status == "APPROVED":
                if overall_wr - recent_10_wr > 0.15:
                    db.update_specialist_status(specialist_id, "WARNING", reason="WR degradation detected")
                    logger.warning(f"[PoolManager] {specialist_id} WR degradation detected: Overall {overall_wr:.1%} vs Recent 10 {recent_10_wr:.1%}")
                    current_status = "WARNING"
                elif recent_10_wr < 0.55:
                    # 3. Add APPROVED threshold ketat
                    db.update_specialist_status(specialist_id, "WARNING", reason="Approved specialist underperforming: WR < 55% in recent 10")
                    logger.warning(f"[PoolManager] {specialist_id} Approved specialist underperforming: WR {recent_10_wr:.1%} < requirement")
                    current_status = "WARNING"
            
            elif current_status == "WARNING":
                # Check 5 next trades to see if it kept dropping.
                if len(recent_20_trades) >= 5:
                    recent_5 = recent_20_trades[:5]
                    recent_5_wins = sum(1 for t in recent_5 if t["pnl"] > 0)
                    recent_5_wr = recent_5_wins / 5.0
                    
                    if recent_5_wr < recent_10_wr:
                        db.update_specialist_status(specialist_id, "SUSPENDED", reason="Continued WR decline in WARNING state")
                        logger.error(f"[PoolManager] {specialist_id} SUSPENDED: Continued decline. Recent 5 WR {recent_5_wr:.1%} < Recent 10 WR {recent_10_wr:.1%}")
                        return

        if len(recent_20_trades) < 20:
            return  # Tunggu sampai ada window 20 trade
            
        if current_status == "APPROVED":
            if recent_wr < 0.60:
                db.update_specialist_status(specialist_id, "SUSPENDED", reason=f"WR drops below 60% ({recent_wr:.1%})")
                logger.error(f"[PoolManager] {specialist_id} SUSPENDED: Recent WR drop ke {recent_wr:.1%}")
                from monitoring.alerting import telegram_alerter
                telegram_alerter.send_alert(f"⚠️ <b>SPECIALIST SUSPENDED</b>\nID: {specialist_id}\nReason: WR drops below 60% ({recent_wr:.1%})")
            elif recent_wr < 0.70:
                db.update_specialist_status(specialist_id, "WARNING", reason=f"WR drops below 70% ({recent_wr:.1%})")
                logger.warning(f"[PoolManager] {specialist_id} WARNING: Recent WR drop ke {recent_wr:.1%}")
                
        elif current_status == "WARNING":
            if recent_wr > 0.75:
                db.update_specialist_status(specialist_id, "APPROVED", reason="Recovered WR > 75%")
                logger.success(f"[PoolManager] {specialist_id} RECOVERED -> APPROVED. Recent WR {recent_wr:.1%}")
            elif recent_wr < 0.60:
                db.update_specialist_status(specialist_id, "SUSPENDED", reason=f"WR drop ke {recent_wr:.1%}")
                logger.error(f"[PoolManager] {specialist_id} SUSPENDED: WR drop ke {recent_wr:.1%}")

    def generate_daily_report(self):
        """
        Generate Enhanced Daily Report
        """
        from datetime import datetime
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        
        trades_today = db.get_trades_today()
        total_trades = len(trades_today)
        approved_count = db.count_specialists("APPROVED")
        
        events_today = db.get_events_today()
        # Cari PROBATION via log
        new_specialists = sum(1 for e in events_today if e["event_type"] == "STATUS_CHANGE" and "PROBATION" in (e.get("description") or ""))
        eliminated_today = sum(1 for e in events_today if e["event_type"] == "STATUS_CHANGE" and "ELIMINATED" in (e.get("description") or ""))
        
        warning_specs = db.get_specialists_by_status("WARNING")
        warning_count = len(warning_specs)
        
        # Fetch reason from recent events for the warnings
        warning_details_list = []
        for s in warning_specs:
            spec_id = s["id"]
            # To get exact reason, we can display WR
            warning_details_list.append(f"{spec_id[:8]} (WR: {s['winrate']:.1%})")
        warning_details = ", ".join(warning_details_list)
        
        # Calculate top and worst performer today based on all approved specialists
        approved_specs = db.get_specialists_by_status("APPROVED")
        valid_performers = [s for s in approved_specs if s["total_trades"] >= 20]
        
        top_performer = max(valid_performers, key=lambda x: x["winrate"]) if valid_performers else None
        worst_performer = min(valid_performers, key=lambda x: x["winrate"]) if valid_performers else None
        
        # Regime distribution
        regimes = {}
        for t in trades_today:
            r = t["regime_at_entry"]
            regimes[r] = regimes.get(r, 0) + 1
            
        regime_dist = []
        for r, count in regimes.items():
            pct = (count / total_trades) * 100 if total_trades > 0 else 0
            regime_dist.append(f"{r} {pct:.0f}%")
        regime_str = ", ".join(regime_dist) if regime_dist else "None"
        
        logger.info(f"=== [PoolManager] Daily Specialist Report ({today_str}) ===")
        logger.info(f"- Total trades: {total_trades}")
        logger.info(f"- Approved count: {approved_count}")
        logger.info(f"- New specialist generated: {new_specialists}")
        logger.info(f"- Eliminated today: {eliminated_today}")
        logger.info(f"- WARNING status: {warning_count} - {warning_details}")
        if top_performer:
            logger.info(f"- Top performer: {top_performer['id'][:8]} (WR {top_performer['winrate']:.0%}, {top_performer['total_trades']} trades)")
        if worst_performer:
            logger.info(f"- Worst performer: {worst_performer['id'][:8]} (WR {worst_performer['winrate']:.0%}, {worst_performer['total_trades']} trades)")
        logger.info(f"- Regime distribution: {regime_str}")
        logger.info("=========================================================")

    def monitor_decay_warnings(self):
        """
        Check WARNING specialists and log their recent 5 trades WR trend.
        Call this every 5 minutes from a scheduler.
        """
        warnings = db.get_specialists_by_status("WARNING")
        for spec in warnings:
            spec_id = spec["id"]
            recent_trades = db.get_recent_trades(spec_id, limit=10)
            if len(recent_trades) >= 5:
                recent_5 = recent_trades[:5]
                wins_5 = sum(1 for t in recent_5 if t["pnl"] > 0)
                wr_5 = wins_5 / 5.0
                
                trend = "STABLE"
                if len(recent_trades) == 10:
                    prev_5 = recent_trades[5:10]
                    wins_prev = sum(1 for t in prev_5 if t["pnl"] > 0)
                    wr_prev = wins_prev / 5.0
                    if wr_5 < wr_prev:
                        trend = "DOWN"
                    elif wr_5 > wr_prev:
                        trend = "UP"
                
                logger.info(f"[DecayMonitor] {spec_id[:8]}: Last 5 trades WR = {wr_5:.0%}, trend = {trend}")

    def retroactive_re_evaluate_all(self):
        """Re-evaluate semua specialist dengan threshold baru"""
        approved = db.get_specialists_by_status("APPROVED")
        probation = db.get_specialists_by_status("PROBATION")
        all_specs = approved + probation
        
        for spec in all_specs:
            wr = spec.get('winrate', 0.0)
            pf = spec.get('profit_factor', 0.0)
            total_trades = spec.get('total_trades', 0)
            
            # ELIMINATE criteria (ketat untuk PROBATION)
            if wr < 0.50 and total_trades >= 10:
                db.update_specialist_status(
                    spec['id'], 
                    'ELIMINATED',
                    reason=f'Retroactive: WR {wr:.1%} < 50% requirement'
                )
                logger.warning(f"✗ {spec['id']}: Downgrade {spec['status']}→ELIMINATED (WR {wr:.1%})")
            
            # SUSPEND criteria
            elif wr < 0.65 and total_trades >= 5:
                db.update_specialist_status(
                    spec['id'],
                    'SUSPENDED', 
                    reason=f'Retroactive: WR {wr:.1%} < 65% requirement'
                )
                logger.warning(f"✗ {spec['id']}: Downgrade {spec['status']}→SUSPENDED (WR {wr:.1%})")

pool_manager = SpecialistPoolManager()
