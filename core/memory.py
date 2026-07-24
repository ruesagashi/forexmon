"""
core/memory.py
Database layer menggunakan SQLite.
Implementasi semua tabel sesuai PRD Section 3.7 + fungsi CRUD lengkap.

Tabel:
  - specialists       : master data semua specialist
  - trades            : history semua trade
  - performance_snapshots : snapshot performa berkala
  - regime_history    : log regime per waktu
  - system_events     : audit log semua event penting
"""

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


class MemoryDB:
    """
    Database manager untuk semua data persistensi sistem trading.
    Thread-safe dengan WAL mode SQLite.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    # ─────────────────────────────────────────────────────────────────────────
    # Connection & Schema
    # ─────────────────────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Dapatkan koneksi SQLite, buat baru jika belum ada."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row  # Akses kolom via nama
            # Aktifkan WAL mode untuk performa dan safety
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _initialize_db(self) -> None:
        """Buat semua tabel jika belum ada."""
        conn = self._get_conn()

        # ── Tabel: specialists ────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS specialists (
                id              TEXT PRIMARY KEY,
                regime_type     TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'PROBATION',
                model_path      TEXT,
                features_used   TEXT,   -- JSON list
                winrate         REAL DEFAULT 0.0,
                profit_factor   REAL DEFAULT 0.0,
                total_trades    INTEGER DEFAULT 0,
                regime_accuracy REAL DEFAULT 0.0,
                created_at      TEXT NOT NULL,
                last_trade_at   TEXT,
                metadata        TEXT    -- JSON untuk data tambahan
            )
        """)

        # ── Tabel: trades ─────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                specialist_id   TEXT NOT NULL,
                ticket          INTEGER,
                symbol          TEXT NOT NULL,
                direction       TEXT NOT NULL,  -- BUY / SELL
                entry_price     REAL,
                sl              REAL,
                tp              REAL,
                lot_size        REAL,
                exit_price      REAL,
                result          TEXT,           -- WIN / LOSS / BREAKEVEN
                pnl             REAL DEFAULT 0.0,
                regime_at_entry TEXT,
                confidence      REAL,
                open_at         TEXT,
                close_at        TEXT,
                FOREIGN KEY (specialist_id) REFERENCES specialists(id)
            )
        """)

        # ── Tabel: performance_snapshots ─────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                specialist_id   TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                winrate         REAL,
                profit_factor   REAL,
                drawdown        REAL,
                total_trades    INTEGER,
                composite_score REAL,
                FOREIGN KEY (specialist_id) REFERENCES specialists(id)
            )
        """)

        # ── Tabel: regime_history ─────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                timeframe   TEXT NOT NULL,
                regime      TEXT NOT NULL,
                confidence  REAL NOT NULL
            )
        """)

        # ── Tabel: system_events ──────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                description TEXT,
                metadata    TEXT    -- JSON
            )
        """)

        # ── Index untuk performa query ────────────────────────────────────────
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_specialist ON trades(specialist_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_close_at ON trades(close_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_regime_timestamp ON regime_history(timestamp, symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_specialist ON performance_snapshots(specialist_id)")

        conn.commit()
        logger.info(f"[DB] Database diinisialisasi: {self.db_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # SPECIALISTS CRUD
    # ─────────────────────────────────────────────────────────────────────────

    def add_specialist(
        self,
        specialist_id: str,
        regime_type: str,
        model_path: str = None,
        features_used: list = None,
        metadata: dict = None,
    ) -> bool:
        """Tambah specialist baru dengan status PROBATION."""
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO specialists
                   (id, regime_type, status, model_path, features_used, created_at, metadata)
                   VALUES (?, ?, 'PROBATION', ?, ?, ?, ?)""",
                (
                    specialist_id,
                    regime_type,
                    model_path,
                    json.dumps(features_used or []),
                    datetime.utcnow().isoformat(),
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            logger.info(f"[DB] Specialist ditambahkan: {specialist_id} ({regime_type})")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"[DB] Specialist sudah ada: {specialist_id}")
            return False

    def get_specialist(self, specialist_id: str) -> Optional[dict]:
        """Ambil data specialist berdasarkan ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM specialists WHERE id = ?", (specialist_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_specialists_by_status(self, status: str) -> list[dict]:
        """Ambil semua specialist dengan status tertentu."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM specialists WHERE status = ?", (status,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_specialists_by_regime(self, regime_type: str, status: str = "APPROVED") -> list[dict]:
        """Ambil specialist APPROVED untuk regime tertentu."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM specialists WHERE regime_type = ? AND status = ?",
            (regime_type, status),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_specialist_status(self, specialist_id: str, status: str) -> bool:
        """Update status specialist (PROBATION/APPROVED/WARNING/SUSPENDED/ELIMINATED)."""
        valid_statuses = {"PROBATION", "APPROVED", "WARNING", "SUSPENDED", "ELIMINATED"}
        if status not in valid_statuses:
            logger.error(f"[DB] Status tidak valid: {status}")
            return False
        conn = self._get_conn()
        conn.execute(
            "UPDATE specialists SET status = ? WHERE id = ?",
            (status, specialist_id),
        )
        conn.commit()
        self.log_event(
            event_type="STATUS_CHANGE",
            description=f"Specialist {specialist_id} → {status}",
            metadata={"specialist_id": specialist_id, "new_status": status},
        )
        return True

    def update_specialist_performance(
        self,
        specialist_id: str,
        winrate: float,
        profit_factor: float,
        total_trades: int,
    ) -> bool:
        """Update metrik performa specialist."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE specialists
               SET winrate = ?, profit_factor = ?, total_trades = ?, last_trade_at = ?
               WHERE id = ?""",
            (winrate, profit_factor, total_trades, datetime.utcnow().isoformat(), specialist_id),
        )
        conn.commit()
        return True



    # ─────────────────────────────────────────────────────────────────────────
    # TRADES CRUD
    # ─────────────────────────────────────────────────────────────────────────

    def open_trade(
        self,
        specialist_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        lot_size: float,
        regime_at_entry: str,
        confidence: float,
        ticket: int = None,
    ) -> int:
        """Catat trade baru (saat open). Return ID trade."""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO trades
               (specialist_id, ticket, symbol, direction, entry_price, sl, tp, lot_size,
                regime_at_entry, confidence, open_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                specialist_id, ticket, symbol, direction, entry_price,
                sl, tp, lot_size, regime_at_entry, confidence,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        result: str,
    ) -> bool:
        """Update trade saat close dengan exit price dan hasil."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE trades
               SET exit_price = ?, pnl = ?, result = ?, close_at = ?
               WHERE id = ?""",
            (exit_price, pnl, result, datetime.utcnow().isoformat(), trade_id),
        )
        conn.commit()
        return True

    def update_specialist_metrics(self, specialist_id: str) -> bool:
        """Hitung ulang dan update winrate + total_trades di tabel specialists."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
            FROM trades
            WHERE specialist_id = ? AND result IS NOT NULL
        """, (specialist_id,)).fetchone()

        total = row['total'] or 0
        wins = row['wins'] or 0
        gross_profit = row['gross_profit'] or 0.0
        gross_loss = row['gross_loss'] or 0.0

        winrate = wins / total if total > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        conn.execute("""
            UPDATE specialists 
            SET winrate = ?, profit_factor = ?, total_trades = ?
            WHERE id = ?
        """, (winrate, profit_factor, total, specialist_id))
        conn.commit()
        return True

    def get_recent_trades(self, specialist_id: str, limit: int = 20) -> list[dict]:
        """Ambil trade history terbaru dari specialist tertentu."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trades WHERE specialist_id = ? ORDER BY close_at DESC LIMIT ?",
            (specialist_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    def get_open_trades(self) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trades WHERE result IS NULL"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_specialist_trades(self, specialist_id: str, last_n: int = None) -> list[dict]:
        """Ambil trade history specialist, opsional N trade terakhir."""
        conn = self._get_conn()
        query = "SELECT * FROM trades WHERE specialist_id = ? AND result IS NOT NULL ORDER BY close_at DESC"
        params: tuple = (specialist_id,)

        if last_n:
            query += " LIMIT ?"
            params = (specialist_id, last_n)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_specialist_performance(self, specialist_id: str, last_n_trades: int = 20) -> dict:
        """
        Hitung WinRate dan ProfitFactor dari N trade terakhir.
        Sesuai PRD: untuk Fast-Kill dan monitoring.
        """
        trades = self.get_specialist_trades(specialist_id, last_n=last_n_trades)
        if not trades:
            return {"winrate": 0.0, "profit_factor": 0.0, "total": 0}

        wins = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]

        winrate = len(wins) / len(trades) if trades else 0.0

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        return {
            "winrate": winrate,
            "profit_factor": profit_factor,
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PERFORMANCE SNAPSHOTS
    # ─────────────────────────────────────────────────────────────────────────

    def save_performance_snapshot(
        self,
        specialist_id: str,
        winrate: float,
        profit_factor: float,
        drawdown: float,
        total_trades: int,
        composite_score: float,
    ) -> bool:
        """Simpan snapshot performa berkala (setiap 6 jam)."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO performance_snapshots
               (specialist_id, timestamp, winrate, profit_factor, drawdown, total_trades, composite_score)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                specialist_id, datetime.utcnow().isoformat(),
                winrate, profit_factor, drawdown, total_trades, composite_score,
            ),
        )
        conn.commit()
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # REGIME HISTORY
    # ─────────────────────────────────────────────────────────────────────────

    def log_regime(
        self,
        symbol: str,
        timeframe: str,
        regime: str,
        confidence: float,
    ) -> bool:
        """Catat deteksi regime ke history."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO regime_history (timestamp, symbol, timeframe, regime, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), symbol, timeframe, regime, confidence),
        )
        conn.commit()
        return True

    def get_latest_regime(self, symbol: str) -> Optional[dict]:
        """Ambil regime terakhir untuk symbol tertentu."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT * FROM regime_history
               WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1""",
            (symbol,),
        ).fetchone()
        return dict(row) if row else None

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM EVENTS (Audit Log)
    # ─────────────────────────────────────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        description: str = None,
        metadata: dict = None,
    ) -> bool:
        """Catat event penting ke audit log."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO system_events (timestamp, event_type, description, metadata)
               VALUES (?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                event_type,
                description,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        return True

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Ambil N event terbaru."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM system_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────────
    # BACKUP
    # ─────────────────────────────────────────────────────────────────────────

    def backup(self) -> str:
        """
        Backup database ke folder backups/.
        Format filename: forexmon_YYYYMMDD_HHMMSS.db
        """
        backup_dir = Path(settings.DB_BACKUP_DIR)
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"forexmon_{timestamp}.db"

        shutil.copy2(self.db_path, backup_path)
        logger.info(f"[DB] Backup berhasil: {backup_path}")
        self.log_event("DB_BACKUP", f"Backup ke {backup_path}")
        return str(backup_path)

    def close(self) -> None:
        """Tutup koneksi database."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("[DB] Koneksi database ditutup.")


# Singleton instance
db = MemoryDB()
