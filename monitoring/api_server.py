"""
monitoring/api_server.py
FastAPI server yang expose data dari SQLite ke dashboard.
Jalankan: uvicorn monitoring.api_server:app --host 0.0.0.0 --port 8765 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import json
from datetime import datetime, timedelta

from core.memory import db
from config.settings import settings
from execution.mt5_connector import connector

app = FastAPI(title="Forexmon Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/api/summary")
def get_summary():
    """Ringkasan utama sistem: PnL, WinRate, balance, dll."""
    conn = db._get_conn()

    # PnL dan trade stats
    row = conn.execute("""
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(COALESCE(pnl, 0)) as total_pnl,
            SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
            SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
        FROM trades
        WHERE result IS NOT NULL
    """).fetchone()

    total = row['total_trades'] or 0
    wins = row['wins'] or 0
    total_pnl = round(row['total_pnl'] or 0, 2)
    gross_profit = row['gross_profit'] or 0
    gross_loss = row['gross_loss'] or 1
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0
    win_rate = round(wins / total * 100, 1) if total > 0 else 0

    # Drawdown hari ini
    today = datetime.utcnow().date().isoformat()
    today_row = conn.execute("""
        SELECT SUM(COALESCE(pnl, 0)) as today_pnl
        FROM trades
        WHERE result IS NOT NULL AND DATE(close_at) = ?
    """, (today,)).fetchone()
    today_pnl = round(today_row['today_pnl'] or 0, 2)

    # Win streak sekarang
    recent = conn.execute("""
        SELECT result FROM trades 
        WHERE result IS NOT NULL 
        ORDER BY close_at DESC LIMIT 20
    """).fetchall()
    streak = 0
    for r in recent:
        if r['result'] == 'WIN':
            streak += 1
        else:
            break

    # Open trades
    open_trades = conn.execute(
        "SELECT COUNT(*) as cnt FROM trades WHERE result IS NULL"
    ).fetchone()['cnt']

    # Specialist counts
    spec_counts = conn.execute("""
        SELECT status, COUNT(*) as cnt FROM specialists GROUP BY status
    """).fetchall()
    spec_map = {r['status']: r['cnt'] for r in spec_counts}

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "today_pnl": today_pnl,
        "win_streak": streak,
        "open_trades": open_trades,
        "specialists": {
            "approved": spec_map.get("APPROVED", 0),
            "probation": spec_map.get("PROBATION", 0),
            "warning": spec_map.get("WARNING", 0),
            "suspended": spec_map.get("SUSPENDED", 0),
            "eliminated": spec_map.get("ELIMINATED", 0),
        }
    }

@app.get("/api/specialists")
def get_specialists():
    """List semua specialist aktif dengan performa."""
    conn = db._get_conn()
    rows = conn.execute("""
        SELECT id, regime_type, status, winrate, profit_factor, total_trades,
               created_at
        FROM specialists
        WHERE status IN ('APPROVED', 'PROBATION', 'WARNING', 'SUSPENDED')
        ORDER BY 
            CASE status 
                WHEN 'APPROVED' THEN 1 
                WHEN 'WARNING' THEN 2
                WHEN 'PROBATION' THEN 3 
                WHEN 'SUSPENDED' THEN 4 
            END,
            winrate DESC
        LIMIT 20
    """).fetchall()
    return [dict(r) for r in rows]

@app.get("/api/recent_trades")
def get_recent_trades(limit: int = 10):
    """Trade terbaru yang sudah close."""
    conn = db._get_conn()
    rows = conn.execute("""
        SELECT t.specialist_id, t.symbol, t.direction, t.pnl, t.result,
               t.regime_at_entry, t.close_at, t.entry_price, t.lot_size
        FROM trades t
        WHERE t.result IS NOT NULL
        ORDER BY t.close_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]

@app.get("/api/live_positions")
def get_live_positions():
    """Mengambil data open positions langsung dari MT5."""
    try:
        positions = connector.get_open_positions()
        # ubah datetime object ke string isoformat biar JSON serializable
        for p in positions:
            if isinstance(p.get("time"), datetime):
                p["time"] = p["time"].isoformat()
        return positions
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/equity_curve")
def get_equity_curve(hours: int = 24):
    """Data equity curve untuk N jam terakhir."""
    conn = db._get_conn()
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    rows = conn.execute("""
        SELECT close_at, 
               SUM(COALESCE(pnl, 0)) OVER (ORDER BY close_at) as cumulative_pnl
        FROM trades
        WHERE result IS NOT NULL AND close_at >= ?
        ORDER BY close_at
    """, (since,)).fetchall()
    return [{"time": r['close_at'], "pnl": round(r['cumulative_pnl'], 2)} for r in rows]

@app.get("/api/regime")
def get_regime():
    """Regime terakhir yang terdeteksi Master Brain."""
    conn = db._get_conn()
    row = conn.execute("""
        SELECT regime, confidence, timestamp
        FROM regime_history
        ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    if not row:
        return {"regime": "UNKNOWN", "confidence": 0, "timestamp": None}
    return dict(row)

@app.get("/api/regime_history")
def get_regime_history(limit: int = 50):
    """History regime untuk visualisasi."""
    conn = db._get_conn()
    rows = conn.execute("""
        SELECT regime, confidence, timestamp
        FROM regime_history
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]

@app.get("/api/fast_kill_log")
def get_fast_kill_log():
    """Log eliminasi specialist dari system_events."""
    conn = db._get_conn()
    rows = conn.execute("""
        SELECT timestamp, event_type, description, metadata
        FROM system_events
        WHERE event_type IN ('ELIMINATED', 'APPROVED', 'WARNING', 'SUSPENDED')
        ORDER BY timestamp DESC LIMIT 20
    """).fetchall()
    return [dict(r) for r in rows]

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "db": settings.DB_PATH
    }
