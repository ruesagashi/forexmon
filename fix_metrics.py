import sqlite3

conn = sqlite3.connect('forexmon.db')
conn.row_factory = sqlite3.Row

specs = conn.execute('SELECT DISTINCT specialist_id FROM trades WHERE result IS NOT NULL').fetchall()

for s in specs:
    sid = s['specialist_id']
    row = conn.execute(
        'SELECT COUNT(*) as total, '
        'SUM(CASE WHEN result="WIN" THEN 1 ELSE 0 END) as wins, '
        'SUM(CASE WHEN pnl>0 THEN pnl ELSE 0 END) as gp, '
        'SUM(CASE WHEN pnl<0 THEN ABS(pnl) ELSE 0 END) as gl '
        'FROM trades WHERE specialist_id=? AND result IS NOT NULL',
        (sid,)
    ).fetchone()

    total = row['total'] or 0
    wins = row['wins'] or 0
    gp = row['gp'] or 0.0
    gl = row['gl'] or 0.01

    wr = wins / total if total > 0 else 0.0
    pf = gp / gl if gl > 0 else 0.0

    conn.execute(
        'UPDATE specialists SET winrate=?, profit_factor=?, total_trades=? WHERE id=?',
        (wr, pf, total, sid)
    )
    print(f"{sid}: WR={wr:.1%} PF={pf:.2f} Trades={total}")

conn.commit()
print("Done! Refresh dashboard.")