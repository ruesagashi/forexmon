import sqlite3
conn = sqlite3.connect('forexmon.db')
conn.row_factory = sqlite3.Row
rows = conn.execute('SELECT id, symbol, regime_type, status FROM specialists ORDER BY created_at DESC LIMIT 20').fetchall()
for r in rows:
    print(dict(r))
