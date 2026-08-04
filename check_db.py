import sqlite3
conn = sqlite3.connect('forexmon.db')
conn.row_factory = sqlite3.Row
rows = conn.execute('SELECT id, symbol, regime_type, status, model_path FROM specialists').fetchall()
for r in rows:
    print(dict(r))
