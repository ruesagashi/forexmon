import sqlite3
conn = sqlite3.connect('forexmon.db')
conn.row_factory = sqlite3.Row
rows = conn.execute('SELECT id, specialist_id, symbol, direction, entry_price, sl, tp, ticket FROM trades WHERE result IS NULL').fetchall()
for r in rows:
    print(dict(r))
