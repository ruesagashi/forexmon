import sqlite3
conn = sqlite3.connect('forexmon.db')
btcusd_ids = ['3f9158ae', '506ef88d', '6a2cc718', 'bcca406d', 'f58bb5ad', 'b42908a9']
for sid in btcusd_ids:
    conn.execute('UPDATE specialists SET symbol=? WHERE id=?', ('BTCUSD', sid))
    print(f'Fixed {sid} -> BTCUSD')
conn.commit()
rows = conn.execute('SELECT id, symbol, regime_type FROM specialists').fetchall()
import sqlite3
conn.row_factory = sqlite3.Row
for r in conn.execute('SELECT id, symbol, regime_type FROM specialists').fetchall():
    print(dict(r))
print('Done!')
