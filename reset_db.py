import sqlite3
conn = sqlite3.connect('forexmon.db')
conn.execute('DELETE FROM specialists')
conn.execute('DELETE FROM trades')
conn.execute('DELETE FROM regime_history')
conn.commit()
print('DB cleared!')
