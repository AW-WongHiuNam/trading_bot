import sqlite3
p='vector_store.sqlite'
try:
    conn=sqlite3.connect(p)
    cur=conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_calls'")
    print('table exists:', bool(cur.fetchone()))
    cur.execute('PRAGMA table_info(api_calls)')
    print('schema:')
    for r in cur.fetchall():
        print(r)
    cur.execute('SELECT COUNT(*) FROM api_calls')
    print('rows:', cur.fetchone()[0])
    cur.execute('SELECT MAX(idx) FROM api_calls')
    print('max idx:', cur.fetchone()[0])
    conn.close()
except Exception as e:
    print('error:', e)
