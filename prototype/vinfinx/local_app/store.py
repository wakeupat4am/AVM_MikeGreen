"""Local-only SQLite demo. Generated accounts and customer files never enter Git."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time

class Connection(sqlite3.Connection):
    """SQLite's normal context manager commits but does not close the connection."""
    def __exit__(self,*args):
        try:return super().__exit__(*args)
        finally:self.close()

HERE=Path(__file__).resolve().parent
STATE=Path(os.environ.get('VINFINX_LOCAL_STATE',str(HERE.parent/'.local-demo')))

def connect():
    db=sqlite3.connect(STATE/'vinfinx.sqlite',timeout=15,factory=Connection)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA busy_timeout=15000')
    db.execute('PRAGMA synchronous=FULL')
    return db

def password_hash(password,salt):
    return hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),240000).hex()

def audit(db,actor,action,entity,detail=None):
    db.execute('INSERT INTO audit(at,actor,action,entity,detail) VALUES(?,?,?,?,?)',
               (time.time(),actor,action,entity,json.dumps(detail or {})))

def initialize(seed=True):
    STATE.mkdir(parents=True,exist_ok=True);os.chmod(STATE,0o700)
    (STATE/'uploads').mkdir(exist_ok=True)
    with connect() as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,email TEXT UNIQUE,role TEXT,salt TEXT,hash TEXT,customer_id TEXT);
        CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,user_id TEXT,expires REAL);
        CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY,name TEXT,income REAL,expenses REAL,debt_payment REAL,cash REAL,debt REAL,verified INTEGER DEFAULT 0,goal TEXT,advisor TEXT);
        CREATE TABLE IF NOT EXISTS properties(id TEXT PRIMARY KEY,customer_id TEXT,input TEXT,result TEXT,created REAL,status TEXT DEFAULT 'pending',reviewer TEXT,note TEXT);
        CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,customer_id TEXT,name TEXT,sha256 TEXT,path TEXT,extraction TEXT,status TEXT,created REAL,reviewer TEXT,note TEXT);
        CREATE TABLE IF NOT EXISTS appointments(id TEXT PRIMARY KEY,customer_id TEXT,advisor TEXT,at TEXT,note TEXT,status TEXT);
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY,at REAL,actor TEXT,action TEXT,entity TEXT,detail TEXT);
        ''')
        db.executescript('''
        CREATE INDEX IF NOT EXISTS properties_customer_time ON properties(customer_id,created DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS documents_customer_hash ON documents(customer_id,sha256);
        CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires);
        CREATE INDEX IF NOT EXISTS documents_customer ON documents(customer_id,created DESC);
        CREATE TABLE IF NOT EXISTS idempotency(user_id TEXT,key TEXT,path TEXT,body_hash TEXT,result TEXT,status INTEGER,created REAL,PRIMARY KEY(user_id,key));
        ''')
        db.execute('DELETE FROM sessions WHERE expires<?',(time.time(),))
        db.execute('DELETE FROM idempotency WHERE created<?',(time.time()-86400,))
        if seed and not db.execute('select 1 from users limit 1').fetchone():
            accounts=[]
            for role in ['customer','advisor','authenticator']:
                email=role+'@vinfinx.demo';password='Vfx-'+secrets.token_urlsafe(9)+'!';salt=secrets.token_hex(16)
                db.execute('INSERT INTO users VALUES(?,?,?,?,?,?)',(role,email,role,salt,password_hash(password,salt),'minh' if role=='customer' else None))
                accounts.append({'role':role,'email':email,'password':password})
            account_file=STATE/'demo_accounts.json';account_file.write_text(json.dumps(accounts,indent=2));os.chmod(account_file,0o600)
            db.executemany('INSERT INTO profiles VALUES(?,?,?,?,?,?,?,?,?,?)',[
                ('minh','Nguyen Minh Long',38000000,18000000,8000000,240000000,900000000,0,'Renovate my home','advisor'),
                ('my','Tran Ha My',29000000,19000000,9500000,50000000,1200000000,0,'Build an emergency fund','advisor'),
                ('nam','Le Hoang Nam',65000000,25000000,6000000,650000000,500000000,0,'Plan long-term wealth','advisor')])
            audit(db,'system','demo_seeded','workspace',{'synthetic':True,'version':'local-demo/1'})
        # A seeded/self-reported value must never claim document verification without a reviewed document.
        db.execute("UPDATE profiles SET verified=0 WHERE verified=1 AND id NOT IN (SELECT customer_id FROM documents WHERE status='approved')")
    os.chmod(STATE/'vinfinx.sqlite',0o600)

def login(email,password):
    with connect() as db:
        row=db.execute('SELECT * FROM users WHERE email=?',(email.lower().strip(),)).fetchone()
        if not row or not secrets.compare_digest(row['hash'],password_hash(password,row['salt'])):return None
        token=secrets.token_urlsafe(32)
        db.execute('INSERT INTO sessions VALUES(?,?,?)',(hashlib.sha256(token.encode()).hexdigest(),row['id'],time.time()+8*3600))
        audit(db,row['id'],'login',row['id'])
        return token

def user_for(token):
    if not token:return None
    with connect() as db:
        row=db.execute('SELECT u.id,u.email,u.role,u.customer_id FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires>?',
                       (hashlib.sha256(token.encode()).hexdigest(),time.time())).fetchone()
        return dict(row) if row else None

def logout(token):
    with connect() as db:db.execute('DELETE FROM sessions WHERE token_hash=?',(hashlib.sha256(token.encode()).hexdigest(),))

def allowed_customer(user,cid):
    if user['role']=='customer':return user['customer_id']==cid
    if user['role']=='authenticator':return True
    with connect() as db:
        row=db.execute('SELECT advisor FROM profiles WHERE id=?',(cid,)).fetchone()
        return bool(row and row['advisor']==user['id'])
