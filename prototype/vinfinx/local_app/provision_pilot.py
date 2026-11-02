"""Explicit operator CLI: fresh isolated state only; no HTTP signup/admin privilege grants."""
import argparse
import json
import os
from pathlib import Path
import secrets

import store
from payslip import generate

def provision(destination,samples=True):
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Pilot state must be a new directory. Existing data/accounts are never reset.')
    os.umask(0o077);store.STATE=destination;store.initialize(seed=False);accounts=[]
    with store.connect() as db:
        for role,count in [('advisor',2),('authenticator',2),('customer',6)]:
            for i in range(1,count+1):
                ident=f'{role}{i:02d}';email=ident+'@vinfinx.demo';password=secrets.token_urlsafe(16);salt=secrets.token_hex(16)
                db.execute('INSERT INTO users VALUES(?,?,?,?,?,?)',(ident,email,role,salt,store.password_hash(password,salt),ident if role=='customer' else None))
                accounts.append({'role':role,'email':email,'password':password,'id':ident})
                if role=='customer':
                    name=f'Pilot Customer {i:02d}';advisor='advisor01' if i<=3 else 'advisor02'
                    db.execute('INSERT INTO profiles VALUES(?,?,?,?,?,?,?,?,?,?)',(ident,name,30000000+i*2000000,14000000+i*500000,3000000+i*500000,80000000+i*20000000,200000000+i*50000000,0,'Synthetic pilot - explore my financial picture',advisor))
        store.audit(db,'operator','pilot_provisioned','pilot',{'roles':{'customer':6,'advisor':2,'authenticator':2},'synthetic':True})
    path=destination/'pilot_accounts.json';path.write_text(json.dumps(accounts,indent=2));path.chmod(0o600)
    if samples:
        for i in range(1,7):generate(destination/'samples'/f'customer{i:02d}.pdf',f'Pilot Customer {i:02d}',f'VFX-PILOT-{i:02d}')
    return path

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--state',required=True);args=parser.parse_args()
    credentials=provision(args.state)
    print('Created6 Customer,2 Advisor,2 Authenticator accounts in isolated synthetic pilot state.')
    print('Owner-only credentials file:',credentials)
