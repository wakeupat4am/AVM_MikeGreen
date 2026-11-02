"""Consistent DB + referenced uploads backup; restore only into a new directory."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

import store

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def backup(target):
    target=Path(target).resolve()
    if target.exists():raise ValueError('Backup destination already exists')
    os.umask(0o077);target.mkdir(mode=0o700,parents=True)
    dest=sqlite3.connect(target/'vinfinx.sqlite')
    try:
        with store.connect() as db:db.backup(dest)
    finally:dest.close()
    snapshot=sqlite3.connect(target/'vinfinx.sqlite')
    try:
        for source, in snapshot.execute('SELECT path FROM documents'):
            source=Path(source).resolve()
            if not source.is_relative_to((store.STATE/'uploads').resolve()):raise ValueError('Document path outside private uploads')
            (target/'uploads').mkdir(exist_ok=True);shutil.copy2(source,target/'uploads'/source.name)
    finally:snapshot.close()
    for name in ['pilot_accounts.json','demo_accounts.json']:
        if (store.STATE/name).exists():shutil.copy2(store.STATE/name,target/name)
    if (store.STATE/'samples').exists():shutil.copytree(store.STATE/'samples',target/'samples')
    manifest={str(path.relative_to(target)):digest(path) for path in target.rglob('*') if path.is_file()}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return verify(target)

def verify(target):
    target=Path(target).resolve();manifest=json.loads((target/'manifest.json').read_text())
    for name,expected in manifest.items():
        path=(target/name).resolve()
        if not path.is_relative_to(target) or digest(path)!=expected:raise ValueError('Backup integrity check failed')
    with closing(sqlite3.connect(f'file:{target}/vinfinx.sqlite?mode=ro',uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Database integrity failed')
    return {'verified':True,'files':len(manifest)}

def restore(source,target):
    source=Path(source).resolve();target=Path(target).resolve();verify(source)
    if target.exists():raise ValueError('Restore destination must not exist')
    os.umask(0o077);shutil.copytree(source,target)
    with closing(sqlite3.connect(target/'vinfinx.sqlite')) as db:
        with db:
            for ident,path in db.execute('SELECT id,path FROM documents').fetchall():db.execute('UPDATE documents SET path=? WHERE id=?',(str(target/'uploads'/Path(path).name),ident))
            db.execute('DELETE FROM sessions')
        # Materialize committed WAL pages before hashing; sidecars are transient, not backup artifacts.
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    manifest={str(path.relative_to(target)):digest(path) for path in target.rglob('*') if path.is_file() and path.name!='manifest.json'}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'restored':True,'sessions_invalidated':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['create','verify','restore']);parser.add_argument('path');parser.add_argument('--to');args=parser.parse_args()
    if args.action=='restore' and not args.to:parser.error('--to required')
    print(json.dumps(backup(args.path) if args.action=='create' else verify(args.path) if args.action=='verify' else restore(args.path,args.to)))
