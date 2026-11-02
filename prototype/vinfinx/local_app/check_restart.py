"""Operator acceptance: gracefully stop only the validated owned child, check restart and data."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import urllib.request

import store

def check():
    root=Path('/home/ubuntu/vinfinx-pilot');release=(root/'current').resolve()
    if store.STATE.resolve()!=root/'state' or not release.is_relative_to(root/'releases'):raise ValueError('Unexpected deployment')
    pid=int((store.STATE/'server.pid').read_text());parent=int((store.STATE/'supervisor.pid').read_text())
    command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    if str(release/'local_app/serve.py').encode() not in command:raise ValueError('Not the expected pilot child')
    status=(Path('/proc')/str(pid)/'status').read_text()
    if f'PPid:\t{parent}\n' not in status:raise ValueError('Unexpected supervisor')
    def snapshot():
        with store.connect() as db:
            rows={table:[list(r) for r in db.execute('SELECT * FROM '+table+' ORDER BY id')] for table in ['users','profiles','properties','documents','appointments']}
        rows['credential_file_sha256']=hashlib.sha256((store.STATE/'pilot_accounts.json').read_bytes()).hexdigest()
        return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    before=snapshot();started=time.perf_counter();os.kill(pid,signal.SIGTERM)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));new=pid
    for _ in range(60):
        time.sleep(.25)
        try:
            new=int((store.STATE/'server.pid').read_text())
            if new==pid:continue
            with opener.open('http://127.0.0.1:4175/healthz',timeout=1) as response:
                if response.status==200:break
        except (OSError,ValueError):continue
    else:raise RuntimeError('Pilot did not restart within15 seconds')
    assert snapshot()==before,'Persistent application state changed during restart'
    result={'passed':True,'new_child':new!=pid,'recovery_s':round(time.perf_counter()-started,2),'persistent_records_and_credentials_unchanged':True}
    output=Path(__file__).parent/'pilot-results';output.mkdir(exist_ok=True);(output/'restart.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':print(json.dumps(check()))
