"""Stop only a validated owned pilot supervisor and its child, never other services."""
import os
from pathlib import Path
import signal
import time

def stop():
    root=Path('/home/ubuntu/vinfinx-pilot');state=root/'state';pid=int((state/'supervisor.pid').read_text())
    proc=Path('/proc')/str(pid)
    if not proc.exists():return 'Supervisor already stopped'
    command=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    scripts=[Path(v) for v in command if v.endswith('/local_app/supervise.py')]
    # supervise.py may be launched relative to its already validated release working directory.
    if 'supervise.py' in command:scripts.append(Path(os.readlink(proc/'cwd'))/'supervise.py')
    if proc.stat().st_uid!=os.getuid() or not any(p.is_relative_to(root/'releases') for p in scripts):raise ValueError('Not the owned pilot supervisor')
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not proc.exists():return 'Pilot supervisor stopped gracefully'
        time.sleep(.1)
    raise RuntimeError('Supervisor still stopping; inspect it before starting another')

if __name__=='__main__':print(stop())
