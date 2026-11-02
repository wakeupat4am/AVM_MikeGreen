"""Unprivileged pilot supervisor. Stops only its own child; private bounded logs."""
import fcntl
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading

def main():
    state=Path(os.environ['VINFINX_LOCAL_STATE']);state.mkdir(parents=True,exist_ok=True);os.umask(0o077)
    lock=(state/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=False;child=None
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (state/'supervisor.pid').write_text(str(os.getpid()))
    failures=[]
    logger=logging.getLogger('vinfinx.child');logger.setLevel(logging.INFO);logger.propagate=False
    handler=RotatingFileHandler(state/'service.log',maxBytes=5*1024*1024,backupCount=1,encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'));logger.addHandler(handler)
    while not stopping:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('serve.py'))],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,bufsize=0)
        def capture(process):
            # Bounded chunks also handle a native error stream without newlines.
            with process.stdout:
                while chunk:=process.stdout.read(16384):logger.info(chunk.decode('utf-8',errors='replace').rstrip('\n'))
        reader=threading.Thread(target=capture,args=(child,),daemon=True);reader.start()
        print('pilot child started',child.pid,flush=True)
        while child.poll() is None:
            if stopping:
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(1)
        reader.join(timeout=3)
        if stopping:break
        now=time.monotonic();failures=[t for t in failures if now-t<300]+[now]
        if len(failures)>=5:raise SystemExit('Five failures within5 minutes; check private service.log. No restart storm.')
        time.sleep(3)
    handler.close();logger.removeHandler(handler)

if __name__=='__main__':main()
