"""Small-pilot resource bounds. No queues or metrics contain document/customer text."""
from contextlib import contextmanager
import threading
import time

class BusyError(Exception):pass

class Gate:
    def __init__(self,limit,timeout):
        self.slots=threading.BoundedSemaphore(limit);self.limit=limit;self.timeout=timeout
        self.lock=threading.Lock();self.active=0;self.completed=0
    @contextmanager
    def enter(self):
        if not self.slots.acquire(timeout=self.timeout):raise BusyError('Service busy. Please retry in a few seconds.')
        with self.lock:self.active+=1
        try:yield
        finally:
            with self.lock:self.active-=1;self.completed+=1
            self.slots.release()
    def snapshot(self):
        with self.lock:return {'active':self.active,'limit':self.limit,'completed':self.completed}

OCR=Gate(2,15)
CHAT=Gate(1,0)
IDEMPOTENCY=[threading.RLock() for _ in range(32)]

class RateLimiter:
    def __init__(self):self.lock=threading.Lock();self.entries={}
    def allow(self,key,limit=120,seconds=60):
        now=time.monotonic()
        with self.lock:
            if len(self.entries)>2048:self.entries={k:v for k,v in self.entries.items() if now-v[0]<seconds}
            if key not in self.entries or now-self.entries[key][0]>=seconds:self.entries[key]=(now,0)
            start,count=self.entries[key]
            if count>=limit:return False
            self.entries[key]=(start,count+1);return True

LIMITER=RateLimiter()
