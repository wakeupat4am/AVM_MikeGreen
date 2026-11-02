"""Actual Waitress + SQLite + CPU model,10 isolated accounts. Never uses the owner's DB."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.cookiejar
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid

from waitress import create_server
import assistant
import backup
from provision_pilot import provision
import runtime
import serve
import server
import store
from valuation import get_model

RAW={'district':'quận đống đa','ward':'phường quang trung','street':'đường thái thịnh','area':52.3,'floors':4,'frontage':4,'road':4,'bedrooms':3,'bathrooms':3,'method':'lightgbm'}

class Pilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='vinfinx-pilot-check-');cls.previous=store.STATE;cls.old_port=server.PORT;cls.old_origins=server.ORIGINS;cls.old_base=assistant.BASE
        path=provision(Path(cls.temp.name)/'state',samples=False)
        cls.accounts={a['id']:a for a in json.loads(path.read_text())};cls.metrics=[];cls.metric_lock=threading.Lock();runtime.LIMITER=runtime.RateLimiter()
        assistant.BASE='http://127.0.0.1:9'
        cls.http=create_server(serve.application,host='127.0.0.1',port=0,threads=16,connection_limit=64,max_request_body_size=9*1024*1024)
        server.PORT=cls.http.effective_port;server.ORIGINS={f'http://127.0.0.1:{server.PORT}'};cls.base=next(iter(server.ORIGINS))
        cls.closing=False;cls.background_errors=[]
        def run():
            try:cls.http.run()
            except OSError as exc:
                if not cls.closing:cls.background_errors.append(type(exc).__name__)
        cls.thread=threading.Thread(target=run,daemon=True);cls.thread.start();get_model()

    @classmethod
    def tearDownClass(cls):
        cls.closing=True;cls.http.close();cls.thread.join(timeout=3);store.STATE=cls.previous;server.PORT=cls.old_port;server.ORIGINS=cls.old_origins;assistant.BASE=cls.old_base;cls.temp.cleanup()
        if cls.background_errors:raise AssertionError(cls.background_errors)

    def request(self,client,path,data=None,expected=200,key=None,headers=None):
        h={'Origin':self.base,**(headers or {})}
        if data is not None:h['Content-Type']='application/json'
        if key:h['Idempotency-Key']=key
        req=urllib.request.Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=h)
        start=time.perf_counter()
        try:
            with client.open(req,timeout=70) as r:code=r.status;body=r.read();content_type=r.headers.get('Content-Type','')
        except urllib.error.HTTPError as e:code=e.code;body=e.read();content_type=e.headers.get('Content-Type','')
        elapsed=(time.perf_counter()-start)*1000
        with self.metric_lock:self.metrics.append({'path':path.split('?')[0],'code':code,'ms':round(elapsed,2)})
        self.assertEqual(code,expected,body[:180]);return json.loads(body) if 'application/json' in content_type else body

    def client(self,ident=None):
        c=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        if ident:
            a=self.accounts[ident];self.request(c,'/api/login',{'email':a['email'],'password':a['password']})
        return c

    def test_01_readiness_origin_and_shapes(self):
        anon=self.client();self.assertEqual(self.request(anon,'/healthz')['status'],'alive');self.request(anon,'/api/models',expected=401)
        c=self.client('customer01');models=self.request(c,'/api/models');self.assertTrue(models['avm']['ready']);self.assertFalse(models['qwen']['online']);self.assertFalse(models['all_models_live'])
        self.request(c,'/api/inference',{'property':RAW},headers={'Origin':'https://evil.invalid'},expected=403)
        self.request(c,'/api/inference',{'property':RAW},headers={'Host':'evil.invalid'},expected=403)
        self.request(c,'/api/inference',[],expected=400)
        self.request(c,'/api/inference',{'property':{**RAW,'area':float('nan')}},expected=400)
        self.request(c,'/api/inference',{'property':{**RAW,'street':['bad']}},expected=400)
        self.request(c,'/api/inference',{'customer_id':'customer02','property':RAW},expected=403)
        self.request(c,'/api/sample/payslip.pdf?customer=customer02',expected=403)
        self.request(c,'/api/documents?customer=customer02',expected=403)

    def test_02_live_parity_and_atomic_idempotency(self):
        c=self.client('customer01');key=uuid.uuid4().hex
        before=self.request(c,'/api/summary')['property']
        r=self.request(c,'/api/inference',{'property':RAW});self.assertFalse(r['persisted']);self.assertEqual(before,self.request(c,'/api/summary')['property'])
        local=get_model().predict(RAW);self.assertAlmostEqual(local['value'],r['result']['value'],places=4)
        self.assertTrue(r['result']['inference']['live']);self.assertEqual(len(r['result']['inference']['artifact_sha256']),4)
        changed=self.request(c,'/api/inference',{'property':{**RAW,'area':80}})['result'];self.assertNotEqual(changed['value'],local['value'])
        a=self.request(c,'/api/valuation',{'property':RAW},key=key);b=self.request(c,'/api/valuation',{'property':RAW},key=key);self.assertEqual(a['id'],b['id'])
        self.request(c,'/api/valuation',{'property':{**RAW,'area':80}},key=key,expected=409)
        with store.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM properties WHERE customer_id="customer01"').fetchone()[0],1)

    def test_03_ten_simultaneous_users_real_inference(self):
        clients={ident:self.client(ident) for ident in self.accounts};barrier=threading.Barrier(10)
        start_index=len(self.metrics)
        def journey(ident):
            c=clients[ident];barrier.wait(timeout=15)
            for round_no in range(8):
                if ident.startswith('customer'):
                    s=self.request(c,'/api/summary');self.assertEqual(s['profile']['id'],ident)
                    raw={**RAW,'area':40+int(ident[-2:])*5+round_no,'method':'hierarchical' if round_no%2 else 'lightgbm'}
                    r=self.request(c,'/api/inference',{'property':raw})['result'];self.assertTrue(r['inference']['live'])
                    expected_hash=hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest();self.assertEqual(r['inference']['input_sha256'],expected_hash)
                    if round_no==0:self.request(c,'/api/profile',{'income':30000000+int(ident[-2:])*2000000,'expenses':14000000,'debt_payment':3000000,'cash':80000000,'debt':200000000},key=uuid.uuid4().hex)
                elif ident.startswith('advisor'):
                    assigned=self.request(c,'/api/customers');self.assertEqual(len(assigned),3)
                    self.assertTrue(all(s['profile']['advisor']==ident for s in assigned))
                    cid=assigned[round_no%3]['profile']['id'];self.request(c,'/api/inference',{'customer_id':cid,'property':{**RAW,'area':60+round_no}})
                    self.request(c,'/api/summary?customer='+('customer06' if ident=='advisor01' else 'customer01'),expected=403)
                else:
                    self.request(c,'/api/reviews');self.request(c,'/api/audit')
        started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as pool:list(pool.map(journey,self.accounts))
        records=self.metrics[start_index:];inferences=[r['ms'] for r in records if r['path']=='/api/inference'];latencies=sorted(r['ms'] for r in records if r['code']==200)
        report={'concurrent_users':10,'roles':{'customer':6,'advisor':2,'authenticator':2},'rounds_per_user':8,'requests':len(records),'unexpected_failures':0,'expected_denials':sum(r['code']==403 for r in records),
                'elapsed_s':round(time.perf_counter()-started,2),'http_p50_ms':round(statistics.median(latencies),2),'http_p95_ms':latencies[int(.95*(len(latencies)-1))],
                'inference_requests':len(inferences),'inference_http_p95_ms':sorted(inferences)[int(.95*(len(inferences)-1))],
                'backend':'Actual Waitress16 threads + SQLite WAL + CPU LightGBM/SHAP','qwen':'Unavailable; not included as successful LLM generation','database':'Temporary isolated synthetic state','artifact_sha256':get_model().artifact_hashes}
        out=Path(__file__).parent/'pilot-results';out.mkdir(exist_ok=True);(out/'loadtest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

    def test_04_duplicate_upload_and_review_race(self):
        c=self.client('customer01');payload={'name':'synthetic.pdf','content':base64.b64encode((server.ROOT/'output/pdf/vinfinx-sample-payslip.pdf').read_bytes()).decode()}
        with ThreadPoolExecutor(max_workers=2) as pool:responses=list(pool.map(lambda _:self.request(c,'/api/upload',payload),range(2)))
        self.assertEqual(responses[0]['id'],responses[1]['id'])
        with store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM documents WHERE customer_id="customer01"').fetchone()[0],1)
            prop=db.execute('SELECT id FROM properties WHERE customer_id="customer01"').fetchone()[0]
        a=self.client('authenticator01');b=self.client('authenticator02')
        def review(client):
            req=urllib.request.Request(self.base+'/api/review',data=json.dumps({'kind':'property','id':prop,'action':'approve','note':'Synthetic concurrency check'}).encode(),headers={'Origin':self.base,'Content-Type':'application/json'})
            try:
                with client.open(req) as r:return r.status
            except urllib.error.HTTPError as e:return e.code
        with ThreadPoolExecutor(max_workers=2) as pool:codes=list(pool.map(review,[a,b]))
        self.assertEqual(sorted(codes),[200,409])
        # The old sample belongs to another persona and cannot verify this customer's income.
        self.request(a,'/api/review',{'kind':'document','id':responses[0]['id'],'action':'approve'},expected=409)
        from payslip import generate
        personal=Path(self.temp.name)/'customer02.pdf';generate(personal,'Pilot Customer 02','SYNTHETIC-HTTP-CHECK')
        c2=self.client('customer02');before=self.request(c2,'/api/summary')['profile']
        uploaded=self.request(c2,'/api/upload',{'name':'personal-synthetic.pdf','content':base64.b64encode(personal.read_bytes()).decode()})
        self.assertTrue(uploaded['extraction']['valid_for_demo_review'])
        self.assertEqual(self.request(c2,'/api/summary')['profile']['income'],before['income'])
        self.request(a,'/api/review',{'kind':'document','id':uploaded['id'],'action':'approve','note':'Synthetic matching fixture checked'})
        after=self.request(c2,'/api/summary')['profile'];self.assertTrue(after['verified']);self.assertEqual(after['income'],43000000)

    def test_05_backup_restore_and_no_provision_reset(self):
        dest=Path(self.temp.name)/'backup';self.assertTrue(backup.backup(dest)['verified']);self.assertTrue(backup.verify(dest)['verified'])
        restored=Path(self.temp.name)/'restored';self.assertTrue(backup.restore(dest,restored)['sessions_invalidated']);self.assertTrue(backup.verify(restored)['verified'])
        subprocess.run([sys.executable,str(Path(backup.__file__)),'verify',str(restored)],check=True,capture_output=True,timeout=15)
        with self.assertRaises(ValueError):provision(store.STATE,samples=False)
        import sqlite3
        with sqlite3.connect(restored/'vinfinx.sqlite') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM users').fetchone()[0],10);self.assertEqual(db.execute('SELECT count(*) FROM sessions').fetchone()[0],0)
            for path, in db.execute('SELECT path FROM documents'):self.assertTrue(Path(path).is_file())

if __name__=='__main__':unittest.main(verbosity=2)
