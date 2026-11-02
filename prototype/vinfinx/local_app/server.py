"""Local VinFinx app, loopback only. Stdlib HTTP + SQLite; no cloud/customer data egress."""
import base64
import csv
import hashlib
import html
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import threading
import time
import urllib.parse
import sqlite3
import logging

import assistant
from documents import extract
from finance import evaluate
import store
from valuation import get_model,readiness
import runtime

HERE=Path(__file__).resolve().parent;ROOT=Path(os.environ.get('VINFINX_WORKSPACE_ROOT',str(HERE.parents[2])))
PORT=int(os.environ.get('VINFINX_LOCAL_PORT','4173'))
WRITE=threading.RLock();ATTEMPTS={}
ORIGINS={f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'}
BRAND='VinFinx x Vietcombank'

class Error(Exception):
    def __init__(self,status,message):self.status=status;self.message=message

def summary(cid):
    with store.connect() as db:
        p=db.execute('SELECT * FROM profiles WHERE id=?',(cid,)).fetchone()
        if not p:raise Error(404,'Customer not found')
        prop=db.execute('SELECT * FROM properties WHERE customer_id=? ORDER BY created DESC LIMIT 1',(cid,)).fetchone()
        prop=dict(prop) if prop else None
        if prop:prop.update(input=json.loads(prop['input']),result=json.loads(prop['result']))
        health=evaluate(dict(p),prop['result']['value'] if prop and prop['status']!='rejected' else 0)
        return {'profile':dict(p),'property':prop,'health':health,'synthetic':True,
                'documents':[dict(d) for d in db.execute('SELECT id,name,status,created,note FROM documents WHERE customer_id=? ORDER BY created DESC',(cid,))],
                'appointments':[dict(a) for a in db.execute('SELECT * FROM appointments WHERE customer_id=? ORDER BY at',(cid,))]}

def seed_properties():
    with store.connect() as db:
        if db.execute('SELECT 1 FROM properties LIMIT 1').fetchone():return
    for cid,area,floors in [('minh',52.3,4),('my',40,3),('nam',80,5)]:
        raw={'district':'quận đống đa','ward':'phường quang trung','street':'đường thái thịnh','area':area,'floors':floors,'frontage':4,'road':4,'bedrooms':3,'bathrooms':3,'subtype':'normal_house','method':'lightgbm'}
        result=get_model().predict(raw)
        with store.connect() as db:
            db.execute('INSERT INTO properties(id,customer_id,input,result,created,status) VALUES(?,?,?,?,?,?)',
                       (secrets.token_hex(8),cid,json.dumps(raw),json.dumps(result),time.time(),'pending'))
            store.audit(db,'system','synthetic_property_scored',cid,{'model':result['model_version'],'hash':result['model_sha256']})

class Handler(BaseHTTPRequestHandler):
    server_version='VinFinxLocal/1'
    def log_message(self,fmt,*args):
        # Do not log query strings, user text, cookies or file names. Closed log pipes must not kill HTTP responses.
        try:logging.getLogger('vinfinx.http').info('%s %s',getattr(self,'command','HTTP'),getattr(self,'response_status','-'))
        except Exception:pass

    def send(self,data,status=200,kind='application/json',headers=None):
        raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode() if kind=='application/json' else data.encode() if isinstance(data,str) else data
        self.response_status=status
        self.send_response(status)
        self.send_header('Content-Type',kind+'; charset=utf-8' if kind.startswith('text/') or kind=='application/json' else kind)
        self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('X-Request-ID',getattr(self,'request_id',secrets.token_hex(12)))
        if status in (429,503):self.send_header('Retry-After','5')
        connect="'self' http://127.0.0.1:8061 http://127.0.0.1:8062" if self.path.startswith('/research/') else "'self'"
        inline=" 'unsafe-inline'" if self.path.startswith('/research/') else ''
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' https://unpkg.com"+inline+"; style-src 'self' 'unsafe-inline' https://unpkg.com; img-src 'self' data: blob: https://*.tile.openstreetmap.org; connect-src "+connect+"; frame-ancestors 'none'")
        for k,v in (headers or {}).items():self.send_header(k,v)
        self.end_headers();self.wfile.write(raw)

    def token(self):
        try:return SimpleCookie(self.headers.get('Cookie','')).get('vinfinx_session').value
        except Exception:return ''

    def require(self,roles=None):
        user=store.user_for(self.token())
        if not user:raise Error(401,'Sign in to continue')
        if roles and user['role'] not in roles:raise Error(403,'This role cannot perform that action')
        return user

    def cid(self,user,value=None):
        cid=user['customer_id'] if user['role']=='customer' else value or 'minh'
        if value and user['role']=='customer' and value!=cid:raise Error(403,'Other customer records are not accessible')
        if not store.allowed_customer(user,cid):raise Error(403,'Customer is not assigned to you')
        return cid

    def body(self):
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=9*1024*1024:raise Error(413,'Maximum upload 6 MB; JSON request too large')
        if not self.headers.get('Content-Type','').startswith('application/json'):raise Error(415,'JSON required')
        try:
            result=json.loads(self.rfile.read(length),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Non-finite JSON number')))
            if not isinstance(result,dict):raise ValueError('JSON object required')
            return result
        except Exception:raise Error(400,'Invalid JSON')

    def do_GET(self):self.dispatch(False)
    def do_POST(self):self.dispatch(True)

    def dispatch(self,post):
        self.request_id=secrets.token_hex(12)
        try:
            if self.headers.get('Host') not in {f'127.0.0.1:{PORT}',f'localhost:{PORT}'}:raise Error(403,'Loopback Host required')
            if post and self.headers.get('Origin') not in ORIGINS:raise Error(403,'Same-origin request required')
            parsed=urllib.parse.urlparse(self.path);path=parsed.path;query=urllib.parse.parse_qs(parsed.query)
            if post:return self.post(path,self.body())
            if path=='/':return self.send((HERE/'index.html').read_bytes(),kind='text/html')
            if path=='/healthz':return self.send({'status':'alive','service':BRAND})
            if path.startswith('/assets/') or path in ['/app.js','/app.css']:
                candidate=(HERE/path.lstrip('/')).resolve()
                if not candidate.is_relative_to(HERE) or not candidate.is_file() or candidate.suffix not in ['.png','.svg','.css','.js']:
                    raise Error(404,'Asset not found')
                return self.send(candidate.read_bytes(),kind=mimetypes.guess_type(candidate)[0] or 'application/octet-stream')
            if path=='/api/me':return self.send({'user':store.user_for(self.token())})
            user=self.require()
            if path in ['/api/status','/api/models','/api/ready']:
                from documents import capabilities
                model=readiness();qwen=assistant.health()
                with store.connect() as db:database=db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
                ocr=capabilities()
                status={'qwen':qwen,'avm':model,'ocr':ocr,'local':True,'database':'SQLite WAL','database_ready':database,'model':'Live LightGBM CPU','synthetic':True,
                        'service':BRAND,'runtime':{'ocr':runtime.OCR.snapshot(),'chat':runtime.CHAT.snapshot()},
                        'macro':{'ready':False,'mode':'separate research pipeline; not connected to customer decisions'},
                        'ready':model['ready'] and database,'core_inference_ready':model['ready'] and database and ocr['image_ready'],
                        'all_models_live':False,'all_models_note':'Macro pipeline and structural hierarchy are not deployed in this pilot; inspect each service separately.'}
                return self.send(status,503 if path=='/api/ready' and not status['ready'] else 200)
            if path=='/api/summary':return self.send(summary(self.cid(user,query.get('customer',[None])[0])))
            if path=='/api/customers':
                self.require(['advisor','authenticator'])
                with store.connect() as db:
                    ids=[r['id'] for r in db.execute('SELECT id FROM profiles WHERE advisor=? OR ?="authenticator"',(user['id'],user['role']))]
                return self.send([summary(cid) for cid in ids])
            if path=='/api/market':
                data=HERE/'assets/market.json'
                return self.send(json.loads(data.read_text()) if data.exists() else {'districts':[],'streets':[],'notice':'Market snapshot not generated'})
            if path=='/api/documents':
                cid=self.cid(user,query.get('customer',[None])[0])
                with store.connect() as db:
                    docs=[dict(d) for d in db.execute('SELECT * FROM documents WHERE customer_id=? ORDER BY created DESC',(cid,))]
                for d in docs:d['extraction']=json.loads(d['extraction']);d.pop('path')
                return self.send(docs)
            if path in ['/api/sample/payslip.pdf','/api/sample/payslip.png']:
                cid=self.cid(user,query.get('customer',[None])[0]);suffix='.pdf' if path.endswith('.pdf') else '.png'
                p=store.STATE/'samples'/(cid+suffix)
                if not p.exists():p=ROOT/'output/pdf'/('vinfinx-sample-payslip'+suffix)
                if not p.exists():raise Error(404,'Sample document has not been generated')
                return self.send(p.read_bytes(),kind='application/pdf' if p.suffix=='.pdf' else 'image/png',headers={'Content-Disposition':'attachment; filename="'+p.name+'"'})
            if path.startswith('/api/documents/') and path.endswith('/file'):
                ident=path.split('/')[3]
                with store.connect() as db:d=db.execute('SELECT * FROM documents WHERE id=?',(ident,)).fetchone()
                if not d:raise Error(404,'Document not found')
                self.cid(user,d['customer_id'])
                return self.send(Path(d['path']).read_bytes(),kind=mimetypes.guess_type(d['path'])[0] or 'application/octet-stream',headers={'Content-Disposition':'attachment; filename="document'+Path(d['path']).suffix+'"'})
            if path=='/api/reviews':
                self.require(['authenticator'])
                with store.connect() as db:
                    docs=[dict(d) for d in db.execute('SELECT d.*,p.name AS customer_name FROM documents d JOIN profiles p ON p.id=d.customer_id ORDER BY d.created DESC')]
                    props=[dict(p) for p in db.execute('SELECT v.*,p.name AS customer_name FROM properties v JOIN profiles p ON p.id=v.customer_id ORDER BY v.created DESC')]
                for d in docs:d['extraction']=json.loads(d['extraction']);d.pop('path')
                for p in props:p['result']=json.loads(p['result']);p['input']=json.loads(p['input'])
                return self.send({'documents':docs,'properties':props})
            if path=='/api/audit':
                self.require(['authenticator'])
                with store.connect() as db:return self.send([dict(r) for r in db.execute('SELECT * FROM audit ORDER BY id DESC LIMIT 200')])
            if path=='/api/report':
                s=summary(self.cid(user,query.get('customer',[None])[0]));p=s['profile'];h=s['health'];prop=s['property']
                content='<html><head><title>VinFinx financial health report</title><style>body{font:16px system-ui;max-width:820px;margin:48px auto;color:#143d32}td,th{padding:14px;text-align:left;border-bottom:1px solid #ddd}table{width:100%}.notice{padding:18px;background:#f0f8ed}</style></head><body>'
                content+='<h1>VinFinx · Personal financial health</h1><p class="notice">SYNTHETIC DEMO DATA. Not a bank statement, appraisal or credit approval.</p><h2>'+html.escape(p['name'])+'</h2><h3>Score '+str(h['score'])+'/1000 · '+h['status']+'</h3><table>'
                for k,v in [('Income',p['income']),('Expenses',p['expenses']),('Debt payment',p['debt_payment']),('Monthly surplus',h['monthly_surplus']),('Cash buffer',p['cash'])]:content+='<tr><th>'+k+'</th><td>VND '+format(v,',.0f')+'</td></tr>'
                content+='</table><p>Income evidence: '+('approved synthetic payslip' if p['verified'] else 'unverified self-report')+'</p>'
                if prop:content+='<h3>Research property estimate</h3><p>VND '+format(prop['result']['value'],',.0f')+'; indicative 90% band VND '+format(prop['result']['low'],',.0f')+' - '+format(prop['result']['high'],',.0f')+'</p><p>'+html.escape(prop['result']['caveat'])+'</p>'
                content+='<p>Comparison cohort: 100 synthetic peers, not bank customers.</p><p>Rule version: '+h['rules_version']+'</p></body></html>'
                with store.connect() as db:store.audit(db,user['id'],'report_exported',p['id'])
                content=content.replace('<title>VinFinx financial health report','<title>'+BRAND+' financial health report').replace('<h1>VinFinx ·','<h1>'+BRAND+' ·')
                return self.send(content,kind='text/html',headers={'Content-Disposition':'attachment; filename="vinfinx-financial-report.html"'})
            if path=='/research/' or path.startswith('/research/'):
                self.require(['authenticator']);name=path[len('/research/'):] or 'index.html'
                if name not in ['index.html','evidence.js','pipeline.js']:raise Error(404,'Not found')
                return self.send((HERE.parent/name).read_bytes(),kind='text/html' if name.endswith('.html') else 'text/javascript')
            raise Error(404,'Not found')
        except Error as e:self.send({'error':e.message,'request_id':self.request_id},e.status)
        except (runtime.BusyError,TimeoutError,sqlite3.OperationalError):self.send({'error':'Service busy or storage unavailable. Retry shortly.','request_id':self.request_id},503)
        except (ValueError,KeyError,TypeError) as e:self.send({'error':str(e)[:180]},400)
        except Exception as e:
            logging.getLogger('vinfinx.http').error('request=%s error=%s',self.request_id,type(e).__name__);self.send({'error':'Operation failed. No fabricated result returned.','request_id':self.request_id},500)

    def post(self,path,data):
        self._idem=None
        if path in ['/api/login','/api/logout','/api/chat','/api/inference']:return self._post(path,data)
        user=self.require();key=self.headers.get('Idempotency-Key','')
        if not key:return self._post(path,data)
        if not re.fullmatch(r'[A-Za-z0-9_-]{16,80}',key):raise Error(400,'Invalid idempotency key')
        digest=hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        self._idem=(user['id'],key,path,digest)
        with runtime.IDEMPOTENCY[int(hashlib.sha256((user['id']+key).encode()).hexdigest(),16)%32]:
            with store.connect() as db:old=db.execute('SELECT * FROM idempotency WHERE user_id=? AND key=?',(user['id'],key)).fetchone()
            if old:
                if old['path']!=path or old['body_hash']!=digest:raise Error(409,'Idempotency key was used for a different request')
                return self.send(json.loads(old['result']),old['status'],headers={'Idempotency-Replayed':'true'})
            return self._post(path,data)

    def cached(self,db,result):
        if self._idem:db.execute('INSERT INTO idempotency VALUES(?,?,?,?,?,?,?)',(*self._idem,json.dumps(result,allow_nan=False),200,time.time()))
        return result

    def _post(self,path,data):
        if path=='/api/login':
            email=str(data.get('email','')).lower().strip()[:120];key=(self.client_address[0],email)
            if not runtime.LIMITER.allow(('login',*key),8):raise Error(429,'Too many attempts. Wait one minute.')
            token=store.login(email,str(data.get('password','')))
            if not token:raise Error(401,'Email or password is incorrect')
            return self.send({'user':store.user_for(token)},headers={'Set-Cookie':f'vinfinx_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800'})
        user=self.require()
        if path=='/api/logout':
            store.logout(self.token());return self.send({'ok':True},headers={'Set-Cookie':'vinfinx_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0'})
        if not runtime.LIMITER.allow((user['id'],'write'),120):raise Error(429,'Too many operations. Wait one minute.')
        cid=self.cid(user,data.get('customer_id')) if path not in ['/api/review'] else None
        if path=='/api/profile':
            self.require(['customer']);values=[]
            for field in ['income','expenses','debt_payment','cash','debt']:
                v=float(data[field])
                if not 0<=v<=1e12:raise Error(400,'Financial values must be between 0 and 1 trillion VND')
                values.append(v)
            with WRITE,store.connect() as db:
                old=db.execute('SELECT income,verified FROM profiles WHERE id=?',(cid,)).fetchone()
                verified=old['verified'] if values[0]==old['income'] else 0
                db.execute('UPDATE profiles SET income=?,expenses=?,debt_payment=?,cash=?,debt=?,verified=? WHERE id=?',values+[verified,cid]);store.audit(db,user['id'],'self_report_updated',cid,{'verified':bool(verified)})
                response=self.cached(db,{'ok':True,'customer_id':cid})
            return self.send(response)
        if path in ['/api/valuation','/api/inference']:
            self.require(['customer','advisor']);raw=data.get('property',{})
            result=get_model().predict(raw);ident=secrets.token_hex(8)
            if path=='/api/inference':return self.send({'result':result,'persisted':False,'customer_id':cid})
            response={'id':ident,'result':result,'status':'pending'}
            with WRITE,store.connect() as db:
                db.execute('INSERT INTO properties(id,customer_id,input,result,created,status) VALUES(?,?,?,?,?,?)',(ident,cid,json.dumps(raw),json.dumps(result),time.time(),'pending'))
                store.audit(db,user['id'],'property_valued',ident,{'customer_id':cid,'model_sha256':result['model_sha256'],'method':result['method']})
                self.cached(db,response)
            return self.send(response)
        if path=='/api/upload':
            self.require(['customer']);name=Path(str(data.get('name','document'))).name[:100]
            try:raw=base64.b64decode(data['content'],validate=True)
            except Exception:raise Error(400,'Invalid file data')
            if not 0<len(raw)<=6*1024*1024:raise Error(413,'File must be at most 6 MB')
            suffix='.pdf' if raw.startswith(b'%PDF-') else '.png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else '.jpg' if raw.startswith(b'\xff\xd8\xff') else None
            if suffix is None:raise Error(415,'Only PDF, PNG and JPEG files are accepted')
            sha=hashlib.sha256(raw).hexdigest()
            with store.connect() as db:
                existing=db.execute('SELECT id FROM documents WHERE customer_id=? AND sha256=?',(cid,sha)).fetchone()
            if existing:
                with WRITE,store.connect() as db:response=self.cached(db,{'id':existing['id'],'duplicate':True})
                return self.send(response)
            ident=secrets.token_hex(12);dest=store.STATE/'uploads'/(ident+suffix);dest.write_bytes(raw);os.chmod(dest,0o600)
            try:
                with runtime.OCR.enter():result=extract(dest,summary(cid)['profile'])
            except runtime.BusyError:
                dest.unlink(missing_ok=True);raise
            except Exception:
                result={'engine':'unreadable','fields':{},'valid_for_demo_review':False,'checks':{},'warnings':['Cannot read this document safely. Upload a supported synthetic sample.'],'evidence':{},'page_spans':[]}
            with WRITE,store.connect() as db:
                # Another thread may have completed this same document while OCR ran.
                existing=db.execute('SELECT id FROM documents WHERE customer_id=? AND sha256=?',(cid,sha)).fetchone()
                if existing:
                    dest.unlink(missing_ok=True);return self.send(self.cached(db,{'id':existing['id'],'duplicate':True}))
                db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)',(ident,cid,name,sha,str(dest),json.dumps(result),'pending',time.time(),None,None))
                store.audit(db,user['id'],'document_extracted',ident,{'customer_id':cid,'sha256':sha,'engine':result['engine'],'valid_for_review':result['valid_for_demo_review']})
                response=self.cached(db,{'id':ident,'extraction':result,'status':'pending'})
            return self.send(response)
        if path=='/api/review':
            self.require(['authenticator']);kind=data.get('kind');action=data.get('action');note=str(data.get('note',''))[:500].strip()
            if kind not in ['document','property'] or action not in ['approve','reject']:raise Error(400,'Invalid review')
            if action=='reject' and not note:raise Error(400,'Rejection needs a note')
            table='documents' if kind=='document' else 'properties'
            with WRITE,store.connect() as db:
                r=db.execute('SELECT * FROM '+table+' WHERE id=?',(data.get('id'),)).fetchone()
                if not r:raise Error(404,'Review item not found')
                if r['status']!='pending':raise Error(409,'This item has already been reviewed')
                if action=='approve' and kind=='document':
                    ex=json.loads(r['extraction'])
                    if not ex.get('valid_for_demo_review'):raise Error(409,'Extraction failed validation. It cannot verify income; reject and request a corrected sample.')
                    db.execute('UPDATE profiles SET income=?,verified=1 WHERE id=?',(ex['fields']['net'],r['customer_id']))
                db.execute('UPDATE '+table+' SET status=?,reviewer=?,note=? WHERE id=?',('approved' if action=='approve' else 'rejected',user['id'],note,data['id']))
                store.audit(db,user['id'],kind+'_'+action,r['customer_id'],{'id':data['id'],'note':note,'income_changed':kind=='document' and action=='approve'})
                response=self.cached(db,{'ok':True})
            return self.send(response)
        if path=='/api/appointment':
            self.require(['customer','advisor']);at=str(data.get('at',''));note=str(data.get('note',''))[:500]
            import datetime
            try:chosen=datetime.datetime.fromisoformat(at)
            except ValueError:raise Error(400,'Choose a valid appointment time')
            if chosen<datetime.datetime.now():raise Error(400,'Choose a future appointment')
            with WRITE,store.connect() as db:
                with store.connect() as lookup:assigned=lookup.execute('SELECT advisor FROM profiles WHERE id=?',(cid,)).fetchone()['advisor']
                ident=secrets.token_hex(8);db.execute('INSERT INTO appointments VALUES(?,?,?,?,?,?)',(ident,cid,assigned,at,note,'requested'));store.audit(db,user['id'],'appointment_requested',cid,{'at':at})
                response=self.cached(db,{'id':ident})
            return self.send(response)
        if path=='/api/chat':
            self.require(['customer','advisor'])
            with runtime.CHAT.enter():result=assistant.answer(str(data.get('message',''))[:2000],summary(cid))
            with store.connect() as db:store.audit(db,user['id'],'assistant_reply',cid,{'mode':result['mode'],'seconds':result['seconds'],'usage':result.get('usage',{})})
            return self.send(result)
        raise Error(404,'Not found')

def main():
    store.initialize()
    try:seed_properties()
    except Exception as e:print('AVM not ready:',type(e).__name__,'- valuation will fail explicitly until dependencies/artifacts are restored.',flush=True)
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    (store.STATE/'server.pid').write_text(str(os.getpid()))
    print(f'VinFinx local: http://127.0.0.1:{PORT}\nDemo accounts: {store.STATE}/demo_accounts.json',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
