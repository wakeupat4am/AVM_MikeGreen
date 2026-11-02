"""Bounded WSGI serving for the internal pilot. Bind loopback, never trust proxy headers."""
import io
import logging
import os
import signal
from email.message import Message
from http import HTTPStatus

import server
import store
from valuation import get_model

class Request(server.Handler):
    def __init__(self,environ):
        self.command=environ['REQUEST_METHOD'];self.path=environ.get('PATH_INFO','/')
        if environ.get('QUERY_STRING'):self.path+='?'+environ['QUERY_STRING']
        self.headers=Message()
        for name,value in environ.items():
            if name.startswith('HTTP_'):self.headers[name[5:].replace('_','-')]=value
        for name in ['CONTENT_TYPE','CONTENT_LENGTH']:
            if environ.get(name):self.headers[name.replace('_','-')]=environ[name]
        self.client_address=(environ.get('REMOTE_ADDR','127.0.0.1'),0)
        self.rfile=environ['wsgi.input'];self.wfile=io.BytesIO();self.status=500;self.response_headers=[]
    def send_response(self,code,message=None):self.status=code
    def send_header(self,key,value):self.response_headers.append((str(key),str(value)))
    def end_headers(self):pass

def application(environ,start_response):
    request=Request(environ)
    if request.command not in ['GET','POST']:
        request.send({'error':'Method not allowed'},405,headers={'Allow':'GET, POST'})
    else:request.dispatch(request.command=='POST')
    start_response(f'{request.status} {HTTPStatus(request.status).phrase}',request.response_headers)
    request.log_message('')
    return [request.wfile.getvalue()]

def prepare():
    os.umask(0o077)
    store.initialize()
    # Fail startup if actual trusted inference cannot run. This warm-up writes no customer record.
    model=get_model()
    result=model.predict({'district':'quận đống đa','ward':'phường quang trung','street':'đường thái thịnh','area':52.3,'floors':4,'frontage':4,'road':4,'bedrooms':3,'bathrooms':3})
    if not result['low']<result['value']<result['high']:raise RuntimeError('AVM warm-up failed')
    if os.environ.get('VINFINX_REQUIRE_OCR')=='1':
        from documents import extract
        sample=store.STATE/'samples/customer01.png'
        reading=extract(sample,{'name':'Pilot Customer 01'})
        if not reading['valid_for_demo_review'] or reading['fields'].get('net')!=43000000:raise RuntimeError('Required image OCR smoke test failed')
    if os.environ.get('VINFINX_REQUIRE_QWEN')=='1' and not server.assistant.health()['online']:raise RuntimeError('Required Qwen endpoint unavailable')
    logging.getLogger('vinfinx').info('AVM warmed version=%s artifact=%s',model.version,model.hash)

def main():
    from waitress import create_server
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    prepare()
    http=create_server(application,host='127.0.0.1',port=server.PORT,threads=16,connection_limit=64,backlog=64,
                       channel_timeout=45,cleanup_interval=5,max_request_body_size=9*1024*1024,
                       max_request_header_size=32768,inbuf_overflow=10*1024*1024,outbuf_overflow=10*1024*1024,
                       clear_untrusted_proxy_headers=True,ident='VinFinxPilot')
    (store.STATE/'server.pid').write_text(str(os.getpid()))
    def stop(*_):
        http.close();raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    logging.info('%s pilot listening on 127.0.0.1:%s',server.BRAND,server.PORT)
    http.run()

if __name__=='__main__':main()
