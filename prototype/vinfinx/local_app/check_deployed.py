"""Read/preview-only acceptance against the real loopback service, never prints secrets."""
from concurrent.futures import ThreadPoolExecutor
import http.cookiejar
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

import store
from documents import extract

def check():
    base='http://127.0.0.1:'+os.environ.get('VINFINX_LOCAL_PORT','4175')
    accounts=json.loads((store.STATE/'pilot_accounts.json').read_text());metrics=[]
    def journey(account):
        client=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        def call(path,data=None):
            request=urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers={'Origin':base,'Content-Type':'application/json'})
            started=time.perf_counter()
            with client.open(request,timeout=60) as response:
                body=response.read();kind=response.headers.get('Content-Type','')
            metrics.append((path,(time.perf_counter()-started)*1000))
            return json.loads(body) if 'application/json' in kind else body
        call('/api/login',{'email':account['email'],'password':account['password']})
        status=call('/api/models');assert status['core_inference_ready']
        if account['role']=='customer':
            before=call('/api/summary')
            for method in ['lightgbm','hierarchical']:
                result=call('/api/inference',{'property':{'district':'quận đống đa','area':52.3,'floors':4,'method':method}})
                assert result['result']['inference']['live'] and not result['persisted']
            after=call('/api/summary');assert before==after
            for suffix in ['pdf','png']:assert len(call('/api/sample/payslip.'+suffix))>1000
        elif account['role']=='advisor':assert len(call('/api/customers'))==3
        else:
            call('/api/reviews');call('/api/audit');assert b'Evidence' in call('/research/')
        call('/api/logout',{})
        return status
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as pool:statuses=list(pool.map(journey,accounts))
    ocr=[]
    for i in range(1,7):
        for suffix in ['pdf','png']:
            result=extract(store.STATE/'samples'/f'customer{i:02d}.{suffix}',{'name':f'Pilot Customer {i:02d}'})
            assert result['valid_for_demo_review'] and result['fields']['net']==43000000
            ocr.append({'customer':i,'format':suffix,'engine':result['engine'],'valid':True})
    latencies=sorted(t for p,t in metrics if p=='/api/inference')
    result={'passed':True,'accounts':len(accounts),'requests':len(metrics),'inferences':len(latencies),'inference_p95_ms':round(latencies[int(.95*(len(latencies)-1))],2),
            'elapsed_s_including_12_extractions':round(time.perf_counter()-started,2),'ocr':ocr,'services':statuses[0],
            'writes':'Only sign-in/sign-out session/audit; property previews do not persist or alter profiles'}
    output=Path(__file__).parent/'pilot-results';output.mkdir(exist_ok=True);(output/'deployed.json').write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k not in ['services','ocr']}

if __name__=='__main__':print(json.dumps(check()))
