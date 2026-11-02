import base64
import http.cookiejar
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error

import assistant
from documents import extract
from finance import evaluate
import server
import store
from valuation import get_model,features,NUM,GEO

class LocalDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='vinfinx-test-');cls.previous=store.STATE;store.STATE=Path(cls.temp.name)
        store.initialize();server.seed_properties()
        cls.accounts={a['role']:a for a in json.loads((store.STATE/'demo_accounts.json').read_text())}
        cls.http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        server.PORT=cls.http.server_port;server.ORIGINS={f'http://127.0.0.1:{server.PORT}'}
        cls.base=f'http://127.0.0.1:{server.PORT}';cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True);cls.thread.start()
        cls.sample=server.ROOT/'output/pdf/vinfinx-sample-payslip.pdf'
        assistant.BASE='http://127.0.0.1:9'

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close();store.STATE=cls.previous;cls.temp.cleanup()

    def client(self,role=None):
        opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        if role:self.request(opener,'/api/login',{'email':self.accounts[role]['email'],'password':self.accounts[role]['password']})
        return opener

    def request(self,c,path,data=None,origin=True,expected=200):
        headers={'Origin':self.base} if origin else {}
        if data is not None:headers['Content-Type']='application/json'
        req=urllib.request.Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
        try:
            r=c.open(req,timeout=60);status=r.status;raw=r.read();kind=r.headers.get('Content-Type','')
        except urllib.error.HTTPError as e:status=e.code;raw=e.read();kind=e.headers.get('Content-Type','')
        self.assertEqual(status,expected,raw[:180])
        return json.loads(raw) if 'application/json' in kind else raw

    def test_01_auth_rbac_csrf_and_privacy(self):
        c=self.client();self.request(c,'/api/summary',expected=401)
        self.request(c,'/api/login',{'email':'customer@vinfinx.demo','password':'bad'},expected=401)
        customer=self.client('customer');self.request(customer,'/api/audit',expected=403)
        self.request(customer,'/api/summary?customer=my',expected=403)
        self.request(customer,'/api/reviews',expected=403)
        self.request(customer,'/api/profile',{},origin=False,expected=403)
        self.request(customer,'/.local-demo/demo_accounts.json',expected=404)
        self.request(customer,'/api/logout',{});self.request(customer,'/api/summary',expected=401)

    def test_02_role_capabilities(self):
        advisor=self.client('advisor');self.assertEqual(len(self.request(advisor,'/api/customers')),3)
        self.request(advisor,'/api/profile',{},expected=403)
        auth=self.client('authenticator');self.assertEqual(len(self.request(auth,'/api/reviews')['properties']),3)
        self.request(auth,'/api/valuation',{'property':{}},expected=403)

    def test_03_real_model_changes_and_research_calibration(self):
        c=self.client('customer');raw={'district':'quận đống đa','ward':'phường quang trung','street':'đường thái thịnh','area':52.3,'floors':4,'frontage':4,'road':4,'bedrooms':3,'bathrooms':3,'method':'lightgbm'}
        a=self.request(c,'/api/valuation',{'property':raw})['result']
        raw['area']=80;b=self.request(c,'/api/valuation',{'property':raw})['result']
        self.assertNotEqual(a['value'],b['value']);self.assertTrue(a['low']<a['value']<a['high'])
        self.assertTrue(a['factors']);self.assertEqual(len(a['model_sha256']),64)
        raw['method']='hierarchical';h=self.request(c,'/api/valuation',{'property':raw})['result']
        self.assertEqual(h['method'],'hierarchical');self.assertLessEqual(abs(h['calibration_log_adjustment']),math.log(1.5))
        raw['area']=-1;self.request(c,'/api/valuation',{'property':raw},expected=400)

    def test_04_pdf_extraction_and_image_ocr(self):
        profile={'name':'Nguyen Minh Long'};ex=extract(self.sample,profile)
        self.assertTrue(ex['valid_for_demo_review'],ex);self.assertEqual(ex['fields']['net'],43000000)
        self.assertTrue(ex['evidence']['net']['spans'])
        scan=extract(self.sample.with_suffix('.png'),profile)
        self.assertTrue(scan['valid_for_demo_review'],scan);self.assertEqual(scan['fields']['net'],43000000)
        self.assertIn('Vision',scan['engine'])

    def test_05_upload_review_changes_income_only_after_approval(self):
        c=self.client('customer');auth=self.client('authenticator')
        before=self.request(c,'/api/summary')['profile'];self.assertFalse(before['verified'])
        data={'name':'synthetic-payroll.pdf','content':base64.b64encode(self.sample.read_bytes()).decode()}
        d=self.request(c,'/api/upload',data);ident=d['id'];self.assertTrue(d['extraction']['valid_for_demo_review'])
        after_upload=self.request(c,'/api/summary')['profile'];self.assertEqual(after_upload['income'],before['income'])
        self.assertFalse(after_upload['verified'])
        self.assertTrue(self.request(c,'/api/upload',data)['duplicate'])
        self.request(c,'/api/review',{'kind':'document','id':ident,'action':'approve'},expected=403)
        self.request(auth,'/api/review',{'kind':'document','id':ident,'action':'approve','note':'Synthetic demo checked'})
        after=self.request(c,'/api/summary')['profile'];self.assertEqual(after['income'],43000000);self.assertTrue(after['verified'])
        self.request(auth,'/api/review',{'kind':'document','id':ident,'action':'approve'},expected=409)
        self.assertIn(b'43,000,000',self.request(c,'/api/report'))
        self.assertTrue(any(v['action']=='document_approve' for v in self.request(auth,'/api/audit')))

    def test_06_bad_document_cannot_verify(self):
        c=self.client('customer');auth=self.client('authenticator')
        self.request(c,'/api/upload',{'name':'bad.html','content':base64.b64encode(b'<html>bad</html>').decode()},expected=415)
        bad=self.request(c,'/api/upload',{'name':'broken.pdf','content':base64.b64encode(b'%PDF-1.4 invalid').decode()})
        self.request(auth,'/api/review',{'kind':'document','id':bad['id'],'action':'approve'},expected=409)
        self.request(auth,'/api/review',{'kind':'document','id':bad['id'],'action':'reject'},expected=400)
        self.request(auth,'/api/review',{'kind':'document','id':bad['id'],'action':'reject','note':'Unreadable synthetic fixture'})

    def test_07_chat_report_appointments_and_market(self):
        c=self.client('customer');r=self.request(c,'/api/chat',{'message':'Explain my score'})
        self.assertIn('Evidence mode',r['mode']);self.assertTrue(r['evidence_ids']);self.assertNotIn('Tran Ha My',r['message'])
        self.request(c,'/api/appointment',{'at':'2099-01-01T10:30','note':'Demo review'})
        self.assertTrue(self.request(c,'/api/summary')['appointments'])
        market=self.request(c,'/api/market');self.assertEqual(len(market['districts']),12);self.assertGreater(len(market['streets']),100)

    def test_08_financial_rules_and_nonapproval(self):
        p={'income':43000000,'expenses':18000000,'debt_payment':8000000,'cash':240000000,'debt':900000000,'verified':True}
        h=evaluate(p,8e9);self.assertEqual(h['monthly_surplus'],17000000);self.assertTrue(0<=h['score']<=1000)
        self.assertIn('Not a credit',h['disclaimer']);self.assertGreater(h['illustrative_borrowing_capacity'],0)

    def test_09_model_parity_with_frozen_test_predictions(self):
        import numpy as np
        import pandas as pd
        from valuation import ROOT,FLAGS
        model=get_model();d=pd.read_parquet(ROOT/'Phase 4/phase1/data/phase1_full_features.parquet')
        expected=pd.read_parquet(ROOT/'Phase 4/phase1/results/group_baseline/test_predictions.parquet',columns=['id','log_point'])
        d=d[d.addr_last_1.eq('hà nội')&d.addr_last_2.eq('quận đống đa')&d.property_subtype.eq('normal_house')]
        d=d.merge(expected,on='id').head(20);self.assertEqual(len(d),20)
        for _,r in d.iterrows():
            def value(col,lo,hi):
                v=r[col]
                return float(v) if pd.notna(v) and lo<=v<=hi else None
            raw={'district':r.addr_last_2,'ward':r.addr_last_3,'street':r.street_proxy,'area':r.area_m2,'floors':value('num_floors',1,30),'subtype':r.property_subtype}
            for key,col,hi in [('frontage','frontage_m',100),('road','road_width_m',100),('width','width_m',100),('length','length_m',500),('bedrooms','bedrooms',100),('bathrooms','bathrooms',100)]:raw[key]=value(col,0,hi)
            for flag in FLAGS:raw[flag]=bool(r[flag])
            frame=features(raw);actual=float(model.model.predict(model.transform.transform(frame[GEO+NUM]),num_threads=2)[0])
            self.assertAlmostEqual(actual,r.log_point,places=9)

if __name__=='__main__':unittest.main(verbosity=2)
