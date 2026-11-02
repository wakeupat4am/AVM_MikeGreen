"""Actual frozen LightGBM inference, optional research hierarchical residual calibration."""
import hashlib
import json
import math
import os
import time
import secrets
from pathlib import Path
import threading
import unicodedata

ROOT=Path(__file__).resolve().parents[3]
ART=Path(os.environ.get('VINFINX_MODEL_DIR',str(ROOT/'Phase 4/phase1/results/group_baseline')))
FLAGS=['is_frontage_text','is_alley_text','is_business_text','is_car_access_text','is_corner_text','has_legal_text','is_new_house_text','is_renovated_text','is_project_text','is_urban_area_text']
NUM=['log_area','floors','log_frontage','log_road','log_width','log_length','log_bedrooms','log_bathrooms','floors_missing','frontage_missing','road_missing']+FLAGS
GEO=['province','district','ward','street','subtype']
LOCK=threading.Lock();MODEL=None

def norm(s):return ''.join(c for c in unicodedata.normalize('NFKD',str(s).lower().replace('đ','d')) if not unicodedata.combining(c) and c.isalnum())

def features(raw):
    import numpy as np
    import pandas as pd
    area=float(raw['area']);floors=float(raw['floors']) if raw.get('floors') not in ('',None) else np.nan
    if not 10<=area<=2000 or (not math.isnan(floors) and not 1<=floors<=30):raise ValueError('Area must be 10–2,000 m² and floors 1–30.')
    if raw.get('district') not in ['quận đống đa','quận ba đình','quận hai bà trưng','quận cầu giấy','quận thanh xuân','quận hoàng mai','quận nam từ liêm','quận bắc từ liêm','quận hà đông','quận long biên','quận tây hồ','quận hoàn kiếm']:
        raise ValueError('Choose a supported Hanoi district.')
    province=norm('hà nội');district=province+'/'+norm(raw['district']);ward=district+'/'+norm(raw.get('ward') or 'missing');street=ward+'/'+norm(raw.get('street') or 'missing')
    subtype=raw.get('subtype','normal_house')
    if subtype not in ['normal_house','alley_frontage_house','frontage_house','villa']:raise ValueError('Only landed-house demo valuations are supported.')
    row=dict(province=province,district=district,ward=ward,street=street,subtype=norm(subtype),log_area=math.log(area),floors=floors,floors_missing=float(math.isnan(floors)))
    for field,label,hi in [('frontage','frontage',100),('road','road',100),('width','width',100),('length','length',500),('bedrooms','bedrooms',100),('bathrooms','bathrooms',100)]:
        value=raw.get(field);value=float(value) if value not in ('',None) else np.nan
        if not math.isnan(value) and not 0<=value<=hi:raise ValueError(field+' is out of supported range')
        row['log_'+label]=math.log1p(value) if not math.isnan(value) else np.nan
        if label in ['frontage','road']:row[label+'_missing']=float(math.isnan(value))
    for flag in FLAGS:row[flag]=float(bool(raw.get(flag,False)))
    return pd.DataFrame([row])

class AVM:
    def __init__(self):
        import joblib
        import lightgbm as lgb
        self.artifact_hashes={name:hashlib.sha256((ART/name).read_bytes()).hexdigest() for name in ['transform.joblib','lightgbm.txt','calibration.json']}
        manifest=os.environ.get('VINFINX_MODEL_MANIFEST')
        if manifest:
            expected=json.loads(Path(manifest).read_text())
            if any(expected.get(name)!=digest for name,digest in self.artifact_hashes.items()):raise RuntimeError('AVM artifact integrity check failed')
        # Load only trusted, hash-checked operator-owned artifacts; joblib is executable pickle.
        self.transform=joblib.load(ART/'transform.joblib')
        self.model=lgb.Booster(model_file=str(ART/'lightgbm.txt'))
        self.q=json.loads((ART/'calibration.json').read_text())['0.9']
        self.hash=self.artifact_hashes['lightgbm.txt']
        self.predict_lock=threading.Lock()
        path=Path(os.environ.get('VINFINX_HYBRID_PATH',str(ROOT/'Phase 4/combined_20261030/hybrid_calibration/local_phase1_not_combined/results/group_selected.json')))
        self.hybrid=json.loads(path.read_text()) if path.exists() else None
        if manifest and expected.get('hierarchy.json') and not self.hybrid:raise RuntimeError('Expected hierarchy artifact missing')
        if self.hybrid:
            self.artifact_hashes['hierarchy.json']=hashlib.sha256(path.read_bytes()).hexdigest()
            if manifest and expected.get('hierarchy.json')!=self.artifact_hashes['hierarchy.json']:raise RuntimeError('Hierarchy artifact integrity check failed')
        self.version='avm-inference/2.0'

    def predict(self,raw):
        started=time.monotonic()
        if not isinstance(raw,dict):raise ValueError('Property must be a JSON object')
        for key,value in raw.items():
            if not isinstance(key,str) or len(key)>60 or isinstance(value,(dict,list)) or (isinstance(value,str) and len(value)>180):raise ValueError('Invalid property field')
        for key in ['area','floors','frontage','road','width','length','bedrooms','bathrooms']:
            if raw.get(key) not in ('',None) and not math.isfinite(float(raw[key])):raise ValueError('Numeric fields must be finite')
        for key in FLAGS:
            if key in raw and raw[key] not in (True,False,0,1):raise ValueError('Property flags must be boolean')
        if not self.predict_lock.acquire(timeout=15):raise TimeoutError('AVM is busy. Retry shortly.')
        try:result=self._predict(raw)
        finally:self.predict_lock.release()
        result['inference']={'id':secrets.token_hex(12),'version':self.version,'computed_at':time.time(),'elapsed_ms':round((time.monotonic()-started)*1000,2),
                             'input_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),'artifact_sha256':dict(self.artifact_hashes),'live':True}
        return result

    def _predict(self,raw):
        import numpy as np
        frame=features(raw);x=self.transform.transform(frame[GEO+NUM]);logvalue=float(self.model.predict(x,num_threads=2)[0]);q=self.q;adjustment=0.
        method=raw.get('method','lightgbm')
        if method not in ['lightgbm','hierarchical']:raise ValueError('Unknown model method')
        if method=='hierarchical':
            if not self.hybrid:raise RuntimeError('Research calibration artifact unavailable')
            h=self.hybrid['parameters']['hierarchical'];offset=h['root']
            for level,effects in h['levels'].items():offset+=effects.get(frame.iloc[0][level],0.)
            adjustment=max(-math.log(1.5),min(math.log(1.5),h['damping']*offset));logvalue+=adjustment
            q=self.hybrid['interval_log_half_widths']['hierarchical_calibration']
        value=math.exp(logvalue)*1e9
        # LightGBM's native SHAP contributions, summed by original feature family.
        contributions=self.model.predict(x,pred_contrib=True,num_threads=2)
        if isinstance(contributions,list):contributions=contributions[0]
        arr=contributions.toarray()[0] if hasattr(contributions,'toarray') else np.asarray(contributions)[0]
        enc=self.transform.named_transformers_['geo'];offset=0;factors=[]
        for label,cats in zip(['Province','District','Ward','Street','Property subtype'],enc.categories_):
            factors.append({'factor':label,'log_contribution':float(arr[offset:offset+len(cats)].sum())});offset+=len(cats)
        for i,col in enumerate(NUM):factors.append({'factor':col.replace('_',' '),'log_contribution':float(arr[offset+i])})
        return {'value':value,'low':math.exp(logvalue-q)*1e9,'high':math.exp(logvalue+q)*1e9,'nominal_coverage':.9,
                'method':method,'model_version':'phase1-baseline-20261015','model_sha256':self.hash,
                'calibration_log_adjustment':adjustment,'factors':sorted(factors,key=lambda v:abs(v['log_contribution']),reverse=True)[:5],
                'completeness':round(100*sum(raw.get(k) not in ('',None) for k in ['district','ward','street','area','floors','frontage','road','bedrooms','bathrooms'])/9),
                'caveat':'Research AVM trained on asking prices; not an appraisal. SHAP describes model associations, not causal effects. Interval coverage can deteriorate over time.'}

def get_model():
    global MODEL
    with LOCK:
        if MODEL is None:MODEL=AVM()
    return MODEL

def readiness():
    try:
        model=get_model()
        return {'ready':True,'engine':'LightGBM CPU','version':model.version,'artifact_sha256':model.artifact_hashes,
                'methods':[{'id':'lightgbm','ready':True,'label':'LightGBM baseline'},
                           {'id':'hierarchical','ready':bool(model.hybrid),'label':'GBM + hierarchy residual calibration (research)'}],
                'structural_hierarchy':{'ready':False,'reason':'Research candidate not promoted to this inference service'}}
    except Exception as exc:return {'ready':False,'engine':'LightGBM CPU','error':type(exc).__name__,'message':'Trusted model artifacts/dependencies unavailable. No synthetic price returned.'}
