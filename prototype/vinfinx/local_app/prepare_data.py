"""Public-listing aggregates only; no names, contacts or individual listing addresses exported."""
import datetime
import hashlib
import json
import unicodedata
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'assets/market.json'
SOURCE=ROOT/'Phase 4/phase1/data/phase1_full_features.parquet'
CENTERS={'quận đống đa':[21.017,105.827],'quận ba đình':[21.035,105.821],'quận hai bà trưng':[21.006,105.85],
         'quận cầu giấy':[21.034,105.794],'quận thanh xuân':[20.993,105.813],'quận hoàng mai':[20.978,105.861],
         'quận nam từ liêm':[21.01,105.773],'quận bắc từ liêm':[21.073,105.77],'quận hà đông':[20.965,105.767],
         'quận long biên':[21.042,105.886],'quận tây hồ':[21.071,105.816],'quận hoàn kiếm':[21.028,105.854]}

def main():
    d=pd.read_parquet(SOURCE,columns=['id','addr_last_1','addr_last_2','addr_last_3','street_proxy','area_m2','price_vnd'])
    for col in ['addr_last_1','addr_last_2','addr_last_3','street_proxy']:
        d[col]=d[col].map(lambda v:unicodedata.normalize('NFC',v) if isinstance(v,str) else v)
    d=d[d.addr_last_1.eq('hà nội') & d.addr_last_2.isin(CENTERS)].copy()
    locations=[]
    for district,x in d.groupby('addr_last_2'):
        wards=[]
        for ward,y in x.groupby('addr_last_3'):
            if ward not in ('missing',''):
                wards.append({'label':ward,'streets':sorted(v for v in y.street_proxy.dropna().unique() if v not in ('missing',''))})
        locations.append({'label':district,'wards':wards,'center':CENTERS[district]})
    bad=set(pd.read_csv(ROOT/'Phase 4/phase1/data/suspicious_ids.csv').id)
    d=d[~d.id.isin(bad)&d.area_m2.between(10,2000)&d.price_vnd.between(1e8,5e11)].copy()
    d['ppm']=d.price_vnd/d.area_m2;d=d[d.ppm.between(20e6,800e6)&~d.street_proxy.eq('missing')]
    streets=[]
    for (district,street),x in d.groupby(['addr_last_2','street_proxy']):
        if len(x)<8:continue
        streets.append({'district':district,'name':street,'n':len(x),'median':float(x.ppm.median()),
                        'low':float(x.ppm.quantile(.25)),'high':float(x.ppm.quantile(.75)),
                        'center':CENTERS[district],'coordinate_basis':'Approximate district center, not surveyed street geometry'})
    result={'version':'phase1-listing-snapshot/1','source':'Phase1 observed asking-price aggregates, not OpenStreetMap prices or appraisals',
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'generated':datetime.datetime.now().isoformat(),
            'notice':'Historical, unverified listing statistics. Dataset parser limitations remain. Map markers use district centers.',
            'districts':locations,'streets':sorted(streets,key=lambda x:-x['n'])}
    OUT.write_text(json.dumps(result,ensure_ascii=False));print('Prepared',len(locations),'districts and',len(streets),'street aggregates')

if __name__=='__main__':main()
