"""Optional public OSM road geometry; never sends customer/property addresses."""
import json
from pathlib import Path
import re
import urllib.request
import urllib.parse
import argparse
from valuation import norm

path=Path(__file__).resolve().parent/'assets/market.json';data=json.loads(path.read_text())
query='[out:json][timeout:20];way[highway][name~"Tây Sơn|Thái Hà|Thái Thịnh|La Thành|Nguyễn Chí Thanh"](20.97,105.78,21.05,105.86);out geom;'
parser=argparse.ArgumentParser();parser.add_argument('--endpoint',default='https://overpass-api.de/api/interpreter');args=parser.parse_args()
if args.endpoint not in ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter']:raise ValueError('Use an approved public OSM endpoint')
req=urllib.request.Request(args.endpoint,data=urllib.parse.urlencode({'data':query}).encode(),headers={'User-Agent':'VinFinx-local-demo/1.0 (public Hanoi road geometry)'})
with urllib.request.urlopen(req,timeout=35) as response:result=json.load(response)
def street_key(s):return norm(re.sub(r'^(?:(?:đường|phố)\s*)+','',s.lower()))
ways=[]
for way in result.get('elements',[]):
    key=street_key(way.get('tags',{}).get('name',''))
    matches=[i for i,s in enumerate(data['streets']) if street_key(s['name'])==key]
    if matches and way.get('geometry'):
        # The public-road geometry can cross district boundaries. One aggregate is selected, not a parcel valuation.
        ways.append({'osm_id':way['id'],'street_index':matches[0],'coords':[[p['lat'],p['lon']] for p in way['geometry']]})
data['ways']=ways;data['geometry_source']='OpenStreetMap contributors via Overpass API; public roads only'
path.write_text(json.dumps(data,ensure_ascii=False));print('Added',len(ways),'public OSM road segments')
