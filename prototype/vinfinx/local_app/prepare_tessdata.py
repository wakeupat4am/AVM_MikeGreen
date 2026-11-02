"""Download official pinned local OCR data; reject a digest mismatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

COMMIT='e12c65a915945e4c28e237a9b52bc4a8f39a0cec'

def prepare(destination,manifest):
    destination=Path(destination);os.umask(0o077);destination.mkdir(parents=True,exist_ok=True)
    records={}
    for line in Path(manifest).read_text().splitlines():
        digest,name=line.split()
        if name not in ['eng.traineddata','vie.traineddata','osd.traineddata']:raise ValueError('Unexpected trained data')
        path=destination/name
        if not path.exists():
            url=f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/{COMMIT}/{name}'
            with urllib.request.urlopen(url,timeout=60) as response:data=response.read(40*1024*1024)
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Downloaded OCR model digest mismatch')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Installed OCR model digest mismatch')
        records[name]=digest
    (destination/'manifest.json').write_text(json.dumps({'commit':COMMIT,'sha256':records},indent=2))
    return {'ready':True,'commit':COMMIT,'sha256':records}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination');parser.add_argument('manifest');args=parser.parse_args()
    print(json.dumps(prepare(args.destination,args.manifest)))
