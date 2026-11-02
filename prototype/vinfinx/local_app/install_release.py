"""Operator-only install: hash-checked private archive; never resets live state."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

ROOT=Path('/home/ubuntu/vinfinx-pilot')

def install(archive,expected):
    os.umask(0o077);archive=Path(archive).resolve()
    if not archive.is_relative_to(ROOT/'incoming'):raise ValueError('Expected a dedicated incoming archive')
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('Release archive hash mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        for item in members:
            path=PurePosixPath(item.name)
            if not item.isfile() or path.is_absolute() or '..' in path.parts:raise ValueError('Only ordinary relative release files allowed')
            if path.parts[0] not in ['local_app','model','fixtures','release.json','index.html','evidence.js','pipeline.js']:raise ValueError('Unexpected release member')
        record=json.load(tar.extractfile('release.json'));tag=record['release']
        if not tag.replace('-','').isalnum():raise ValueError('Invalid release tag')
        target=ROOT/'releases'/tag
        target.mkdir(parents=True,exist_ok=False,mode=0o700)
        tar.extractall(target,filter='data')
    for name,expected_hash in record['files'].items():
        path=(target/'local_app'/name).resolve()
        if not path.is_relative_to(target/'local_app') or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:raise ValueError('Source hash mismatch')
    for name,expected_hash in record.get('research_files',{}).items():
        if name not in ['index.html','evidence.js','pipeline.js'] or hashlib.sha256((target/name).read_bytes()).hexdigest()!=expected_hash:raise ValueError('Research source hash mismatch')
    temporary=ROOT/'current-next'
    if temporary.exists() or temporary.is_symlink():raise ValueError('Unresolved previous deployment pointer')
    temporary.symlink_to(target,target_is_directory=True);temporary.replace(ROOT/'current')
    return {'installed':str(target),'archive_sha256':expected,'state_untouched':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('sha256');args=parser.parse_args()
    print(json.dumps(install(args.archive,args.sha256)))
