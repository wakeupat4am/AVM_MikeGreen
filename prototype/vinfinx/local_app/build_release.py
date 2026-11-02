"""Private deployment bundle: allowlisted source, trusted models and synthetic fixtures only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

from valuation import ART,ROOT,get_model

def build():
    os.umask(0o077);here=Path(__file__).resolve().parent
    files=[p for p in here.rglob('*') if p.is_file() and p.suffix in ['.py','.js','.css','.html','.png','.svg','.json','.txt'] and not any(part in ['__pycache__','test-results','pilot-results'] for part in p.relative_to(here).parts)]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    research=[here.parent/name for name in ['index.html','evidence.js','pipeline.js']]
    code_hash=hashlib.sha256(''.join(str(p.relative_to(here.parent))+digest(p) for p in sorted(files+research)).encode()).hexdigest()
    tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+code_hash[:10]
    output=here.parent/'pilot-private/releases';output.mkdir(parents=True,exist_ok=True);output.parent.chmod(0o700)
    archive=output/(tag+'.tar.gz');model=get_model()
    with tempfile.TemporaryDirectory(prefix='vinfinx-release-') as temporary:
        temp=Path(temporary);(temp/'manifest.json').write_text(json.dumps(model.artifact_hashes,indent=2))
        record={'release':tag,'source_sha256':code_hash,'files':{str(p.relative_to(here)):digest(p) for p in files},'research_files':{p.name:digest(p) for p in research},'models':model.artifact_hashes,'service':'VinFinx x Vietcombank','synthetic_pilot':True}
        (temp/'release.json').write_text(json.dumps(record,indent=2))
        with tarfile.open(archive,'w:gz') as tar:
            for p in files:tar.add(p,arcname='local_app/'+str(p.relative_to(here)),recursive=False)
            for p in research:tar.add(p,arcname=p.name,recursive=False)
            for name in ['transform.joblib','lightgbm.txt','calibration.json']:tar.add(ART/name,arcname='model/'+name)
            hybrid=Path(os.environ.get('VINFINX_HYBRID_PATH',str(ROOT/'Phase 4/combined_20261030/hybrid_calibration/local_phase1_not_combined/results/group_selected.json')))
            if model.hybrid:tar.add(hybrid,arcname='model/hierarchy.json')
            tar.add(temp/'manifest.json',arcname='model/manifest.json');tar.add(temp/'release.json',arcname='release.json')
            for suffix in ['.pdf','.png']:tar.add(ROOT/'output/pdf'/('vinfinx-sample-payslip'+suffix),arcname='fixtures/output/pdf/vinfinx-sample-payslip'+suffix)
    archive.chmod(0o600)
    info={'archive':str(archive),'sha256':digest(archive),'release':tag};(output/'latest.json').write_text(json.dumps(info,indent=2))
    return info

if __name__=='__main__':print(json.dumps(build()))
