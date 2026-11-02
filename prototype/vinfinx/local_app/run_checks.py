import datetime
import hashlib
import json
from pathlib import Path
import unittest

root=Path(__file__).resolve().parent
suite=unittest.defaultTestLoader.discover(str(root),pattern='test_local.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
out=root/'test-results';out.mkdir(exist_ok=True)
record={'at':datetime.datetime.now().isoformat(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'passed':result.wasSuccessful(),'database':'Temporary isolated SQLite, not the user demo database',
        'coverage':['role authentication/authorization','CSRF and customer isolation','actual AVM inference','PDF and image OCR','human approval gates','invalid-file rejection','audit/report/chat/appointments','20-row frozen-model parity'],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.suffix in ['.py','.js','.css','.html']}}
(out/'checks.json').write_text(json.dumps(record,indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
