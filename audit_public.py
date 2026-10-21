"""Read-only publication boundary, data-inventory and rights audit."""
import hashlib
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE = {'PACKAGE.sha256'}
DATA_PATHS = {'data/festivals.json', 'evidence/config.json', 'results/metrics.json', 'results/tables.md',
              'robustness/v2/results/tables-v2.md', 'robustness/v2/results/robustness-v2.json',
              'robustness/v3/results/final-v3.json', 'robustness/v3/results/final-v3-addon.json',
              'robustness/v3/results/final-verification.json', 'final/claims-ledger.json'}
PATTERNS = {
    'LOCAL_PATH': re.compile(rb'/(?:Users|Volumes)/[A-Za-z0-9_./-]+'),
    'EMAIL': re.compile(rb'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I),
    'SECRET_ASSIGNMENT': re.compile(rb'(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[=:]\s*["\x27]?[A-Za-z0-9_./+\-=]{12,}'),
    'GITHUB_TOKEN': re.compile(rb'gh[opsu]_[A-Za-z0-9]{20,}'),
    'PRIVATE_KEY': re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
}

def event(run, name, outcome, **fields):
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), severity='INFO' if outcome == 'pass' else 'ERROR',
                          event=name, app='mithai-public-audit', version='source-7069f5a', run_id=run,
                          outcome=outcome, **fields), sort_keys=True), flush=True)

def audit():
    errors=[]
    manifest=ROOT/'PACKAGE.sha256'
    lines=manifest.read_text().splitlines()
    expected={line.split('  ',1)[1] for line in lines}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    for rel in sorted(actual-(expected|EXCLUDE)):
        errors.append('UNEXPECTED_FILE:'+rel)
    for rel in sorted((expected|EXCLUDE)-actual):
        errors.append('MISSING_FILE:'+rel)
    for rel in sorted(actual):
        p=ROOT/rel
        if p.is_symlink() or p.suffix.lower() in {'.npz','.npy','.pt','.bin','.ckpt','.safetensors','.pem','.key','.log'} or p.name.startswith('.env'):
            errors.append('FORBIDDEN_FILE:'+rel)
            continue
        if p.suffix.lower() in {'.png'}:
            continue
        b=p.read_bytes()
        for code,pat in PATTERNS.items():
            if pat.search(b): errors.append('PRIVACY_MATCH:'+code+':'+rel)
    inv=json.loads((ROOT/'DATA_INVENTORY.json').read_text())
    if {x['path'] for x in inv['artifacts']} != DATA_PATHS:
        errors.append('DATA_INVENTORY:PATHS')
    for item in inv['artifacts']:
        rel=item['path']; p=ROOT/rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            errors.append('DATA_INVENTORY:HASH:'+rel)
        if rel=='data/festivals.json':
            if item['license']!='upstream terms retained; no 23N relicensing' or not item.get('source_attribution'):
                errors.append('CALENDAR_PROVENANCE')
        elif item['license']!='CC-BY-4.0' or item['data_type']!='synthetic-derived':
            errors.append('DATA_INVENTORY:TERMS:'+rel)
    cal=json.loads((ROOT/'data/festivals.json').read_text())
    if len(cal)!=78 or any(not x.get('source_url') or not x.get('source_sha256') for x in cal):
        errors.append('CALENDAR_PROVENANCE:RECORDS')
    notice=(ROOT/'NOTICE.md').read_text()
    release=json.loads((ROOT/'RELEASE.json').read_text())
    if not all(s in notice for s in ('Apache License 2.0','Creative Commons Attribution 4.0','outside both 23N grants','Office Holidays','Drik Panchang')):
        errors.append('LICENSE_MISSING:NOTICE')
    if release.get('code_license')!='Apache-2.0 (original code only)' or release.get('original_content_and_synthetic_data_license')!='CC-BY-4.0' or release.get('doi') is not None or release.get('venue') is not None:
        errors.append('LICENSE_MISSING:METADATA')
    if not (ROOT/'LICENSE-CODE.txt').read_text().lstrip().startswith('Apache License\n'):
        errors.append('LICENSE_MISSING:APACHE')
    if 'Creative Commons Attribution 4.0 International Public License' not in (ROOT/'LICENSE-CONTENT.txt').read_text():
        errors.append('LICENSE_MISSING:CCBY')
    return errors, {'files':len(actual),'data_artifacts':len(inv['artifacts']),'calendar_records':len(cal)}

def main():
    run=uuid.uuid4().hex[:12]; start=time.monotonic(); event(run,'audit_start','pass')
    try: errors,counts=audit()
    except (OSError,ValueError,KeyError) as exc: errors,counts=['INPUT_ERROR:'+type(exc).__name__],{}
    event(run,'audit_complete','fail' if errors else 'pass',error_codes=errors,duration_ms=round((time.monotonic()-start)*1000),**counts)
    return bool(errors)
if __name__=='__main__': sys.exit(main())
