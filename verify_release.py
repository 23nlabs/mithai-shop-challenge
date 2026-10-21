"""Read-only integrity gate for the public synthetic research package."""
import argparse
import hashlib
import json
import re
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def event(run, name, outcome, **counts):
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), severity='INFO' if outcome == 'pass' else 'ERROR',
                          event=name, app='mithai-public-package', version='source-7069f5a', run_id=run,
                          outcome=outcome, **counts), sort_keys=True), flush=True)


def verify(root):
    errors = []
    seen = 0
    manifest = root / 'PACKAGE.sha256'
    if not manifest.exists():
        errors.append('MANIFEST_MISSING')
    else:
        for line in manifest.read_text().splitlines():
            digest, rel = line.split('  ', 1)
            p = root / rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                errors.append('HASH_MISMATCH:' + rel)
            seen += 1
    ledger = json.loads((root / 'final/claims-ledger.json').read_text())
    if len(ledger['claims']) != 40 or ledger['failures'] or any(not c['value_matches_print'] or not all(c['present_in'].values()) for c in ledger['claims']):
        errors.append('CLAIMS_INVALID')
    notebooks = 0
    for p in sorted((root / 'final/notebooks').glob('*.ipynb')):
        n = json.loads(p.read_text())
        notebooks += 1
        if n['nbformat'] != 4 or n.get('metadata', {}).get('kernelspec', {}).get('name') != 'python3':
            errors.append('NOTEBOOK_METADATA:' + p.name)
        for c in n['cells']:
            if c['cell_type'] == 'code' and (c.get('outputs') or c.get('execution_count') is not None):
                errors.append('NOTEBOOK_OUTPUT:' + p.name)
            s = ''.join(c['source'])
            if re.search(r'/Users/|/Volumes/', s, re.I):
                errors.append('NOTEBOOK_SCOPE:' + p.name)
    if notebooks != 3:
        errors.append('NOTEBOOK_COUNT')
    figs = 0
    for p in (root / 'final/figures').iterdir():
        b = p.read_bytes(); figs += 1
        if p.suffix == '.png' and (not b.startswith(b'\x89PNG\r\n\x1a\n') or b[-8:-4] != b'IEND'):
            errors.append('FIGURE_INVALID:' + p.name)
        if p.suffix == '.svg':
            try: ET.fromstring(b)
            except ET.ParseError: errors.append('FIGURE_INVALID:' + p.name)
    if figs != 12:
        errors.append('FIGURE_COUNT')
    return errors, dict(files=seen, claims=len(ledger['claims']), notebooks=notebooks, figures=figs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    a = parser.parse_args()
    start = time.monotonic(); run = uuid.uuid4().hex[:12]
    event(run, 'verify_start', 'pass')
    try:
        errors, counts = verify(a.root)
    except (OSError, ValueError, KeyError) as exc:
        errors, counts = ['INPUT_ERROR:' + type(exc).__name__], {}
    if errors:
        event(run, 'verify_complete', 'fail', error_codes=errors, duration_ms=round((time.monotonic()-start)*1000), **counts)
        return 1
    event(run, 'verify_complete', 'pass', duration_ms=round((time.monotonic()-start)*1000), **counts)
    return 0


if __name__ == '__main__':
    sys.exit(main())
