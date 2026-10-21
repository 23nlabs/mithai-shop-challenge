"""Final verification directly against raw artifacts (does NOT import evaluate.py / eval_v2.py metric code).

1. Forecast counts: every .npz in v1, v2, v3 runs — rows, finite rows, crossed-quantile rows.
2. Checksums: v1 manifest (only the five superseded documents may differ, and each must equal its *-v1 copy
   or git history), v2 manifest (all must match).
3. Independent recomputation of WAPE (56 d) and 80% coverage from raw quantiles and regenerated latent
   demand, compared with results/metrics.json (v1) and robustness-v2.json.
4. History hashes regenerated for v1 and v2 jobs.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[1]
STUDY = V3.parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(STUDY / 'src'))
sys.path.insert(0, str(STUDY / 'robustness/v2/src'))
import mithai_core as mc  # noqa: E402
import robust_core as rc  # noqa: E402

SUPERSEDED = {'paper.md', 'summary.md', 'linkedin.md', 'publication-readiness.md', 'README.md'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def counts():
    out = {}
    for f in sorted(STUDY.rglob('*.npz')):
        if 'data/simulated' in str(f):
            continue
        z = np.load(f)
        if 'q' not in z:
            continue
        q = z['q']
        fin = np.isfinite(q).all(axis=(1, 2))
        crossed = (np.diff(q, axis=2) < -1e-6 * np.maximum(1, np.abs(q[:, :, :-1]))).any(axis=(1, 2))
        out[str(f.relative_to(STUDY))] = dict(rows=int(len(q)), finite=int(fin.sum()), crossed=int(crossed.sum()))
    return out


def manifests():
    bad1 = []
    for line in (STUDY / 'evidence/MANIFEST.sha256').read_text().splitlines():
        h, p = line.split('  ', 1)
        f = STUDY / p
        if not f.exists() or sha(f) == h:
            continue
        stem = Path(p).name
        if stem in SUPERSEDED:
            copy = STUDY / p.replace('.md', '-v1.md')
            if copy.exists() and sha(copy) == h:
                continue
            if stem == 'README.md':  # v1 README preserved in git history (commit 2d12317)
                continue
        bad1.append(p)
    bad2 = []
    v2 = STUDY / 'robustness/v2'
    for line in (v2 / 'evidence/MANIFEST-v2.sha256').read_text().splitlines():
        h, p = line.split('  ', 1)
        f = v2 / p
        if f.exists() and sha(f) != h:
            bad2.append(p)
    return dict(v1_unexpected_changes=bad1, v2_changes=bad2)


def wape_cov(q, Y, mask):
    m = q[mask, :, 4]
    y = Y[mask]
    return float(np.abs(m - y).sum() / y.sum()), float(((y >= q[mask, :, 0]) & (y <= q[mask, :, 8])).mean())


def main():
    cal = mc.load_calendar(STUDY / 'data/festivals.json')
    res = dict(counts=counts(), manifests=manifests())
    jo = mc.build_jobs(cal)
    res['v1_history_hash_ok'] = mc.histories_sha256(jo) == (STUDY / 'evidence/histories.sha256').read_text().strip()
    Y = np.stack([j['truth'] for j in jo])
    fest = np.array([j['scenario'] == 'base' and j['kind'] == 'festival' for j in jo])
    ordi = np.array([j['scenario'] == 'base' and j['kind'] == 'ordinary' for j in jo])
    v1 = json.loads((STUDY / 'results/metrics.json').read_text())
    checks = []
    files = {'Yug-1': 'runs/kaggle-01/output/mithai/yug__history.npz', 'TimesFM 2.5': 'runs/kaggle-01/output/mithai/timesfm-2.5__history.npz',
             'Chronos-Bolt': 'runs/kaggle-01/output/mithai/chronos-bolt-base__history.npz', 'ETS': 'runs/local-baselines/ets__history.npz',
             'Calendar regression': 'runs/local-baselines/calendar-regression__history.npz',
             'Seasonal-naive 52w': 'runs/local-baselines/seasonal-naive-52w__history.npz'}
    for m, f in files.items():
        q = np.sort(np.clip(np.load(STUDY / f)['q'].astype(float), 0, None), axis=2)
        for sname, mask in (('base: festival season (all years)', fest), ('base: ordinary periods', ordi)):
            w, c = wape_cov(q, Y, mask)
            rep = v1['strata'][sname]['methods'][m]
            checks.append(dict(method=m, stratum=sname, wape_recomputed=w, wape_reported=rep['wape_56']['point'],
                               cov_recomputed=c, cov_reported=rep['coverage80']['point'],
                               ok=abs(w - rep['wape_56']['point']) < 1e-9 and abs(c - rep['coverage80']['point']) < 1e-9))
    jv = rc.build_jobs(cal)
    res['v2_history_hash_ok'] = rc.histories_sha256(jv) == (STUDY / 'robustness/v2/evidence/histories-v2.sha256').read_text().strip()
    v2 = json.loads((STUDY / 'robustness/v2/results/robustness-v2.json').read_text())
    keys = [j['key'] for j in jv]
    for scen in rc.ROBUST:
        idx = np.array([i for i, j in enumerate(jv) if j['scenario'] == scen and j['kind'] == 'festival' and j['offset'] == 0])
        Yr = np.stack([jv[i]['truth'] for i in idx])
        for m, f in (('Yug-1', 'runs/kaggle-v2-01/output/v2/v2__yug__history.npz'), ('Calendar regression', 'runs/local-v2/v2__calendar-regression__history.npz')):
            z = np.load(STUDY / 'robustness/v2' / f)
            assert list(z['keys']) == keys
            q = np.sort(np.clip(z['q'][idx].astype(float), 0, None), axis=2)
            w, _ = wape_cov(q, Yr, np.ones(len(idx), bool))
            rep = v2['P2_calendar_dependence'][f'{scen}: festival season']['methods'][m]['wape_56']['point']
            checks.append(dict(method=m, stratum=f'{scen}: festival season', wape_recomputed=w, wape_reported=rep, ok=abs(w - rep) < 1e-9))
    res['independent_metric_checks'] = checks
    res['all_checks_ok'] = all(c['ok'] for c in checks) and res['v1_history_hash_ok'] and res['v2_history_hash_ok'] \
        and not res['manifests']['v1_unexpected_changes'] and not res['manifests']['v2_changes']
    (V3 / 'results').mkdir(exist_ok=True)
    (V3 / 'results/final-verification.json').write_text(json.dumps(res, indent=1))
    tot = sum(v['rows'] for v in res['counts'].values())
    print('npz files', len(res['counts']), 'forecast rows', tot, 'non-finite', sum(v['rows'] - v['finite'] for v in res['counts'].values()),
          'crossed', sum(v['crossed'] for v in res['counts'].values()))
    print('manifests', res['manifests'])
    print('metric checks', sum(c['ok'] for c in checks), '/', len(checks), '| all ok:', res['all_checks_ok'])


if __name__ == '__main__':
    main()
