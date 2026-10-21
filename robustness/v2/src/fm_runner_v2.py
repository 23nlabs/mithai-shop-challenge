"""Robustness v2 foundation-model inference (Kaggle GPU). Separate run; original outputs untouched.

(a) Yug-1, TimesFM 2.5, Chronos-Bolt on the 9,360 v2 jobs (calendar-dependence scenarios + rolling
    weekly origins), inputs `history` and `history_clean`.
(b) Yug-1 with four extra sampling seeds (1-4; the original run used 67067) on the ORIGINAL 2,160 jobs.
Histories are regenerated and must hash-match the local builds.
"""
import time
from pathlib import Path

import numpy as np

import benchmark_core as bc
import mithai_core as mc
import robust_core as rc

YUG_SEEDS = (1, 2, 3, 4)


def run(model, jobs, field, batch, out, rec, tag):
    n = len(jobs)
    Q = np.full((n, mc.HORIZON, 9), np.nan, dtype=np.float32)
    fail = 0
    t0 = time.perf_counter()
    for i in range(0, n, batch):
        chunk = jobs[i:i + batch]
        qs = model.forecast([np.asarray(j[field], dtype=float) for j in chunk], mc.HORIZON)
        for k, q in enumerate(qs):
            Q[i + k] = np.asarray(q, dtype=np.float32)
            try:
                bc.check_forecast(q, mc.HORIZON)
            except Exception as exc:
                fail += 1
                rec.append('failures', dict(tag=tag, key=chunk[k]['key'], error=repr(exc)[:200]))
    np.savez_compressed(Path(out) / f'{tag}.npz', q=Q, keys=np.array([j['key'] for j in jobs]))
    rec.log('fm_done', tag=tag, jobs=n, gate_failures=fail, seconds=time.perf_counter() - t0)


def main(calendar_path, out, weights, sha_v2, sha_orig):
    rec = bc.Recorder(out)
    rec.write('environment.json', bc.environment())
    cal = mc.load_calendar(calendar_path)
    jv2 = rc.build_jobs(cal, with_truth=False)
    jor = mc.build_jobs(cal, with_truth=False)
    h2, ho = rc.histories_sha256(jv2), mc.histories_sha256(jor)
    rec.log('histories', v2=h2, v2_match=h2 == sha_v2, original=ho, original_match=ho == sha_orig)
    if h2 != sha_v2 or ho != sha_orig:
        raise RuntimeError('HISTORY_HASH_MISMATCH')
    loads = {}
    for key in ('chronos-bolt-base', 'timesfm-2.5', 'yug'):
        folder, receipts = bc.download_weights(key, weights)
        model = {'yug': bc.Yug, 'timesfm-2.5': bc.TimesFM, 'chronos-bolt-base': bc.ChronosBolt}[key](folder)
        loads[key] = dict(receipts=receipts)
        rec.write('loads.json', loads)
        bc.pilot(model, rec)
        batch = 1 if key == 'yug' else 64
        for field in ('history', 'history_clean'):
            run(model, jv2, field, batch, out, rec, f'v2__{key}__{field}')
        if key == 'yug':
            for s in YUG_SEEDS:
                bc.SEED = s  # read at call time by bc.Yug.forecast
                for field in ('history', 'history_clean'):
                    run(model, jor, field, 1, out, rec, f'orig-seed{s}__yug__{field}')
            bc.SEED = 67067
        del model
    rec.log('complete')
