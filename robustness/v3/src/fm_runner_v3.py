"""v3: Yug-1 sampling seeds 1-4 on the 9,360 v2 jobs (calendar-dependence scenarios + rolling origins),
inputs `history` and `history_clean`. Seed 67067 outputs come from the v2 run. Histories must hash-match."""
import time
from pathlib import Path

import numpy as np

import benchmark_core as bc
import mithai_core as mc
import robust_core as rc

SEEDS = (1, 2, 3, 4)


def main(calendar_path, out, weights, sha_v2):
    rec = bc.Recorder(out)
    rec.write('environment.json', bc.environment())
    cal = mc.load_calendar(calendar_path)
    jobs = rc.build_jobs(cal, with_truth=False)
    h = rc.histories_sha256(jobs)
    rec.log('histories', v2=h, match=h == sha_v2)
    if h != sha_v2:
        raise RuntimeError('HISTORY_HASH_MISMATCH')
    folder, receipts = bc.download_weights('yug', weights)
    rec.write('loads.json', dict(yug=dict(receipts=receipts)))
    model = bc.Yug(folder)
    bc.pilot(model, rec)
    for s in SEEDS:
        bc.SEED = s
        for field in ('history', 'history_clean'):
            Q = np.full((len(jobs), mc.HORIZON, 9), np.nan, dtype=np.float32)
            fail = 0
            t0 = time.perf_counter()
            for i, j in enumerate(jobs):
                q = model.forecast([np.asarray(j[field], dtype=float)], mc.HORIZON)[0]
                Q[i] = np.asarray(q, dtype=np.float32)
                try:
                    bc.check_forecast(q, mc.HORIZON)
                except Exception:
                    fail += 1
            np.savez_compressed(Path(out) / f'v2-seed{s}__yug__{field}.npz', q=Q, keys=np.array([j['key'] for j in jobs]))
            rec.log('fm_done', seed=s, input=field, jobs=len(jobs), gate_failures=fail, seconds=time.perf_counter() - t0)
    rec.log('complete')
