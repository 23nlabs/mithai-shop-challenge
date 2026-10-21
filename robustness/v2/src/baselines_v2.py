"""Robustness v2 local forecasters. Imports the original baselines unchanged and adds two methods:

shopkeeper-calibrated   the original shopkeeper point forecast × empirical quantiles of his own one-step
                        weekday-level errors (sales / mean of previous 4 same weekdays) over the last 364
                        non-festival days of history. Gives 9 quantiles so the SAME critical-ratio policy
                        can be applied to him as to every model.
calreg-censoring-aware  calendar regression where sold-out days (the shop's own stockout records, available
                        at the cutoff) are treated as right-censored: Tobit-style EM in log space
                        (10 iterations), residual quantiles from uncensored days.
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

V2 = Path(__file__).resolve().parents[1]
STUDY = V2.parents[1]
sys.path.insert(0, str(STUDY / 'src'))
sys.path.insert(0, str(V2 / 'src'))
import baselines as b0  # noqa: E402  (original, unmodified)
import mithai_core as mc  # noqa: E402
import robust_core as rc  # noqa: E402

Q = np.asarray(mc.QUANTILES)
H = mc.HORIZON


def shopkeeper_calibrated(j, cal):
    point = b0.shopkeeper(j, cal)[:, 4]
    hist, h0, ci = j['history'], j['h0'], j['ci']
    offs = mc.festival_offsets(cal, h0, ci)
    clean = np.array([o is None for o in offs])
    idx = np.arange(max(28, len(hist) - 364), len(hist))
    ok = clean[idx] & clean[idx - 7] & clean[idx - 14] & clean[idx - 21] & clean[idx - 28]
    idx = idx[ok]
    level = (hist[idx - 7] + hist[idx - 14] + hist[idx - 21] + hist[idx - 28]) / 4
    ratio = hist[idx] / np.maximum(level, 1e-6)
    qr = np.quantile(ratio, Q)
    return np.clip(point[:, None] * qr[None, :], 0, None)


def calreg_censoring_aware(j, cal, alpha=1.0, iters=10):
    h0, ci = j['h0'], j['ci']
    X = b0._design(np.arange(h0, ci), cal)
    y_obs = np.log1p(j['history'])
    cens = np.asarray(j['stockout'], dtype=bool)
    A = X.T @ X + alpha * np.eye(X.shape[1])
    A[0, 0] -= alpha
    y = y_obs.copy()
    for _ in range(iters):
        beta = np.linalg.solve(A, X.T @ y)
        mu = X @ beta
        sd = np.std((y_obs - mu)[~cens])
        a = (y_obs[cens] - mu[cens]) / sd
        y[cens] = mu[cens] + sd * norm.pdf(a) / np.maximum(norm.sf(a), 1e-12)  # E[Y | Y > observed]
        y[cens] = np.maximum(y[cens], y_obs[cens])
    res = (y_obs - X @ beta)[~cens][-730:]
    mu_f = b0._design(np.arange(ci, ci + H), cal) @ beta
    return np.clip(np.expm1(mu_f[:, None] + np.quantile(res, Q)[None, :]), 0, None)


NEW = {'shopkeeper-calibrated': shopkeeper_calibrated, 'calreg-censoring-aware': calreg_censoring_aware}
ALL = list(b0.METHODS) + list(NEW)


def _one(args):
    name, j, cal = args
    if name in NEW:
        return NEW[name](j, cal)
    return b0._one((name, j, cal))


def run(jobs, cal, out_dir, names, prefix):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = np.array([j['key'] for j in jobs])
    slim = [{k: v for k, v in j.items() if k not in ('truth', 'no_diwali', 'cap', 'confirmed')} for j in jobs]
    for name in names:
        f = out_dir / f'{prefix}__{name}__history.npz'
        if f.exists():
            print('skip existing', f.name, flush=True)
            continue
        with mp.get_context('fork').Pool(mp.cpu_count()) as pool:
            qs = pool.map(_one, [(name, j, cal) for j in slim], chunksize=8)
        np.savez_compressed(f, q=np.asarray(qs, dtype=np.float32), keys=keys)
        print(prefix, name, 'done', flush=True)


def extend_calendar_features(days=150):
    """Forecast windows of late rolling origins run past 2026-12-31. Extend ONLY the calendar feature
    arrays used by forecasters (weekday, day-of-year, trend clock, wedding flag); called after all jobs
    are built, so simulation and history hashes are unaffected."""
    import datetime as dt
    extra = [mc.END + dt.timedelta(days=i + 1) for i in range(days)]
    mc.WEEKDAY = np.concatenate([mc.WEEKDAY, [d.weekday() for d in extra]])
    mc.DOY = np.concatenate([mc.DOY, [d.timetuple().tm_yday for d in extra]])
    mc.YEARS_FROM_START = np.arange(mc.N + days) / 365.25
    mc.WEDDING_MASK = np.concatenate([mc.WEDDING_MASK, [mc.in_wedding(d) for d in extra]])


def original_jobs_with_flags(cal):
    jobs = mc.build_jobs(cal, with_truth=False)
    cache = {}
    for j in jobs:
        k = (j['scenario'], j['seed'])
        if k not in cache:
            cache = {k: rc.simulate(j['seed'], cal, j['scenario'])['stockout']}
        j['stockout'] = cache[k][mc.PRODUCTS.index(j['product']), j['h0']:j['ci']]
    return jobs


if __name__ == '__main__':
    cal = mc.load_calendar(STUDY / 'data/festivals.json')
    out = V2 / 'runs/local-v2'
    run(original_jobs_with_flags(cal), cal, out, list(NEW), 'orig')
    order = ['seasonal-naive-52w', 'calendar-regression', 'shopkeeper', 'shopkeeper-calibrated',
             'calreg-censoring-aware', 'ets', 'ets+calendar']
    v2jobs = rc.build_jobs(cal, with_truth=False)
    extend_calendar_features()
    run(v2jobs, cal, out, order, 'v2')
