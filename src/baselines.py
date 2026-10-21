"""Local (CPU) forecasters for the Mithai Shop Challenge. Every method sees only `history`
(observed sales before the cutoff) — plus, for Competition B methods, the public festival calendar.

Competition A (history only)
  seasonal-naive-52w   same weekday 52 weeks earlier (t-364); residual quantiles of 364-day differences
  ets                  ETS(A,N,A), weekly season, fitted on the history window; Gaussian intervals
Competition B (calendar-informed)
  calendar-regression  ridge regression on log1p(sales): trend, weekday, yearly Fourier(4), wedding-season
                       flag, festival-offset bins (calendar dates known in advance); residual quantiles
  shopkeeper           explicit heuristic: weekday level from the last 4 comparable weekdays × last year's
                       same-festival uplift by days-to-festival; point forecast only
  ets+calendar         ETS on calendar-cleaned history × history-estimated festival profile
                       (the same application-layer correction applied to the foundation models)
"""
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mithai_core as mc  # noqa: E402

Q = np.asarray(mc.QUANTILES)
H = mc.HORIZON


def seasonal_naive_52w(j):
    hist, ci, h0 = j['history'], j['ci'], j['h0']
    pos = np.arange(ci, ci + H) - 364 - h0
    # beyond the history (t-364 >= cutoff) does not occur for H=56; assert anyway
    assert pos.max() < len(hist)
    point = hist[pos]
    res = hist[364:] - hist[:-364]
    res = res[-364:]
    off = np.quantile(res, Q) - np.median(res)
    return np.clip(point[:, None] + off[None, :], 0, None)


def _ets(y):
    import pandas as pd
    from scipy.stats import norm
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = ETSModel(pd.Series(np.asarray(y, dtype='float64')), error='add', trend=None, seasonal='add',
                       seasonal_periods=7).fit(disp=False, maxiter=2000)
    pred = fit.get_prediction(start=len(y), end=len(y) + H - 1)
    c = np.asarray(pred.predicted_mean)
    sd = np.sqrt(np.asarray(pred.var_pred_mean))
    return c[:, None] + sd[:, None] * __import__('scipy.stats', fromlist=['norm']).norm.ppf(Q)[None, :]


def ets_raw(j):
    return np.clip(_ets(j['history']), 0, None)


def ets_calendar(j):
    q = _ets(j['history_clean'])
    return np.clip(apply_future_profile(q, j), 0, None)


def apply_future_profile(q, j):
    out = np.array(q, dtype=float)
    for i, o in enumerate(j['future_offsets']):
        if o is None or o[0] not in j['profile']:
            continue
        r = max(j['profile'][o[0]][o[1] - mc.CORRECTION_WINDOWS[o[0]][0]], 1e-3)
        out[i] *= r
    return out


DIWALI_BINS = [(-35, -29), (-28, -22), (-21, -15), (-14, -8), (-7, -4), (-3, -1), (0, 0), (1, 1), (2, 3), (4, 5)]
OTHER_BINS = {'Dussehra': [(-10, -1), (0, 0)], 'Holi': [(-3, 0)], 'Raksha Bandhan': [(-4, -1), (0, 0)],
              'Ganesh Chaturthi': [(0, 0), (1, 10)]}


def _design(idx, cal):
    """Features for absolute day indices `idx` (calendar dates are public and known in advance)."""
    idx = np.asarray(idx)
    n = len(idx)
    cols = [np.ones(n), mc.YEARS_FROM_START[idx]]
    wd = mc.WEEKDAY[idx]
    cols += [(wd == k).astype(float) for k in range(1, 7)]
    doy = mc.DOY[idx] / 365.25
    for k in range(1, 5):
        cols += [np.sin(2 * np.pi * k * doy), np.cos(2 * np.pi * k * doy)]
    cols.append(mc.WEDDING_MASK[idx].astype(float))
    bins = [('Diwali', b) for b in DIWALI_BINS] + [(f, b) for f, bs in OTHER_BINS.items() for b in bs]
    for f, (lo, hi) in bins:
        c = np.zeros(n)
        for day in cal[f].values():
            d = mc.IDX.get(day)
            if d is None:
                continue
            c += ((idx - d >= lo) & (idx - d <= hi))
        cols.append(np.minimum(c, 1))
    return np.column_stack(cols)


def calendar_regression(j, cal, alpha=1.0):
    h0, ci = j['h0'], j['ci']
    X = _design(np.arange(h0, ci), cal)
    y = np.log1p(j['history'])
    A = X.T @ X + alpha * np.eye(X.shape[1])
    A[0, 0] -= alpha  # do not shrink the intercept
    beta = np.linalg.solve(A, X.T @ y)
    res = (y - X @ beta)[-730:]
    Xf = _design(np.arange(ci, ci + H), cal)
    mu = Xf @ beta
    return np.clip(np.expm1(mu[:, None] + np.quantile(res, Q)[None, :]), 0, None)


def shopkeeper(j, cal):
    """Point forecast. Weekday level = mean of the last 4 same-weekday sales outside festival windows.
    Festival days: level × last year's ratio (sales on the same days-to-festival ÷ last year's 4-week
    pre-season mean, 3-day smoothed). No growth term, no uncertainty."""
    hist, h0, ci = j['history'], j['h0'], j['ci']
    offs = mc.festival_offsets(cal, h0, ci)
    level = np.zeros(7)
    for w in range(7):
        vals = [hist[i] for i in range(len(hist) - 1, -1, -1) if mc.WEEKDAY[h0 + i] == w and offs[i] is None][:4]
        level[w] = np.mean(vals)
    out = level[mc.WEEKDAY[ci:ci + H]].astype(float)
    year = j['year']
    for i, o in enumerate(j['future_offsets']):
        if o is None:
            continue
        f, k = o
        ly = cal[f].get(year - 1)
        d = mc.IDX.get(ly) if ly else None
        if d is None or d - 63 < h0 or d + k + 1 >= ci:
            continue
        pre = hist[d - 63 - h0:d - 35 - h0].mean()
        sm = hist[d + k - 1 - h0:d + k + 2 - h0].mean()
        out[i] = level[mc.WEEKDAY[ci + i]] * sm / max(pre, 1e-6)
    return np.repeat(out[:, None], 9, axis=1)


METHODS = {'seasonal-naive-52w': 'A', 'ets': 'A', 'calendar-regression': 'B', 'shopkeeper': 'B', 'ets+calendar': 'B'}


def _one(args):
    name, j, cal = args
    if name == 'seasonal-naive-52w':
        return seasonal_naive_52w(j)
    if name == 'ets':
        return ets_raw(j)
    if name == 'ets+calendar':
        return ets_calendar(j)
    if name == 'calendar-regression':
        return calendar_regression(j, cal)
    if name == 'shopkeeper':
        return shopkeeper(j, cal)
    raise ValueError(name)


def run_all(jobs, cal, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = np.array([j['key'] for j in jobs])
    slim = [{k: v for k, v in j.items() if k not in ('truth', 'no_diwali', 'cap', 'confirmed')} for j in jobs]
    for name in METHODS:
        with mp.get_context('fork').Pool(mp.cpu_count()) as pool:
            qs = pool.map(_one, [(name, j, cal) for j in slim], chunksize=8)
        np.savez_compressed(out_dir / f'{name}__history.npz', q=np.asarray(qs, dtype=np.float32), keys=keys)
        print(name, 'done', flush=True)


if __name__ == '__main__':
    here = Path(__file__).resolve().parents[1]
    cal = mc.load_calendar(here / 'data/festivals.json')
    jobs = mc.build_jobs(cal, with_truth=False)
    run_all(jobs, cal, here / 'runs/local-baselines')
