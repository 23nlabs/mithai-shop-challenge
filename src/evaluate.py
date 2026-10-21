"""Scoring and inventory simulation for the Mithai Shop Challenge (pre-specified before results).

Forecast handling (identical for every method): negative values clipped to 0; crossed quantiles are
counted as gate failures and then rearranged by sorting (monotone rearrangement). The shopkeeper
heuristic is point-only: probabilistic metrics are N/A for it.

Accuracy (median forecast vs latent demand):
  WAPE_h   sum|y - m| / sum y over the first h days, pooled over jobs (h = 28, 42, 56)
  MAE_h    mean |y - m| in kg
  weekly   WAPE by horizon week 1..8
  peak     7-day centred moving sums; actual peak = argmax of truth, forecast peak = argmax of median.
           Scored only for festival-season jobs with Diwali on horizon day 7..48 and a DISTINCT actual
           peak (max 7-day sum >= 1.5 x median 7-day sum). error = forecast - actual (days). A forecast
           without its own distinct peak (max 7-day sum < 1.2 x its median) counts as 'no peak forecast'
           (rule added after the baselines-only dry run; ETS forecasts are flat).
  surge    window = Diwali-35 .. Diwali+3 inside the horizon; actual surge = sum(truth - no_diwali),
           forecast surge = sum(median - no_diwali); relative error = (fc - act) / act.
  phantom  (shift years 2023, 2026 only) forecast excess over truth on last year's Diwali date
           +364 days +/-5, as a share of truth in that window: the "Diwali trap" metric.
  phantom_rel (post-hoc diagnostic, added after first results) phantom ratio divided by the method's
           forecast/actual ratio over the whole horizon — removes overall level bias.
  prob.    80% coverage (q0.1..q0.9), width / mean truth, wQL = 2*sum pinball / (9*sum y).
Business (latent demand, 56-day horizon, forecast fixed at the cutoff):
  PRIMARY  'committed': the production plan is fixed at the cutoff (ingredients pre-procured), no feedback.
  SECONDARY 'order-up-to': production tops up stock on hand to the planned level each day.
  (Primary/secondary roles were set after a baselines-only dry run showed ~0 % waste for every method
  under order-up-to, which hides the cost of over-forecasting; before any foundation-model output was seen.)
  target(t) = max(Q_tau(t), confirmed advance orders(t)); tau = critical ratio
  (price - cost + goodwill) / (price - salvage_expired + goodwill), salvage_expired = 0.
  Shopkeeper: target = 1.10 x point forecast (his documented safety margin).
  Production = target(t) (committed) or order-up-to target(t) given stock on hand, capped at daily capacity
  (1.4 x historical capacity); targets above capacity are pre-built on earlier days.
  FIFO sales, stock older than shelf life is wasted, unexpired stock at day 56 salvaged at 50 % of cost.
  Hindsight reference: same mechanics with target = actual demand. Regret = reference - policy.
Uncertainty: bootstrap over independent simulated shops (seeds), 2,000 draws, seed 23.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mithai_core as mc  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
Q = np.asarray(mc.QUANTILES)
H = mc.HORIZON
B = 2000

FMS = ('yug', 'timesfm-2.5', 'chronos-bolt-base')
METHODS = {
    # name: (competition, source file stem, input field, calendar-corrected?)
    'Yug-1': ('A', 'yug', 'history', False),
    'TimesFM 2.5': ('A', 'timesfm-2.5', 'history', False),
    'Chronos-Bolt': ('A', 'chronos-bolt-base', 'history', False),
    'Seasonal-naive 52w': ('A', 'seasonal-naive-52w', 'history', False),
    'ETS': ('A', 'ets', 'history', False),
    'Calendar regression': ('B', 'calendar-regression', 'history', False),
    'Shopkeeper': ('B', 'shopkeeper', 'history', False),
    'ETS + calendar': ('B', 'ets+calendar', 'history', False),
    'Yug-1 + calendar': ('B', 'yug', 'history_clean', True),
    'TimesFM + calendar': ('B', 'timesfm-2.5', 'history_clean', True),
    'Chronos + calendar': ('B', 'chronos-bolt-base', 'history_clean', True),
}
ECON = {  # INR per kg; shelf life in days. Documented assumptions for an Ahmedabad shop, not survey data.
    'kaju-katli': dict(price=1000.0, cost=600.0, shelf=10),
    'motichoor-ladoo': dict(price=450.0, cost=250.0, shelf=5),
    'gulab-jamun': dict(price=320.0, cost=160.0, shelf=4),
}
HOLD = 2.0          # INR / kg / day
END_SALVAGE = 0.5   # share of cost recovered for unexpired stock at horizon end
CAP_FACTOR = 1.4    # festival production capacity vs historical capacity
SHOPKEEPER_MARGIN = 1.10


def load_forecasts(jobs, dirs):
    keys = np.array([j['key'] for j in jobs])
    raw = {}
    for d in dirs:
        for f in Path(d).glob('*__*.npz'):
            z = np.load(f)
            if not np.array_equal(z['keys'], keys):
                raise RuntimeError(f'KEY_ORDER_MISMATCH {f}')
            raw[f.stem] = z['q'].astype(float)
    F, gate = {}, {}
    for name, (_, stem, field, corrected) in METHODS.items():
        k = f'{stem}__{field}'
        if k not in raw:
            continue
        q = raw[k].copy()
        nonfinite = ~np.isfinite(q).all(axis=(1, 2))
        crossed = (np.diff(q, axis=2) < -1e-6 * np.maximum(1, np.abs(q[:, :, :-1]))).any(axis=(1, 2))
        q = np.where(np.isfinite(q), q, 0.0)
        if corrected:
            for i, j in enumerate(jobs):
                for t, o in enumerate(j['future_offsets']):
                    if o is None or o[0] not in j['profile']:
                        continue
                    q[i, t] *= max(j['profile'][o[0]][o[1] - mc.CORRECTION_WINDOWS[o[0]][0]], 1e-3)
        q = np.sort(np.clip(q, 0, None), axis=2)
        F[name] = q
        gate[name] = dict(nonfinite=int(nonfinite.sum()), crossed_quantiles=int(crossed.sum()), n=len(jobs))
    return F, gate


def roll7(x):
    k = np.ones(7)
    return np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), -1, x)


def per_job_accuracy(jobs, F, cal):
    Y = np.stack([j['truth'] for j in jobs])
    ND = np.stack([j['no_diwali'] for j in jobs])
    out = {}
    for name, q in F.items():
        m = q[:, :, 4]
        e = np.abs(Y - m)
        rec = dict(abs_err={h: e[:, :h].sum(1) for h in mc.HORIZONS}, actual={h: Y[:, :h].sum(1) for h in mc.HORIZONS},
                   week_err=e.reshape(len(jobs), 8, 7).sum(2), week_act=Y.reshape(len(jobs), 8, 7).sum(2))
        if name != 'Shopkeeper':
            err = Y[:, :, None] - q
            pin = np.maximum(Q * err, (Q - 1) * err).sum((1, 2))
            rec['pinball'] = 2 * pin / 9
            rec['covered'] = ((Y >= q[:, :, 0]) & (Y <= q[:, :, 8])).sum(1)
            rec['width'] = (q[:, :, 8] - q[:, :, 0]).sum(1)
        peak, surge, phantom, no_peak, phantom_rel = (np.full(len(jobs), np.nan) for _ in range(5))
        ry, rm = roll7(Y), roll7(m)
        for i, j in enumerate(jobs):
            D = mc.IDX[cal['Diwali'][j['year']]] - j['ci']
            if j['kind'] == 'festival' and 7 <= D <= 48 and ry[i].max() >= 1.5 * np.median(ry[i]):
                # forecast must itself contain a distinct peak, otherwise 'no peak forecast' (counted separately)
                no_peak[i] = 0.0
                if rm[i].max() >= 1.2 * np.median(rm[i]):
                    peak[i] = np.argmax(rm[i]) - np.argmax(ry[i])
                else:
                    no_peak[i] = 1.0
            if 0 <= D < H:
                w = slice(max(0, D - 35), min(H, D + 4))
                act = (Y[i, w] - ND[i, w]).sum()
                if act > 0:
                    surge[i] = ((m[i, w] - ND[i, w]).sum() - act) / act
            if j['kind'] == 'festival' and j['year'] in (2023, 2026):
                ly = mc.IDX[cal['Diwali'][j['year'] - 1]] + 364 - j['ci']
                w = slice(max(0, ly - 5), min(H, ly + 6))
                if ly - 5 >= 0 and ly + 6 <= H:
                    phantom[i] = (m[i, w] - Y[i, w]).sum() / Y[i, w].sum()
                    # post-hoc diagnostic: phantom relative to the method's own bias over the whole horizon
                    phantom_rel[i] = (m[i, w].sum() / Y[i, w].sum()) / (m[i].sum() / Y[i].sum()) - 1
        rec.update(peak=peak, surge=surge, phantom=phantom, no_peak=no_peak, phantom_rel=phantom_rel,
                   bias=(m.sum(1) - Y.sum(1)) / Y.sum(1))
        out[name] = rec
    return out


def simulate_inventory(T, D, cap, shelf, price, cost, goodwill=0.0, mode='committed'):
    """Vectorised over jobs. T, D, cap: (n, H). shelf, price, cost: (n,). Returns per-job dict of arrays."""
    n = len(D)
    E = np.array(T, dtype=float)
    for t in range(H - 1, 0, -1):  # pre-build targets that exceed capacity on the previous day
        ex = np.maximum(E[:, t] - cap[:, t], 0)
        E[:, t] -= ex
        E[:, t - 1] += ex
    S = int(shelf.max())
    inv = np.zeros((n, S))
    sold = np.zeros(n); lost = np.zeros(n); waste = np.zeros(n); made = np.zeros(n); hold = np.zeros(n)
    stockout_days = np.zeros(n)
    ages = np.arange(S)[None, :]
    for t in range(H):
        if mode == 'committed':  # plan fixed at the cutoff: no feedback from stock on hand
            prod = np.clip(E[:, t], 0, cap[:, t])
        else:  # daily order-up-to: production tops stock on hand up to the planned level
            prod = np.clip(E[:, t] - inv.sum(1), 0, cap[:, t])
        inv[:, 0] += prod
        made += prod
        need = D[:, t].copy()
        for a in range(S - 1, -1, -1):  # FIFO: oldest first
            s = np.minimum(inv[:, a], need)
            inv[:, a] -= s
            need -= s
            sold += s
        lost += need
        stockout_days += need > 1e-6
        # ageing: stock reaching shelf life is wasted
        expire = (ages == (shelf[:, None] - 1))
        waste += (inv * expire).sum(1)
        inv = np.where(ages >= shelf[:, None] - 1, 0.0, inv)
        inv = np.concatenate([np.zeros((n, 1)), inv[:, :-1]], axis=1)
        hold += inv.sum(1) * HOLD
    salvage = inv.sum(1) * cost * END_SALVAGE
    contribution = price * sold + salvage - cost * made - hold - goodwill * lost
    return dict(contribution=contribution, waste=waste, lost=lost, sold=sold, made=made,
                stockout_rate=stockout_days / H, demand=D.sum(1))


def quantile_at(q, tau):
    """Linear interpolation of the 9 quantiles at level tau (per job)."""
    out = np.empty(q.shape[:2])
    for i in range(len(q)):
        out[i] = np.array([np.interp(tau[i], Q, q[i, t]) for t in range(H)])
    return out


def business(jobs, F, econ=ECON, goodwill_share=0.0, shelf_scale=1.0, price_scale=1.0, policy='critical', mode='committed'):
    D = np.stack([j['truth'] for j in jobs])
    C = np.stack([j['confirmed'] for j in jobs])
    cap = np.stack([j['cap'] for j in jobs]) * CAP_FACTOR
    pr = np.array([econ[j['product']]['price'] * price_scale for j in jobs])
    co = np.array([econ[j['product']]['cost'] for j in jobs])
    sh = np.array([max(1, int(round(econ[j['product']]['shelf'] * shelf_scale))) for j in jobs])
    gw = goodwill_share * pr
    tau = np.clip((pr - co + gw) / (pr + gw), 0.1, 0.9) if policy == 'critical' else np.full(len(jobs), 0.5)
    ref = simulate_inventory(np.maximum(D, 0), D, cap, sh, pr, co, gw, mode)
    out = {'Hindsight reference': ref}
    for name, q in F.items():
        point = q[:, :, 4] * SHOPKEEPER_MARGIN if name == 'Shopkeeper' else quantile_at(q, tau)
        T = np.maximum(point, C)
        r = simulate_inventory(T, D, cap, sh, pr, co, gw, mode)
        r['regret'] = ref['contribution'] - r['contribution']
        out[name] = r
    return out, tau


# ------------------------------------------------------------------ aggregation with seed bootstrap

def seed_index(jobs, mask):
    seeds = sorted({j['seed'] for j, m in zip(jobs, mask) if m})
    return seeds, {s: np.array([i for i, j in enumerate(jobs) if mask[i] and j['seed'] == s]) for s in seeds}


def boot_ratio(num, den, groups, rng):
    seeds = list(groups)
    ns = np.array([np.nansum(num[groups[s]]) for s in seeds])
    ds = np.array([np.nansum(den[groups[s]]) for s in seeds])
    point = ns.sum() / ds.sum()
    idx = rng.integers(0, len(seeds), (B, len(seeds)))
    draws = ns[idx].sum(1) / ds[idx].sum(1)
    return dict(point=float(point), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)))


def boot_mean(x, groups, rng, stat=np.nanmean):
    seeds = list(groups)
    vals = [x[groups[s]] for s in seeds]
    allv = np.concatenate(vals)
    point = stat(allv)
    idx = rng.integers(0, len(seeds), (B, len(seeds)))
    draws = [stat(np.concatenate([vals[k] for k in row])) for row in idx]
    return dict(point=float(point), lo=float(np.nanpercentile(draws, 2.5)), hi=float(np.nanpercentile(draws, 97.5)),
                n=int(np.isfinite(allv).sum()))


def strata(jobs):
    S = {}
    base = np.array([j['scenario'] == 'base' for j in jobs])
    fest = np.array([j['kind'] == 'festival' for j in jobs])
    shift = np.array([j['year'] in (2023, 2026) for j in jobs])
    S['base: festival season (all years)'] = base & fest
    S['base: festival season, +19-day shift years (2023, 2026)'] = base & fest & shift
    S['base: festival season, earlier-Diwali years (2024, 2025)'] = base & fest & ~shift
    S['base: ordinary periods'] = base & ~fest
    paired = np.array([j['scenario'] == 'base' and j['year'] == 2026 and j['cutoff'] in mc.STRESS_CUTOFFS
                       and j['seed'] in mc.STRESS_SEEDS for j in jobs])
    S['stress reference: base 2026, seeds 0-19, Sep 15 / Oct 1'] = paired
    for s in mc.STRESSES:
        S[f'stress: {s}'] = np.array([j['scenario'] == s for j in jobs])
    return S
