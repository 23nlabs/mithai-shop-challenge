"""Robustness v2 evaluation. Reuses the ORIGINAL metric code (src/evaluate.py, imported unchanged) and adds:

1. Rolling weekly replanning simulator (vectorised over jobs):
   * re-forecast every 7 days (origins day 0, 7, ..., 49 of the 56-day plan), each with its own
     information cutoff; every method uses the same origins
   * ingredient procurement committed with a 7-day supplier lead time: at day 0 the shop orders for
     weeks 0 and 1 (week 0 assumed sourced at the cutoff); at day 7k it orders for week k+1, netting the
     projected unused ingredient stock; order = sum of that week's planned targets (capacity-limited)
   * ingredients are 70% of unit cost, paid at ordering, and expire 28 / 14 / 7 days after arrival
     (kaju katli / motichoor ladoo / gulab jamun); conversion cost (30%) is paid on production
   * at each weekly origin, MRP-style netting: on-hand finished stock is allocated FIFO (shelf-life aware)
     to upcoming targets; the net requirement is capacity-scheduled (pre-build onto earlier days); that
     week's daily production follows the schedule, limited by ingredients actually received
   * finished goods FIFO, shelf life 10 / 5 / 4 days, holding 2 INR/kg/day; salvage at day 56: 50% of cost
     (finished) and 50% of ingredient cost (unused ingredients)
   * targets: max(Q_tau(t), confirmed orders known at that origin); original shopkeeper: 1.10 x point
   * hindsight reference: identical mechanics with every forecast equal to realised demand
2. Independent re-implementation of the original committed-plan simulator with explicit constraint
   assertions (capacity, non-negative stock, shelf life), compared against src/evaluate.py outputs.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
STUDY = V2.parents[1]
sys.path.insert(0, str(STUDY / 'src'))
sys.path.insert(0, str(V2 / 'src'))
import evaluate as ev  # noqa: E402
import mithai_core as mc  # noqa: E402

H = mc.HORIZON
LT = 7
ING_SHARE = 0.7
ING_SHELF = {'kaju-katli': 28, 'motichoor-ladoo': 14, 'gulab-jamun': 7}

METHODS_V2 = dict(ev.METHODS)
METHODS_V2['Shopkeeper (calibrated, τ policy)'] = ('B', 'shopkeeper-calibrated', 'history', False)
METHODS_V2['Calendar regression (censoring-aware)'] = ('B', 'calreg-censoring-aware', 'history', False)


def process(jobs, raw):
    """raw: {(stem, field): array}. Same handling as the original load_forecasts."""
    F = {}
    for name, (_, stem, field, corrected) in METHODS_V2.items():
        if (stem, field) not in raw:
            continue
        q = np.where(np.isfinite(raw[(stem, field)]), raw[(stem, field)], 0.0).astype(float)
        if corrected:
            for i, j in enumerate(jobs):
                for t, o in enumerate(j['future_offsets']):
                    if o is None or o[0] not in j['profile']:
                        continue
                    q[i, t] *= max(j['profile'][o[0]][o[1] - mc.CORRECTION_WINDOWS[o[0]][0]], 1e-3)
        F[name] = np.sort(np.clip(q, 0, None), axis=2)
    return F


def load_npz(path, keys):
    z = np.load(path)
    if not np.array_equal(z['keys'], np.asarray(keys)):
        raise RuntimeError(f'KEY_ORDER_MISMATCH {path}')
    return z['q'].astype(float)


def targets(name, q, conf, tau):
    if name == 'Shopkeeper':
        point = q[:, :, 4] * ev.SHOPKEEPER_MARGIN
    else:
        point = ev.quantile_at(q, tau)
    return np.maximum(point, conf)


def spill(E, cap, start=0):
    E = np.array(E, dtype=float)
    for t in range(H - 1, start, -1):
        ex = np.maximum(E[:, t] - cap[:, t], 0)
        E[:, t] -= ex
        E[:, t - 1] += ex
    return E


def net_requirements(T, inv, shelf, t0):
    """Allocate on-hand finished stock (by age) FIFO to targets on days >= t0, respecting shelf life.
    Returns targets net of usable stock (MRP-style netting)."""
    n, S = inv.shape
    stock = inv.copy()
    net = np.array(T, dtype=float)
    for d in range(t0, min(H, t0 + S)):
        for a in range(S - 1, -1, -1):
            fresh = (a + (d - t0)) <= (shelf - 1)
            take = np.where(fresh, np.minimum(stock[:, a], net[:, d]), 0.0)
            stock[:, a] -= take
            net[:, d] -= take
    return net


def rolling_inventory(T_by_origin, D, cap, shelf, ing_shelf, price, cost, check=False):
    """T_by_origin: (8, n, H) daily targets in plan-day coordinates from each weekly origin.
    At each weekly origin: net requirements = targets - usable on-hand stock (FIFO, shelf-life aware),
    capacity-scheduled (pre-build onto earlier days); that week's production follows the schedule,
    limited by ingredients received. Ingredient orders use the same schedule."""
    n = D.shape[0]
    S = int(shelf.max())
    inv = np.zeros((n, S))
    lots = np.zeros((n, 9))  # ingredient lot for week w arrives at day 7w
    sold, lost, waste, made, hold, ing_cost, conv_cost, ing_waste, ordered = (np.zeros(n) for _ in range(9))
    stockout_days = np.zeros(n)
    ages = np.arange(S)[None, :]
    sched = None

    def order(week, sched_k, t_now):
        nonlocal ing_cost, ordered
        if week > 7:
            return
        lo = 7 * week
        need = sched_k[:, lo:lo + 7].sum(1)
        usable = np.zeros(n)
        for w in range(week):
            alive = (7 * w + ing_shelf) > lo
            usable += np.where(alive, lots[:, w], 0)
        planned = sched_k[:, t_now:lo].sum(1) if lo > t_now else 0
        q = np.maximum(need - np.maximum(usable - planned, 0), 0)
        lots[:, week] += q
        ordered += q
        ing_cost += q * cost * ING_SHARE

    for t in range(H):
        k = t // 7
        if t % 7 == 0:
            sched = spill(net_requirements(T_by_origin[k], inv, shelf, t), cap, start=t)
            if t == 0:
                order(0, sched, 0)
                order(1, sched, 0)
            else:
                order(k + 1, sched, t)
        for w in range(min(k + 1, 8)):
            exp = (t == 7 * w + ing_shelf)
            ing_waste += np.where(exp, lots[:, w], 0)
            lots[:, w] = np.where(exp, 0, lots[:, w])
        avail = sum(np.where(7 * w <= t, lots[:, w], 0) for w in range(8))
        prod = np.minimum(np.minimum(np.maximum(sched[:, t], 0), cap[:, t]), avail)
        if check:
            assert (prod <= cap[:, t] + 1e-9).all() and (prod <= avail + 1e-9).all() and (prod >= 0).all()
        rem = prod.copy()
        for w in range(8):
            use = np.where(7 * w <= t, np.minimum(lots[:, w], rem), 0)
            lots[:, w] -= use
            rem -= use
        inv[:, 0] += prod
        made += prod
        conv_cost += prod * cost * (1 - ING_SHARE)
        need = D[:, t].copy()
        for a in range(S - 1, -1, -1):
            s_ = np.minimum(inv[:, a], need)
            inv[:, a] -= s_
            need -= s_
            sold += s_
        lost += need
        stockout_days += need > 1e-6
        expire = ages == (shelf[:, None] - 1)
        waste += (inv * expire).sum(1)
        inv = np.where(ages >= shelf[:, None] - 1, 0.0, inv)
        inv = np.concatenate([np.zeros((n, 1)), inv[:, :-1]], axis=1)
        hold += inv.sum(1) * ev.HOLD
        if check:
            assert (inv >= -1e-9).all() and (lots >= -1e-9).all()
    left_ing = lots.sum(1)
    salvage = inv.sum(1) * cost * ev.END_SALVAGE + left_ing * cost * ING_SHARE * ev.END_SALVAGE
    contribution = price * sold + salvage - ing_cost - conv_cost - hold
    return dict(contribution=contribution, waste=waste, ingredient_waste=ing_waste, lost=lost, sold=sold, made=made,
                ordered=ordered, stockout_rate=stockout_days / H, demand=D.sum(1))


def committed_checked(T, D, cap, shelf, price, cost):
    """Independent re-implementation of ev.simulate_inventory(mode='committed') with assertions."""
    n = len(D)
    E = spill(T, cap)
    S = int(shelf.max())
    inv = np.zeros((n, S))
    sold, lost, waste, made, hold = (np.zeros(n) for _ in range(5))
    for t in range(H):
        prod = np.clip(E[:, t], 0, cap[:, t])
        assert (prod <= cap[:, t] + 1e-9).all()
        inv[:, 0] += prod
        made += prod
        need = D[:, t].copy()
        for a in range(S - 1, -1, -1):
            s = np.minimum(inv[:, a], need)
            inv[:, a] -= s
            need -= s
            sold += s
        assert (need >= -1e-9).all() and (inv >= -1e-9).all()
        lost += need
        for i in range(n):  # nothing older than shelf life may remain sellable
            assert inv[i, shelf[i]:].sum() < 1e-9
        waste += (inv * (np.arange(S)[None, :] == shelf[:, None] - 1)).sum(1)
        inv = np.where(np.arange(S)[None, :] >= shelf[:, None] - 1, 0.0, inv)
        inv = np.concatenate([np.zeros((n, 1)), inv[:, :-1]], axis=1)
        hold += inv.sum(1) * ev.HOLD
    contribution = price * sold + inv.sum(1) * cost * ev.END_SALVAGE - cost * made - hold
    return dict(contribution=contribution, waste=waste, lost=lost, made=made)


def econ_arrays(jobs):
    pr = np.array([ev.ECON[j['product']]['price'] for j in jobs])
    co = np.array([ev.ECON[j['product']]['cost'] for j in jobs])
    sh = np.array([ev.ECON[j['product']]['shelf'] for j in jobs])
    ish = np.array([ING_SHELF[j['product']] for j in jobs])
    tau = np.clip((pr - co) / pr, 0.1, 0.9)
    return pr, co, sh, ish, tau


def summarise(jobs, F, acc, bus, mask, rng):
    seeds, groups = ev.seed_index(jobs, mask)
    out = dict(n_jobs=int(mask.sum()), n_seeds=len(seeds), methods={})
    for name in list(F) + ['Hindsight reference']:
        r = {}
        if name in acc:
            a = acc[name]
            r['wape_56'] = ev.boot_ratio(a['abs_err'][56], a['actual'][56], groups, rng)
            r['bias'] = ev.boot_ratio(a['bias'] * a['actual'][56], a['actual'][56], groups, rng)
            if 'pinball' in a:
                r['coverage80'] = ev.boot_ratio(a['covered'].astype(float), np.full(len(jobs), float(H)), groups, rng)
                r['wql'] = ev.boot_ratio(a['pinball'], a['actual'][56], groups, rng)
            for k in ('peak', 'surge', 'phantom_rel'):
                if np.isfinite(a[k][mask]).sum():
                    r[f'{k}_median'] = ev.boot_mean(a[k], groups, rng, stat=np.nanmedian)
            if np.isfinite(a['no_peak'][mask]).sum():
                r['distinct_peak_share'] = ev.boot_mean(1 - a['no_peak'], groups, rng)
                r['peak_early_share'] = ev.boot_mean((a['peak'] <= -7).astype(float) + 0 * a['peak'], groups, rng)
        b = bus[name]
        r['regret_per_job'] = ev.boot_mean(b['regret'], groups, rng) if 'regret' in b else None
        r['contribution_per_job'] = ev.boot_mean(b['contribution'], groups, rng)
        r['waste_share'] = ev.boot_ratio(b['waste'], b['made'], groups, rng)
        r['lost_share'] = ev.boot_ratio(b['lost'], b['demand'], groups, rng)
        if 'ingredient_waste' in b:
            r['ingredient_waste_share'] = ev.boot_ratio(b['ingredient_waste'], b['ordered'], groups, rng)
        out['methods'][name] = r
    return out


def paired_ratio_diff(numA, denA, numB, denB, groups, rng):
    seeds = list(groups)
    a = np.array([[numA[groups[s]].sum(), denA[groups[s]].sum(), numB[groups[s]].sum(), denB[groups[s]].sum()] for s in seeds])
    pt = a[:, 0].sum() / a[:, 1].sum() - a[:, 2].sum() / a[:, 3].sum()
    idx = rng.integers(0, len(seeds), (ev.B, len(seeds)))
    s = a[idx].sum(1)
    d = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    return dict(point=float(pt), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
