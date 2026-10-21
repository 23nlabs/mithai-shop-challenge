"""Mithai Shop Challenge — robustness v2 (separately versioned; the original experiment is untouched).

Adds, on top of mithai_core (imported, never modified):
  * calendar-dependence scenarios (all years 2019-2026, extra randomness from a SEPARATE stream so the
    base draws are identical):
      R1-shape    festival response shape varies strongly year to year: Diwali lead time x exp(N(0,0.35)),
                  ramp exponent U(0.8,3.5), uplift x exp(N(0,0.35)), whole curve shifted -6..+2 days;
                  other festivals' uplift x exp(N(0,0.35)) per year
      R2-shocks   festival-independent shop-level demand shocks: Poisson(3)/year, 3-21 days,
                  multiplier exp(U(-0.6, 0.6)), placed uniformly in the year (can hit festival seasons)
      R3-both     R1 + R2
  * stockout flags (a real shop knows the days it sold out), used only by the censoring-aware baseline
  * rolling weekly origins (cutoff + 0, 7, ..., 49 days) for weekly replanning
"""
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1] / 'src'))
import mithai_core as mc  # noqa: E402

ROBUST = ('R1-shape', 'R2-shocks', 'R3-both')
ROBUST_SEEDS = tuple(range(20))
ROLL_SEEDS = tuple(range(20))
ROLL_OFFSETS = tuple(range(0, 56, 7))


def robust_draws(seed):
    rng = np.random.default_rng([mc.SEED_BASE, seed, 777])
    yrs = range(2019, 2027)
    shape = {y: dict(L=float(np.exp(rng.normal(0, 0.35))), a=float(rng.uniform(0.8, 3.5)),
                     U=float(np.exp(rng.normal(0, 0.35))), shift=int(rng.integers(-6, 3)),
                     other=float(np.exp(rng.normal(0, 0.35)))) for y in yrs}
    shocks = []
    for y in yrs:
        for _ in range(rng.poisson(3)):
            start = mc.IDX[dt.date(y, 1, 1)] + int(rng.integers(0, 365))
            shocks.append((start, int(rng.integers(3, 22)), float(np.exp(rng.uniform(-0.6, 0.6)))))
    return shape, shocks


def simulate(seed, cal, scenario='base'):
    """Identical to mithai_core.simulate for scenario='base' (tested); see module docstring otherwise."""
    if scenario in mc.STRESSES:
        sim = mc.simulate(seed, cal, scenario)
        return _flags(sim)
    P = mc.draw_params(seed)
    shape, shocks = robust_draws(seed)
    vary = scenario in ('R1-shape', 'R3-both')
    shock_on = scenario in ('R2-shocks', 'R3-both')
    t = np.arange(mc.N)
    shock_mult = np.ones(mc.N)
    if shock_on:
        for start, dur, m in shocks:
            shock_mult[start:start + dur] *= m
    demand = np.zeros((len(mc.PRODUCTS), mc.N))
    no_diwali = np.zeros_like(demand)
    cap = np.zeros_like(demand)
    walk_all = np.zeros_like(demand)
    for pi, p in enumerate(mc.PRODUCTS):
        q = P[p]
        trend = (1 + q['growth']) ** mc.YEARS_FROM_START
        season = 1 - q['summer_dip'] * np.exp(-((mc.DOY - 150) / 35.0) ** 2)
        base = q['level'] * trend * q['dow'][mc.WEEKDAY] * season
        other = np.ones(mc.N)
        diwali = np.zeros(mc.N)
        for year in range(2019, 2027):
            yi = year - 2019
            sh = shape[year]
            d = mc.IDX[cal['Diwali'][year]] + (sh['shift'] if vary else 0)
            U = 1 + (q['U'] - 1) * q['year_U'][yi] * (sh['U'] if vary else 1)
            L = max(3.0, (q['L'] + q['year_L'][yi]) * (sh['L'] if vary else 1))
            a = sh['a'] if vary else q['shape']
            diwali += (U - 1) * mc._diwali_shape(t - d, L, a, q['post'])
            o = sh['other'] if vary else 1.0
            ds = mc.IDX[cal['Dussehra'][year]]
            other += o * q['navratri'] * ((t >= ds - 10) & (t <= ds)).astype(float)
            h = mc.IDX[cal['Holi'][year]]
            other += o * q['holi'] * ((t >= h - 3) & (t <= h)).astype(float)
            rb = mc.IDX[cal['Raksha Bandhan'][year]]
            other += o * q['raksha'] * np.clip((t - (rb - 4)) / 4.0, 0, 1) * (t <= rb)
            g = mc.IDX[cal['Ganesh Chaturthi'][year]]
            other += o * q['ganesh'] * (((t >= g) & (t <= g + 10)) * 0.5 + (t == g) * 1.0)
        other += q['wedding'] * mc.WEDDING_MASK
        noise = np.exp(q['sigma'] * P['z'][pi] - q['sigma'] ** 2 / 2)
        rate = q['bulk_rate'] * np.where(mc.WEDDING_MASK, 3.0, 1.0)
        bulk = (P['bulk_u'][pi] < rate) * P['bulk_size'][pi] * q['level'] * trend
        walk_in = base * other * noise * shock_mult
        no_diwali[pi] = walk_in + bulk
        demand[pi] = walk_in * (1 + diwali) + bulk
        cap[pi] = q['cap_mult'] * q['level'] * trend
    lead = P['corporate']['lead']
    corp = np.zeros(mc.N)
    orders = []
    for o in P['corp']:
        sh = shape[o['year']]
        lc = lead * (sh['L'] if vary else 1)
        d = mc.IDX[cal['Diwali'][o['year']]] + (sh['shift'] if vary else 0)
        deliver = d - int(round(3 + o['pos'] * (lc - 3)))
        kg = o['size'] * P['kaju-katli']['level'] / 11.0
        demand[0, deliver] += kg
        corp[deliver] += kg
        orders.append(dict(deliver=deliver, book=deliver - o['book_lead'], kg=kg))
    walk = demand.copy()
    walk[0] -= corp
    observed = np.round(np.minimum(walk, cap), 2)
    observed[0] = np.round(observed[0] + corp, 2)
    return dict(demand=demand, observed=observed, no_diwali=no_diwali, cap=cap, orders=orders,
                stockout=walk > cap)


def _flags(sim):
    walk = sim['demand'].copy()
    corp = np.zeros(mc.N)
    for o in sim['orders']:
        corp[o['deliver']] += o['kg']
    walk[0] -= corp
    sim['stockout'] = walk > sim['cap']
    return sim


def specs():
    """(scenario, seed, year, cutoff, offset). Separate families: robust scenarios and rolling origins."""
    out = []
    for scen in ROBUST:
        for seed in ROBUST_SEEDS:
            for year in mc.YEARS:
                for c in mc.FESTIVAL_CUTOFFS + mc.ORDINARY_CUTOFFS:
                    out.append((scen, seed, year, c, 0))
    for seed in ROLL_SEEDS:
        for year in mc.YEARS:
            for c in mc.FESTIVAL_CUTOFFS:
                for off in ROLL_OFFSETS:
                    out.append(('base', seed, year, c, off))
    return out


def build_jobs(cal, spec_list=None, with_truth=True):
    jobs = []
    cache = {}
    for scen, seed, year, c, off in (spec_list or specs()):
        if (scen, seed) not in cache:
            cache = {(scen, seed): simulate(seed, cal, scen)}
        sim = cache[(scen, seed)]
        c0 = mc.cutoff_index(year, c)
        ci = c0 + off
        h0 = max(0, ci - mc.MAX_CONTEXT)
        f_offs = mc.festival_offsets(cal, ci, ci + mc.HORIZON)
        for pi, p in enumerate(mc.PRODUCTS):
            hist = sim['observed'][pi, h0:ci]
            prof, offs, _ = mc.correction_profile(hist, h0, cal)
            clean = np.round(mc.apply_profile(hist, offs, prof, invert=True), 2)
            j = dict(key=f'{scen}|{seed}|{year}|{c}|+{off}|{p}', scenario=scen, seed=seed, year=year, cutoff=c,
                     offset=off, product=p, ci=ci, c0=c0, h0=h0, history=hist, history_clean=clean,
                     stockout=sim['stockout'][pi, h0:ci], profile={f: v.tolist() for f, v in prof.items()},
                     future_offsets=f_offs, kind='festival' if c in mc.FESTIVAL_CUTOFFS else 'ordinary',
                     diwali_in_horizon=mc.diwali_in_horizon(cal, year, ci))
            if with_truth:
                end = min(mc.N, c0 + mc.HORIZON)
                j['truth'] = sim['demand'][pi, c0:end]  # always the ORIGINAL plan window
                j['no_diwali'] = sim['no_diwali'][pi, c0:end]
                j['cap'] = sim['cap'][pi, c0:end]
                j['confirmed'] = np.zeros(mc.HORIZON)
                if p == 'kaju-katli':
                    for o in sim['orders']:
                        if o['book'] < ci and c0 <= o['deliver'] < c0 + mc.HORIZON:
                            j['confirmed'][o['deliver'] - c0] += o['kg']
            jobs.append(j)
    return jobs


def histories_sha256(jobs):
    h = hashlib.sha256()
    for j in jobs:
        h.update(j['key'].encode())
        h.update(np.asarray(j['history'], dtype='<f8').tobytes())
        h.update(np.asarray(j['history_clean'], dtype='<f8').tobytes())
    return h.hexdigest()
