"""Mithai Shop Challenge — synthetic retailer simulator, forecasting jobs and calendar correction.

SYNTHETIC, CONTROLLED RESEARCH. No real retailer data is used. numpy-only, deterministic.

Shop: Ahmedabad, three products sold in kg: kaju katli, motichoor ladoo, gulab jamun.
Simulated 2019-01-01 .. 2026-12-31. For every product and day the simulator separates:
  demand     latent customer demand (what forecasts are scored against)
  observed   historical sales = min(demand, historical daily capacity)  (stockouts censor peaks)
  no_diwali  the same demand with the Diwali season effect and Diwali corporate orders removed
             (shared noise) — used only to score festival surge magnitude
  orders     Diwali corporate gift-box orders (kaju katli) with booking and delivery dates
Forecasters only ever receive `observed` before the cutoff (and, in Competition B, the public
festival calendar). Generator parameters are never passed to any forecaster.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

SEED_BASE = 20261108
START = dt.date(2019, 1, 1)
END = dt.date(2026, 12, 31)
N = (END - START).days + 1
DATES = [START + dt.timedelta(days=i) for i in range(N)]
IDX = {d: i for i, d in enumerate(DATES)}
WEEKDAY = np.array([d.weekday() for d in DATES])  # Mon=0
DOY = np.array([d.timetuple().tm_yday for d in DATES])
YEARS_FROM_START = np.arange(N) / 365.25
PRODUCTS = ('kaju-katli', 'motichoor-ladoo', 'gulab-jamun')
NAMES = {'kaju-katli': 'Kaju Katli', 'motichoor-ladoo': 'Motichoor Ladoo', 'gulab-jamun': 'Gulab Jamun'}
HORIZON = 56
HORIZONS = (28, 42, 56)
MAX_CONTEXT = 2048
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FESTIVALS = ('Diwali', 'Dussehra', 'Holi', 'Raksha Bandhan', 'Ganesh Chaturthi')
# drikpanchang (New Delhi) for 2024; all other Diwali dates agree between both sources.
DIWALI_OVERRIDE = {2024: dt.date(2024, 11, 1)}
WEDDING = (((1, 15), (2, 28)), ((4, 20), (5, 31)), ((11, 20), (12, 15)))  # fixed approximations

# Forecasting design
YEARS = (2023, 2024, 2025, 2026)
FESTIVAL_CUTOFFS = ('09-15', '10-01', '10-15')
ORDINARY_CUTOFFS = ('02-15', '06-01')
BASE_SEEDS = tuple(range(30))
STRESS_SEEDS = tuple(range(20))
STRESSES = ('early-gifting', 'weak-festival', 'local-shock')
STRESS_CUTOFFS = ('09-15', '10-01')
SHOCK = (dt.date(2026, 9, 24), dt.date(2026, 10, 14))  # +50 % local demand shock, all products

# Calendar-correction windows (days relative to festival day), estimated from history only
CORRECTION_WINDOWS = {'Diwali': (-35, 5), 'Holi': (-7, 1), 'Raksha Bandhan': (-7, 1), 'Ganesh Chaturthi': (-3, 10)}


# ------------------------------------------------------------------ calendar

def load_calendar(path):
    recs = json.loads(Path(path).read_text())
    cal = {f: {} for f in FESTIVALS}
    for r in recs:
        if r['festival'] in cal and 2018 <= r['year'] <= 2027:
            cal[r['festival']][r['year']] = dt.date.fromisoformat(r['primary_date'])
    cal['Diwali'].update(DIWALI_OVERRIDE)
    return cal


def in_wedding(d):
    md = (d.month, d.day)
    return any(a <= md <= b for a, b in WEDDING)


WEDDING_MASK = np.array([in_wedding(d) for d in DATES])


# ------------------------------------------------------------------ simulator

def draw_params(seed):
    """All randomness for one simulated shop. Ranges are documented assumptions, not estimates."""
    rng = np.random.default_rng([SEED_BASE, seed])
    P = {}
    base_dow = np.array([0.90, 0.85, 0.90, 0.95, 1.05, 1.25, 1.30])
    spec = {  # level kg/day, Diwali peak multiplier range, lead-days range, post-Diwali profile (k=0..3)
        'kaju-katli': dict(level=(8, 14), U=(3.0, 6.0), L=(18, 28), post=(0.55, 0.65, 0.25, 0.10), bulk=0.02),
        'motichoor-ladoo': dict(level=(8, 14), U=(2.0, 4.0), L=(10, 16), post=(0.90, 0.70, 0.45, 0.15), bulk=0.04),
        'gulab-jamun': dict(level=(10, 18), U=(1.4, 2.2), L=(5, 9), post=(0.85, 0.60, 0.50, 0.20), bulk=0.03),
    }
    for p in PRODUCTS:
        s = spec[p]
        dow = base_dow * np.exp(rng.normal(0, 0.03, 7))
        P[p] = dict(
            level=rng.uniform(*s['level']), growth=rng.uniform(-0.04, 0.10), dow=dow / dow.mean(),
            summer_dip=rng.uniform(0.05, 0.15), sigma=rng.uniform(0.15, 0.25),
            U=rng.uniform(*s['U']), L=rng.uniform(*s['L']), shape=rng.uniform(1.5, 2.5), post=np.array(s['post']),
            year_U=np.exp(rng.normal(0, 0.10, 9)), year_L=rng.normal(0, 2.0, 9),  # year-to-year variation 2019..2027
            bulk_rate=s['bulk'], cap_mult=rng.uniform(2.5, 3.5),
            navratri=rng.uniform(0.15, 0.40) if p != 'gulab-jamun' else rng.uniform(0.0, 0.15),
            holi=rng.uniform(0.4, 0.9) if p != 'kaju-katli' else rng.uniform(0.1, 0.3),
            raksha=rng.uniform(0.8, 1.6) if p == 'kaju-katli' else rng.uniform(0.3, 0.8),
            ganesh=rng.uniform(0.5, 1.0) if p == 'motichoor-ladoo' else rng.uniform(0.0, 0.2),
            wedding=rng.uniform(0.05, 0.15) if p != 'kaju-katli' else rng.uniform(0.0, 0.05),
        )
    P['corporate'] = dict(per_year_mean=rng.uniform(3, 8), lead=rng.uniform(20, 35), size=(15, 60))
    # pre-drawn noise / event streams (identical across stress variants)
    P['z'] = rng.normal(0, 1, (len(PRODUCTS), N))
    P['bulk_u'] = rng.uniform(0, 1, (len(PRODUCTS), N))
    P['bulk_size'] = rng.uniform(2, 6, (len(PRODUCTS), N))
    corp = []
    for year in range(2019, 2027):
        for _ in range(rng.poisson(P['corporate']['per_year_mean'])):
            corp.append(dict(year=year, pos=rng.uniform(0, 1), size=rng.uniform(*P['corporate']['size']),
                             book_lead=int(rng.integers(7, 26))))
    P['corp'] = corp
    return P


def _diwali_shape(k, L, a, post):
    """Uplift share (0..1) at day k relative to Diwali: power ramp from -L to the Dhanteras peak, then post profile."""
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(k)
    pre = (k >= -L) & (k <= -1)
    out[pre] = ((k[pre] + L + 1) / (L + 1)) ** a
    for j in range(4):
        out[k == j] = post[j]
    return out


def simulate(seed, cal, stress='base'):
    P = draw_params(seed)
    t = np.arange(N)
    demand = np.zeros((len(PRODUCTS), N))
    no_diwali = np.zeros_like(demand)
    cap = np.zeros_like(demand)
    orders = []
    for pi, p in enumerate(PRODUCTS):
        q = P[p]
        trend = (1 + q['growth']) ** YEARS_FROM_START
        season = 1 - q['summer_dip'] * np.exp(-((DOY - 150) / 35.0) ** 2)
        base = q['level'] * trend * q['dow'][WEEKDAY] * season
        other = np.ones(N)
        diwali = np.zeros(N)
        for year in range(2019, 2027):
            yi = year - 2019
            d = IDX.get(cal['Diwali'][year])
            U = 1 + (q['U'] - 1) * q['year_U'][yi]
            L = max(3.0, q['L'] + q['year_L'][yi])
            if year == 2026 and stress == 'early-gifting' and p == 'kaju-katli':
                L *= 1.6
            if year == 2026 and stress == 'weak-festival':
                U = 1 + (U - 1) * 0.5
            diwali += (U - 1) * _diwali_shape(t - d, L, q['shape'], q['post'])
            ds = IDX[cal['Dussehra'][year]]
            other += q['navratri'] * ((t >= ds - 10) & (t <= ds)).astype(float)
            h = IDX[cal['Holi'][year]]
            other += q['holi'] * ((t >= h - 3) & (t <= h)).astype(float)
            rb = IDX[cal['Raksha Bandhan'][year]]
            other += q['raksha'] * np.clip((t - (rb - 4)) / 4.0, 0, 1) * (t <= rb)
            g = IDX[cal['Ganesh Chaturthi'][year]]
            other += q['ganesh'] * (((t >= g) & (t <= g + 10)) * 0.5 + (t == g) * 1.0)
        other += q['wedding'] * WEDDING_MASK
        noise = np.exp(q['sigma'] * P['z'][pi] - q['sigma'] ** 2 / 2)
        rate = q['bulk_rate'] * np.where(WEDDING_MASK, 3.0, 1.0)
        bulk = (P['bulk_u'][pi] < rate) * P['bulk_size'][pi] * q['level'] * trend
        walk_in = base * other * noise
        no_diwali[pi] = walk_in + bulk
        demand[pi] = walk_in * (1 + diwali) + bulk
        if stress == 'local-shock':
            m = (t >= IDX[SHOCK[0]]) & (t <= IDX[SHOCK[1]])
            demand[pi] = np.where(m, demand[pi] * 1.5, demand[pi])
            no_diwali[pi] = np.where(m, no_diwali[pi] * 1.5, no_diwali[pi])
        cap[pi] = q['cap_mult'] * q['level'] * trend
    # Diwali corporate gift-box orders (kaju katli): delivered L_c..3 days before Diwali, booked 7-25 days ahead
    lead = P['corporate']['lead']
    corp = np.zeros(N)
    for n, o in enumerate(P['corp']):
        lc = lead + (20 if (o['year'] == 2026 and stress == 'early-gifting') else 0)
        if o['year'] == 2026 and stress == 'weak-festival' and n % 2:
            continue
        d = IDX[cal['Diwali'][o['year']]]
        deliver = d - int(round(3 + o['pos'] * (lc - 3)))
        kg = o['size'] * P['kaju-katli']['level'] / 11.0
        demand[0, deliver] += kg
        corp[deliver] += kg
        orders.append(dict(deliver=deliver, book=deliver - o['book_lead'], kg=kg))
    # Walk-in and bulk sales are censored at historical capacity; pre-booked corporate orders were fulfilled.
    walk = demand.copy()
    walk[0] -= corp
    observed = np.round(np.minimum(walk, cap), 2)  # sales recorded to 10 g
    observed[0] = np.round(observed[0] + corp, 2)
    return dict(demand=demand, observed=observed, no_diwali=no_diwali, cap=cap, orders=orders, params=P)


# ------------------------------------------------------------------ jobs

def cutoff_index(year, mmdd):
    return IDX[dt.date.fromisoformat(f'{year}-{mmdd}')]


def job_specs():
    specs = []
    for seed in BASE_SEEDS:
        for year in YEARS:
            for c in FESTIVAL_CUTOFFS + ORDINARY_CUTOFFS:
                specs.append(('base', seed, year, c))
    for stress in STRESSES:
        for seed in STRESS_SEEDS:
            for c in STRESS_CUTOFFS:
                specs.append((stress, seed, 2026, c))
    return specs


def diwali_in_horizon(cal, year, ci):
    d = IDX[cal['Diwali'][year]]
    return ci <= d < ci + HORIZON


# ------------------------------------------------------------------ calendar correction (history only)

def festival_offsets(cal, idx_lo, idx_hi):
    """For each day in [idx_lo, idx_hi): (festival, k) of the window it falls in, else None. Diwali has priority."""
    out = [None] * (idx_hi - idx_lo)
    for f in ('Ganesh Chaturthi', 'Raksha Bandhan', 'Holi', 'Diwali'):
        lo, hi = CORRECTION_WINDOWS[f]
        for year, day in cal[f].items():
            if day not in IDX:
                continue
            d = IDX[day]
            for k in range(lo, hi + 1):
                i = d + k
                if idx_lo <= i < idx_hi:
                    out[i - idx_lo] = (f, k)
    return out


def correction_profile(hist, h0, cal):
    """Estimate multiplicative uplift profiles r_f(k) from observed history only.
    hist covers absolute day indices [h0, h0+len(hist)). Baseline = local level of non-window days
    (±56 days) × weekday factor. Returns (profiles, offsets, baseline)."""
    n = len(hist)
    offs = festival_offsets(cal, h0, h0 + n)
    clean = np.array([o is None for o in offs])
    x = np.where(clean, hist, 0.0)
    w = clean.astype(float)
    ker = np.ones(113)
    lvl = np.convolve(x, ker, 'same') / np.maximum(np.convolve(w, ker, 'same'), 1)
    wd = WEEKDAY[h0:h0 + n]
    ratio = np.where(clean & (lvl > 0), hist / np.maximum(lvl, 1e-9), np.nan)
    dowf = np.array([np.nanmedian(ratio[wd == j]) for j in range(7)])
    dowf = dowf / np.nanmean(dowf)
    base = lvl * dowf[wd]
    prof = {}
    for f, (lo, hi) in CORRECTION_WINDOWS.items():
        rows = []
        for year, day in cal[f].items():
            d = IDX.get(day)
            if d is None or d + lo < h0 or d + hi >= h0 + n:
                continue  # need the whole window observed
            ks = np.arange(lo, hi + 1)
            rows.append(hist[d + ks - h0] / np.maximum(base[d + ks - h0], 1e-9))
        if rows:
            prof[f] = np.median(np.array(rows), axis=0)
    return prof, offs, base


def apply_profile(values, offs, prof, invert):
    out = np.array(values, dtype=float)
    for i, o in enumerate(offs):
        if o is None or o[0] not in prof:
            continue
        r = max(prof[o[0]][o[1] - CORRECTION_WINDOWS[o[0]][0]], 1e-3)
        out[i] = out[i] / r if invert else out[i] * r
    return out


def build_jobs(cal, with_truth=True):
    """Every forecasting job. history = observed sales strictly before the cutoff (≤2048 days).
    history_clean = same history with history-estimated festival uplift divided out (Competition B input)."""
    jobs = []
    cache = {}
    for scen, seed, year, c in job_specs():
        key = (scen, seed)
        if key not in cache:
            cache = {key: simulate(seed, cal, scen)}
        sim = cache[key]
        ci = cutoff_index(year, c)
        h0 = max(0, ci - MAX_CONTEXT)
        f_offs = festival_offsets(cal, ci, ci + HORIZON)
        for pi, p in enumerate(PRODUCTS):
            hist = sim['observed'][pi, h0:ci]
            prof, offs, _ = correction_profile(hist, h0, cal)
            clean = np.round(apply_profile(hist, offs, prof, invert=True), 2)
            j = dict(key=f'{scen}|{seed}|{year}|{c}|{p}', scenario=scen, seed=seed, year=year, cutoff=c, product=p,
                     ci=ci, h0=h0, history=hist, history_clean=clean, profile={f: v.tolist() for f, v in prof.items()},
                     future_offsets=f_offs, kind='festival' if c in FESTIVAL_CUTOFFS + STRESS_CUTOFFS else 'ordinary',
                     diwali_in_horizon=diwali_in_horizon(cal, year, ci))
            if with_truth:
                j['truth'] = sim['demand'][pi, ci:ci + HORIZON]
                j['no_diwali'] = sim['no_diwali'][pi, ci:ci + HORIZON]
                j['cap'] = sim['cap'][pi, ci:ci + HORIZON]
                j['confirmed'] = np.zeros(HORIZON)
                if p == 'kaju-katli':
                    for o in sim['orders']:
                        if o['book'] < ci <= o['deliver'] < ci + HORIZON:
                            j['confirmed'][o['deliver'] - ci] += o['kg']
            jobs.append(j)
    return jobs


def histories_sha256(jobs):
    h = hashlib.sha256()
    for j in jobs:
        h.update(j['key'].encode())
        h.update(np.asarray(j['history'], dtype='<f8').tobytes())
        h.update(np.asarray(j['history_clean'], dtype='<f8').tobytes())
    return h.hexdigest()
