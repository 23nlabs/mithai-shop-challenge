"""Mithai Shop Challenge — v3 final analysis (post-hoc relative to v1; requested in the final review pass).

SYNTHETIC retail demand — not real shop transaction records.

A. Yug-1 across five sampling seeds (67067, 1, 2, 3, 4) for every experiment family:
   original jobs (base + stress), v2 calendar-dependence scenarios (R1-R3) and weekly rolling replanning.
B. Training-only calibration applied identically to every method (incl. both shopkeepers):
   none     : target = Q_tau(t)              (original shopkeeper: 1.10 x point, the v1 rule)
   bias     : target = b x Q_tau(t),  b = sum(observed) / sum(median forecast) over the same shop/product's
              earlier forecast windows that ENDED BEFORE the cutoff (same-kind windows preferred)
   bias+cal : target = k x Q_tau(t),  k = empirical tau-quantile of observed / Q_tau over those windows
              (point-only original shopkeeper uses his point forecast, without the 1.10 margin)
   Learned from OBSERVED (censored) sales only — information a real shop has. No window -> unchanged.
C. Paired regret differences among calendar-informed methods, hierarchical bootstrap: resample simulated
   shops (clusters) and, for Yug-1 variants, resample sampling seeds within each draw. A difference whose 95%
   interval contains 0 is reported as inconclusive.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[1]
STUDY = V3.parents[1]
V2 = STUDY / 'robustness/v2'
sys.path.insert(0, str(STUDY / 'src'))
sys.path.insert(0, str(V2 / 'src'))
import eval_v2 as e2  # noqa: E402
import evaluate as ev  # noqa: E402
import mithai_core as mc  # noqa: E402
import robust_core as rc  # noqa: E402
import run_v2 as r2  # noqa: E402

H = mc.HORIZON
Q = np.asarray(mc.QUANTILES)
SEEDS = ('67067', '1', '2', '3', '4')
YUG = ('Yug-1', 'Yug-1 + calendar')
B = 2000
CAL_B = ('Calendar regression', 'Calendar regression (censoring-aware)', 'ETS + calendar', 'Yug-1 + calendar',
         'TimesFM + calendar', 'Chronos + calendar', 'Shopkeeper (calibrated, τ policy)', 'Shopkeeper')
V3_FM = V3 / 'runs/kaggle-v3-01/output/v3'


def qat(q, tau):
    """Vectorised linear interpolation of the 9 quantiles at per-job level tau (clamped to 0.1..0.9)."""
    pos = np.interp(tau, Q, np.arange(9))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, 8)
    w = (pos - lo)[:, None]
    idx = np.arange(len(q))
    return q[idx, :, lo] * (1 - w) + q[idx, :, hi] * w


def base_point(name, q, tau, variant):
    if name == 'Shopkeeper':
        return q[:, :, 4] * (ev.SHOPKEEPER_MARGIN if variant == 'none' else 1.0)
    return qat(q, tau)


# ------------------------------------------------------------------ calibration (training-only)

def history_family(job):
    """Which simulated history a job's past windows come from."""
    scen = job['scenario']
    return ('base' if scen in mc.STRESSES else scen, job['seed'], job['product'])


def calibration_factors(jobs, fam_jobs, fam_q, observed, tau, name):
    """Per job: (b, k) from the same family's past windows ending before the job's cutoff.
    fam_jobs: candidate past jobs (dicts with ci, kind, product, seed, family); fam_q: their forecasts."""
    by_fam = defaultdict(list)
    for i, j in enumerate(fam_jobs):
        by_fam[history_family(j)].append(i)
    b = np.ones(len(jobs))
    k = np.ones(len(jobs))
    pt_all = base_point(name, fam_q, tau_of(fam_jobs), 'cal')
    med_all = fam_q[:, :, 4]
    for n, j in enumerate(jobs):
        cut = j.get('c0', j['ci'])
        cands = [i for i in by_fam[history_family(j)] if fam_jobs[i]['ci'] + H <= cut]
        same = [i for i in cands if fam_jobs[i]['kind'] == j['kind']]
        use = same or cands
        if not use:
            continue
        obs = np.concatenate([observed[history_family(fam_jobs[i])][fam_jobs[i]['ci']:fam_jobs[i]['ci'] + H] for i in use])
        med = np.concatenate([med_all[i] for i in use])
        pt = np.concatenate([pt_all[i] for i in use])
        b[n] = obs.sum() / max(med.sum(), 1e-9)
        ok = pt > 0.1
        if ok.sum() >= 20:
            k[n] = np.quantile(obs[ok] / pt[ok], tau[n])
    return b, k


def tau_of(jobs):
    pr = np.array([ev.ECON[j['product']]['price'] for j in jobs])
    co = np.array([ev.ECON[j['product']]['cost'] for j in jobs])
    return np.clip((pr - co) / pr, 0.1, 0.9)


def targets(name, q, jobs, variant, bk):
    tau = tau_of(jobs)
    pt = base_point(name, q, tau, variant)
    if variant == 'bias':
        pt = pt * bk[0][:, None]
    elif variant == 'bias+cal':
        pt = pt * bk[1][:, None]
    return np.maximum(pt, np.stack([j['confirmed'] for j in jobs]))


def committed_regret(T, jobs):
    D = np.stack([j['truth'] for j in jobs])
    cap = np.stack([j['cap'] for j in jobs]) * ev.CAP_FACTOR
    pr, co, sh, ish, tau = e2.econ_arrays(jobs)
    ref = ev.simulate_inventory(D, D, cap, sh, pr, co)
    r = ev.simulate_inventory(T, D, cap, sh, pr, co)
    return ref['contribution'] - r['contribution'], r


# ------------------------------------------------------------------ bootstrap helpers

def groups_of(jobs, mask):
    seeds = sorted({j['seed'] for j, m in zip(jobs, mask) if m})
    return [np.array([i for i, (j, mm) in enumerate(zip(jobs, mask)) if mm and j['seed'] == s]) for s in seeds]


def hboot_mean(R, groups, rng):
    """R: (n_jobs, n_model_seeds). Resample shops, and model seeds within each draw."""
    R = np.atleast_2d(R.T).T
    S = R.shape[1]
    point = float(np.concatenate([R[g] for g in groups]).mean())
    draws = []
    for _ in range(B):
        gs = [groups[i] for i in RNG.integers(0, len(groups), len(groups))]
        ss = RNG.integers(0, S, S)
        draws.append(np.concatenate([R[g][:, ss] for g in gs]).mean())
    return dict(point=point, lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)), model_seeds=S,
                seed_range=[float(R[np.concatenate(groups)][:, s].mean()) for s in range(S)] if S > 1 else None)


def hboot_diff(RA, RB, groups):
    RA, RB = np.atleast_2d(RA.T).T, np.atleast_2d(RB.T).T
    sa, sb = RA.shape[1], RB.shape[1]
    allg = np.concatenate(groups)
    point = float(RA[allg].mean() - RB[allg].mean())
    draws = []
    for _ in range(B):
        gs = np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
        a = RA[gs][:, RNG.integers(0, sa, sa)].mean()
        b = RB[gs][:, RNG.integers(0, sb, sb)].mean()
        draws.append(a - b)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    verdict = 'inconclusive' if lo <= 0 <= hi else ('A lower regret' if hi < 0 else 'B lower regret')
    return dict(point=point, lo=float(lo), hi=float(hi), verdict=verdict)


RNG = np.random.default_rng(23)


# ------------------------------------------------------------------ main

def load_family(kind, cal):
    """Return jobs, {method: [forecasts per seed]} and the observed-sales lookup for calibration."""
    if kind == 'orig':
        jobs = mc.build_jobs(cal)
        keys = [j['key'] for j in jobs]
        raw = r2.raw_original(keys)
        seed_raw = {s: {('yug', f): e2.load_npz(V2 / f'runs/kaggle-v2-01/output/v2/orig-seed{s}__yug__{f}.npz', keys)
                        for f in ('history', 'history_clean')} for s in SEEDS[1:]}
        for j in jobs:
            j['c0'] = j['ci']
    else:
        jobs = rc.build_jobs(cal)
        keys = [j['key'] for j in jobs]
        raw = r2.raw_v2(keys)
        seed_raw = {s: {('yug', f): e2.load_npz(V3_FM / f'v2-seed{s}__yug__{f}.npz', keys)
                        for f in ('history', 'history_clean')} for s in SEEDS[1:]}
    F = e2.process(jobs, raw)
    Fs = {m: [F[m]] for m in F}
    for s in SEEDS[1:]:
        P = e2.process(jobs, seed_raw[s])
        for m in YUG:
            Fs[m].append(P[m])
    return jobs, Fs


def observed_lookup(cal, families):
    out = {}
    for fam in families:
        scen, seed, prod = fam
        if (scen, seed) not in out:
            out[(scen, seed)] = rc.simulate(seed, cal, scen)['observed']
    return {fam: out[(fam[0], fam[1])][mc.PRODUCTS.index(fam[2])] for fam in families}


def regrets_committed(jobs, Fs, fam_jobs, fam_Fs, observed, idx=None):
    """{method: {variant: (n, seeds) regret}} for the jobs in idx."""
    sel = list(range(len(jobs))) if idx is None else list(idx)
    js = [jobs[i] for i in sel]
    tau = tau_of(js)
    out = {}
    for m, qs in Fs.items():
        out[m] = {}
        for variant in ('none', 'bias', 'bias+cal'):
            cols = []
            for si, q in enumerate(qs):
                qq = q[sel]
                bk = calibration_factors(js, fam_jobs, fam_Fs[m][si], observed, tau, m) if variant != 'none' else None
                T = targets(m, qq, js, variant, bk)
                cols.append(committed_regret(T, js)[0])
            out[m][variant] = np.stack(cols, axis=1)
    return out


def summarise_regret(Rm, jobs, mask):
    g = groups_of(jobs, mask)
    return {m: {v: hboot_mean(R[mask], [np.searchsorted(np.where(mask)[0], gi) for gi in g], RNG) for v, R in d.items()}
            for m, d in Rm.items()}


def pairwise(Rm, jobs, mask, methods, variant):
    g = groups_of(jobs, mask)
    gl = [np.searchsorted(np.where(mask)[0], gi) for gi in g]
    out = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            out[f'{a} − {b}'] = hboot_diff(Rm[a][variant][mask], Rm[b][variant][mask], gl)
    return out


def main():
    cal = mc.load_calendar(STUDY / 'data/festivals.json')
    res = dict(notes=__doc__, families={})
    # ---------------- original jobs (base + stress); calibration windows from base jobs of the same shop
    jo, Fo = load_family('orig', cal)
    fams = {history_family(j) for j in jo}
    obs = observed_lookup(cal, fams)
    base_idx = [i for i, j in enumerate(jo) if j['scenario'] == 'base']
    fam_jobs = [jo[i] for i in base_idx]
    fam_Fs = {m: [q[base_idx] for q in qs] for m, qs in Fo.items()}
    Ro = regrets_committed(jo, Fo, fam_jobs, fam_Fs, obs)
    S = ev.strata(jo)
    res['families']['original'] = dict(
        regret={s: summarise_regret(Ro, jo, m) for s, m in S.items()},
        pairwise_festival={v: pairwise(Ro, jo, S['base: festival season (all years)'], list(CAL_B), v) for v in ('none', 'bias', 'bias+cal')},
        yug_vs_history_only={v: {f'Yug-1 − {b}': hboot_diff(Ro['Yug-1'][v][S['base: festival season (all years)']],
                                                           Ro[b][v][S['base: festival season (all years)']],
                                                           [np.searchsorted(np.where(S['base: festival season (all years)'])[0], gi)
                                                            for gi in groups_of(jo, S['base: festival season (all years)'])])
                                 for b in ('TimesFM 2.5', 'Chronos-Bolt')} for v in ('none', 'bias', 'bias+cal')})
    print('original done', flush=True)
    # ---------------- v2 jobs: calendar-dependence scenarios (committed) and rolling
    jv, Fv = load_family('v2', cal)
    fams = {history_family(j) for j in jv if j['scenario'] in rc.ROBUST}
    obs_r = observed_lookup(cal, fams)
    ridx = [i for i, j in enumerate(jv) if j['scenario'] in rc.ROBUST]
    jr = [jv[i] for i in ridx]
    Fr = {m: [q[ridx] for q in qs] for m, qs in Fv.items()}
    Rr = regrets_committed(jr, Fr, jr, Fr, obs_r)
    SR = r2.strata_v2(jr)
    fest = {s: m for s, m in SR.items() if s.endswith('festival season')}
    res['families']['calendar_dependence'] = dict(
        regret={s: summarise_regret(Rr, jr, m) for s, m in SR.items()},
        pairwise_festival={s: {v: pairwise(Rr, jr, m, list(CAL_B), v) for v in ('none', 'bias+cal')} for s, m in fest.items()})
    print('calendar dependence done', flush=True)
    # ---------------- rolling: calibration factors from the original base jobs of the same shop/product
    base_map = {(j['seed'], j['year'], j['cutoff'], j['product'], j['offset']): i for i, j in enumerate(jv) if j['scenario'] == 'base'}
    j0 = [jv[base_map[(s, y, c, p, 0)]] for s in rc.ROLL_SEEDS for y in mc.YEARS for c in mc.FESTIVAL_CUTOFFS for p in mc.PRODUCTS]
    D = np.stack([j['truth'] for j in j0])
    cap = np.stack([j['cap'] for j in j0]) * ev.CAP_FACTOR
    pr, co, sh, ish, tau = e2.econ_arrays(j0)
    T_ref = np.repeat(D[None], len(rc.ROLL_OFFSETS), axis=0)
    ref = e2.rolling_inventory(T_ref, D, cap, sh, ish, pr, co)
    Rroll = {}
    for m in Fv:
        Rroll[m] = {}
        for variant in ('none', 'bias', 'bias+cal'):
            cols = []
            for si in range(len(Fv[m])):
                bk = calibration_factors(j0, fam_jobs, fam_Fs[m][si], obs, tau, m) if variant != 'none' else None
                Tk = np.zeros((len(rc.ROLL_OFFSETS), len(j0), H))
                for k_, off in enumerate(rc.ROLL_OFFSETS):
                    ik = [base_map[(j['seed'], j['year'], j['cutoff'], j['product'], off)] for j in j0]
                    jk = [jv[i] for i in ik]
                    q = Fv[m][si][ik]
                    pt = base_point(m, q, tau, variant)
                    if variant == 'bias':
                        pt = pt * bk[0][:, None]
                    elif variant == 'bias+cal':
                        pt = pt * bk[1][:, None]
                    plan = np.zeros_like(D)
                    plan[:, off:] = pt[:, :H - off]
                    Tk[k_] = np.maximum(plan, np.stack([j['confirmed'] for j in jk]))
                cols.append(ref['contribution'] - e2.rolling_inventory(Tk, D, cap, sh, ish, pr, co)['contribution'])
            Rroll[m][variant] = np.stack(cols, axis=1)
    allm = np.ones(len(j0), bool)
    res['families']['rolling'] = dict(regret={'base festival season, 20 shops': summarise_regret(Rroll, j0, allm)},
                                      pairwise_festival={v: pairwise(Rroll, j0, allm, list(CAL_B), v) for v in ('none', 'bias+cal')},
                                      yug_vs_history_only={v: {f'Yug-1 − {b}': hboot_diff(Rroll['Yug-1'][v], Rroll[b][v], groups_of(j0, allm))
                                                               for b in ('TimesFM 2.5', 'Chronos-Bolt')} for v in ('none', 'bias+cal')})
    print('rolling done', flush=True)
    (V3 / 'results').mkdir(exist_ok=True)
    (V3 / 'results/final-v3.json').write_text(json.dumps(res, indent=1, default=float))
    print('wrote final-v3.json')


if __name__ == '__main__':
    main()
