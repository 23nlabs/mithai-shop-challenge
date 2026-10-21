"""Robustness v2 analysis → robustness/v2/results/robustness-v2.json (+ tables-v2.md written by tables_v2.py).

Parts
  P1 decision-policy fairness on the ORIGINAL 2,160 jobs: original 11 methods + calibrated shopkeeper
     (τ policy) + censoring-aware calendar regression; original results kept as a separate comparison.
  P2 calendar dependence: R1-shape / R2-shocks / R3-both (20 shops each) vs the matched base (seeds 0-19).
  P3 rolling weekly replanning vs the original 56-day committed plan (base festival season, seeds 0-19).
  P4 Yug-1 sampling seeds (67067 original + 1-4) on the original jobs.
  V  verification: counts, independent metric recomputation, hindsight feasibility, forecast consistency
     between runs, original-artifact checksums.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
STUDY = V2.parents[1]
sys.path.insert(0, str(STUDY / 'src'))
sys.path.insert(0, str(V2 / 'src'))
import evaluate as ev  # noqa: E402
import eval_v2 as e2  # noqa: E402
import mithai_core as mc  # noqa: E402
import robust_core as rc  # noqa: E402

H = mc.HORIZON
ORIG_LOCAL = STUDY / 'runs/local-baselines'
ORIG_FM = STUDY / 'runs/kaggle-01/output/mithai'
V2_LOCAL = V2 / 'runs/local-v2'
V2_FM = V2 / 'runs/kaggle-v2-01/output/v2'
FM_STEMS = ('yug', 'timesfm-2.5', 'chronos-bolt-base')
LOCAL_STEMS = ('seasonal-naive-52w', 'ets', 'calendar-regression', 'shopkeeper', 'ets+calendar',
               'shopkeeper-calibrated', 'calreg-censoring-aware')


def raw_original(keys):
    raw = {}
    for stem in LOCAL_STEMS[:5]:
        raw[(stem, 'history')] = e2.load_npz(ORIG_LOCAL / f'{stem}__history.npz', keys)
    for stem in LOCAL_STEMS[5:]:
        raw[(stem, 'history')] = e2.load_npz(V2_LOCAL / f'orig__{stem}__history.npz', keys)
    for stem in FM_STEMS:
        for field in ('history', 'history_clean'):
            raw[(stem, field)] = e2.load_npz(ORIG_FM / f'{stem}__{field}.npz', keys)
    return raw


def raw_v2(keys):
    raw = {}
    for stem in LOCAL_STEMS:
        f = V2_LOCAL / f'v2__{stem}__history.npz'
        if f.exists():
            raw[(stem, 'history')] = e2.load_npz(f, keys)
        else:
            print('MISSING (dry run only):', f.name)
    for stem in FM_STEMS:
        for field in ('history', 'history_clean'):
            raw[(stem, field)] = e2.load_npz(V2_FM / f'v2__{stem}__{field}.npz', keys)
    return raw


def subset(raw, idx):
    return {k: v[idx] for k, v in raw.items()}


def strata_v2(jobs):
    S = {}
    for scen in rc.ROBUST:
        m = np.array([j['scenario'] == scen and j['offset'] == 0 for j in jobs])
        fest = np.array([j['kind'] == 'festival' for j in jobs])
        shift = np.array([j['year'] in (2023, 2026) for j in jobs])
        S[f'{scen}: festival season'] = m & fest
        S[f'{scen}: festival season, +19-day years'] = m & fest & shift
        S[f'{scen}: ordinary periods'] = m & ~fest
    return S


def calendar_gain(jobs, acc, mask, rng):
    """WAPE(model) - WAPE(model + calendar), paired within shops; positive = the calendar helps."""
    _, groups = ev.seed_index(jobs, mask)
    out = {}
    for a, b in (('Yug-1', 'Yug-1 + calendar'), ('TimesFM 2.5', 'TimesFM + calendar'),
                 ('Chronos-Bolt', 'Chronos + calendar'), ('ETS', 'ETS + calendar')):
        if a not in acc or b not in acc:
            continue
        out[f'{a} → {b}'] = e2.paired_ratio_diff(acc[a]['abs_err'][56], acc[a]['actual'][56],
                                                 acc[b]['abs_err'][56], acc[b]['actual'][56], groups, rng)
    best_a = min([m for m in ('Yug-1', 'TimesFM 2.5', 'Chronos-Bolt', 'ETS') if m in acc],
                 key=lambda m: acc[m]['abs_err'][56][mask].sum() / acc[m]['actual'][56][mask].sum())
    out[f'best history-only ({best_a}) → calendar regression'] = e2.paired_ratio_diff(
        acc[best_a]['abs_err'][56], acc[best_a]['actual'][56],
        acc['Calendar regression']['abs_err'][56], acc['Calendar regression']['actual'][56], groups, rng)
    return out


def main():
    rng = np.random.default_rng(23)
    cal = mc.load_calendar(STUDY / 'data/festivals.json')
    res = dict(verification={})
    V = res['verification']

    # ---------------- P1: original jobs with fair-policy additions
    jo = mc.build_jobs(cal)
    V['original_history_sha256_match'] = mc.histories_sha256(jo) == (STUDY / 'evidence/histories.sha256').read_text().strip()
    ko = [j['key'] for j in jo]
    raw_o = raw_original(ko)
    Fo = e2.process(jo, raw_o)
    acc_o = ev.per_job_accuracy(jo, Fo, cal)
    bus_o, tau_o = ev.business(jo, Fo)
    res['P1_policy_fairness'] = {s: e2.summarise(jo, Fo, acc_o, bus_o, m, rng) for s, m in ev.strata(jo).items()}
    # original numbers reproduced?
    orig = json.loads((STUDY / 'results/metrics.json').read_text())
    diffs = []
    for s, st in orig['strata'].items():
        for m, r in st['methods'].items():
            if 'wape_56' in r:
                diffs.append(abs(r['wape_56']['point'] - res['P1_policy_fairness'][s]['methods'][m]['wape_56']['point']))
            if r.get('regret_per_job'):
                diffs.append(abs(r['regret_per_job']['point'] - res['P1_policy_fairness'][s]['methods'][m]['regret_per_job']['point']) / 1e5)
    V['original_point_estimates_reproduced_max_abs_diff'] = float(max(diffs))

    # ---------------- V: independent recomputation of WAPE and counts
    Y = np.stack([j['truth'] for j in jo])
    indep = {}
    for m in ('Yug-1', 'TimesFM 2.5', 'Shopkeeper', 'Yug-1 + calendar'):
        mask = ev.strata(jo)['base: festival season (all years)']
        indep[m] = float(np.abs(Fo[m][mask, :, 4] - Y[mask]).sum() / Y[mask].sum())
    V['independent_wape_festival'] = indep
    V['job_counts_original'] = dict(Counter(j['scenario'] for j in jo))
    # hindsight reference feasibility (committed plan): independent checked re-implementation
    D = Y
    cap = np.stack([j['cap'] for j in jo]) * ev.CAP_FACTOR
    pr, co, sh, ish, tau = e2.econ_arrays(jo)
    chk = e2.committed_checked(np.maximum(D, 0), D, cap, sh, pr, co)
    V['hindsight_committed_feasible'] = True  # assertions inside committed_checked passed
    V['hindsight_committed_matches_original_sim'] = bool(np.allclose(chk['contribution'], bus_o['Hindsight reference']['contribution']))
    V['hindsight_committed_lost_share'] = float(chk['lost'].sum() / D.sum())
    V['hindsight_committed_waste_kg_total'] = float(chk['waste'].sum())
    # same checked simulator on one forecast-driven policy, to verify the business code path
    T = np.maximum(ev.quantile_at(Fo['Yug-1 + calendar'], tau), np.stack([j['confirmed'] for j in jo]))
    chk2 = e2.committed_checked(T, D, cap, sh, pr, co)
    V['policy_sim_matches_original_sim'] = bool(np.allclose(chk2['contribution'], bus_o['Yug-1 + calendar']['contribution']))

    # ---------------- P4: Yug-1 sampling seeds on original jobs
    seeds = {}
    for s in ('67067', '1', '2', '3', '4'):
        if s == '67067':
            rs = {('yug', f): raw_o[('yug', f)] for f in ('history', 'history_clean')}
        else:
            rs = {('yug', f): e2.load_npz(V2_FM / f'orig-seed{s}__yug__{f}.npz', ko) for f in ('history', 'history_clean')}
        Fs = e2.process(jo, rs)
        a = ev.per_job_accuracy(jo, Fs, cal)
        b, _ = ev.business(jo, Fs)
        row = {}
        for name in ('Yug-1', 'Yug-1 + calendar'):
            for sname in ('base: festival season (all years)', 'base: ordinary periods',
                          'base: festival season, +19-day shift years (2023, 2026)'):
                msk = ev.strata(jo)[sname]
                row[f'{name} | {sname} | wape_56'] = float(a[name]['abs_err'][56][msk].sum() / a[name]['actual'][56][msk].sum())
                row[f'{name} | {sname} | regret'] = float(b[name]['regret'][msk].mean())
                row[f'{name} | {sname} | coverage80'] = float(a[name]['covered'][msk].sum() / (msk.sum() * H))
                if 'shift' in sname:
                    row[f'{name} | {sname} | phantom_rel_median'] = float(np.nanmedian(a[name]['phantom_rel'][msk]))
                    row[f'{name} | {sname} | distinct_peak_share'] = float(np.nanmean(1 - a[name]['no_peak'][msk]))
        seeds[s] = row
    res['P4_yug_seeds'] = seeds

    # ---------------- P2 + P3 on v2 jobs
    jv = rc.build_jobs(cal)
    V['v2_history_sha256_match'] = rc.histories_sha256(jv) == (V2 / 'evidence/histories-v2.sha256').read_text().strip()
    V['job_counts_v2'] = {f'{k[0]} +{k[1]}': v for k, v in Counter((j['scenario'], j['offset']) for j in jv).items()}
    kv = [j['key'] for j in jv]
    raw_all = raw_v2(kv)
    # P2
    idx = np.array([i for i, j in enumerate(jv) if j['scenario'] in rc.ROBUST])
    jr = [jv[i] for i in idx]
    Fr = e2.process(jr, subset(raw_all, idx))
    acc_r = ev.per_job_accuracy(jr, Fr, cal)
    bus_r, _ = ev.business(jr, Fr)
    P2 = {s: e2.summarise(jr, Fr, acc_r, bus_r, m, rng) for s, m in strata_v2(jr).items()}
    gains = {}
    for s, m in strata_v2(jr).items():
        if 'ordinary' not in s:
            gains[s] = calendar_gain(jr, acc_r, m, rng)
    # matched base reference: original jobs, seeds 0-19, same cut-offs
    ref = {
        'base (seeds 0-19): festival season': np.array([j['scenario'] == 'base' and j['seed'] < 20 and j['kind'] == 'festival' for j in jo]),
        'base (seeds 0-19): festival season, +19-day years': np.array([j['scenario'] == 'base' and j['seed'] < 20 and j['kind'] == 'festival' and j['year'] in (2023, 2026) for j in jo]),
        'base (seeds 0-19): ordinary periods': np.array([j['scenario'] == 'base' and j['seed'] < 20 and j['kind'] == 'ordinary' for j in jo])}
    for s, m in ref.items():
        P2[s] = e2.summarise(jo, Fo, acc_o, bus_o, m, rng)
        if 'ordinary' not in s:
            gains[s] = calendar_gain(jo, acc_o, m, rng)
    res['P2_calendar_dependence'] = P2
    res['P2_calendar_gain'] = gains

    # P3 rolling
    base_idx = {(j['seed'], j['year'], j['cutoff'], j['product'], j['offset']): i for i, j in enumerate(jv) if j['scenario'] == 'base'}
    i0 = np.array([base_idx[(s, y, c, p, 0)] for s in rc.ROLL_SEEDS for y in mc.YEARS for c in mc.FESTIVAL_CUTOFFS for p in mc.PRODUCTS])
    j0 = [jv[i] for i in i0]
    D = np.stack([j['truth'] for j in j0])
    cap = np.stack([j['cap'] for j in j0]) * ev.CAP_FACTOR
    pr, co, sh, ish, tau = e2.econ_arrays(j0)
    roll, comm = {}, {}
    # forecasts per origin
    F_by_off = {}
    for k, off in enumerate(rc.ROLL_OFFSETS):
        ik = np.array([base_idx[(j['seed'], j['year'], j['cutoff'], j['product'], off)] for j in j0])
        jk = [jv[i] for i in ik]
        F_by_off[off] = (jk, e2.process(jk, subset(raw_all, ik)))
    # consistency: offset-0 v2 forecasts must equal the original run's forecasts for the same jobs
    okey = {j['key']: i for i, j in enumerate(jo)}
    io = np.array([okey[f"base|{j['seed']}|{j['year']}|{j['cutoff']}|{j['product']}"] for j in j0])
    cons = {}
    for name, q in F_by_off[0][1].items():
        if name in Fo:
            cons[name] = float(np.max(np.abs(q - Fo[name][io]) / np.maximum(1, np.abs(Fo[name][io]))))
    V['offset0_vs_original_forecasts_max_rel_diff'] = cons
    names = list(F_by_off[0][1])
    for name in names + ['Hindsight reference']:
        Tk = np.zeros((len(rc.ROLL_OFFSETS), len(j0), H))
        for k, off in enumerate(rc.ROLL_OFFSETS):
            jk, Fk = F_by_off[off]
            conf = np.stack([j['confirmed'] for j in jk])
            if name == 'Hindsight reference':
                t_plan = D.copy()
            else:
                tg = e2.targets(name, Fk[name], np.zeros_like(D), tau)  # forecast-day coordinates
                t_plan = np.zeros_like(D)
                t_plan[:, off:] = tg[:, :H - off]
            Tk[k] = np.maximum(t_plan, conf)
        roll[name] = e2.rolling_inventory(Tk, D, cap, sh, ish, pr, co, check=(name == 'Hindsight reference'))
    V['hindsight_rolling_feasible'] = True  # assertions passed inside rolling_inventory(check=True)
    V['hindsight_rolling_lost_share'] = float(roll['Hindsight reference']['lost'].sum() / D.sum())
    for name in roll:
        if name != 'Hindsight reference':
            roll[name]['regret'] = roll['Hindsight reference']['contribution'] - roll[name]['contribution']
    F0 = {n: q for n, q in F_by_off[0][1].items()}
    comm, _ = ev.business(j0, F0)
    mask = np.ones(len(j0), bool)
    acc0 = ev.per_job_accuracy(j0, F0, cal)
    res['P3_rolling'] = dict(rolling=e2.summarise(j0, F0, acc0, roll, mask, rng),
                             committed_same_jobs=e2.summarise(j0, F0, acc0, comm, mask, rng),
                             design=dict(lead_time_days=e2.LT, ingredient_share=e2.ING_SHARE, ingredient_shelf=e2.ING_SHELF,
                                         origins=rc.ROLL_OFFSETS, jobs=len(j0)))
    # original artifacts unchanged?
    bad = []
    for line in (STUDY / 'evidence/MANIFEST.sha256').read_text().splitlines():
        h, p = line.split('  ', 1)
        f = STUDY / p
        if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() != h:
            bad.append(p)
    V['original_artifacts_changed'] = bad
    (V2 / 'results').mkdir(exist_ok=True)
    (V2 / 'results/robustness-v2.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(V, indent=1, default=float))


if __name__ == '__main__':
    main()
