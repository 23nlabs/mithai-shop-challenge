"""Write robustness/v2/results/tables-v2.md from robustness-v2.json only."""
import json
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]


def f(x, pct=False, dec=1, sign=False):
    if not x:
        return 'n/a'
    s = 100 if pct else 1
    fm = f"{{:{'+' if sign else ''}.{dec}f}}"
    return f"{fm.format(x['point']*s)} [{fm.format(x['lo']*s)}, {fm.format(x['hi']*s)}]"


def ftab(st, methods, cols):
    head = '| Method | ' + ' | '.join(c[0] for c in cols) + ' |'
    lines = [head, '|---' * (len(cols) + 1) + '|']
    for m in methods:
        r = st['methods'].get(m)
        if r is None:
            continue
        lines.append(f'| {m} | ' + ' | '.join(c[1](r) for c in cols) + ' |')
    return lines


ACC = [('WAPE 56d %', lambda r: f(r.get('wape_56'), True)), ('Bias %', lambda r: f(r.get('bias'), True, 0, True)),
       ('80% cov %', lambda r: f(r.get('coverage80'), True, 0)),
       ('Distinct peak %', lambda r: f(r.get('distinct_peak_share'), True, 0)),
       ('Peak err d (median)', lambda r: f(r.get('peak_median'), False, 0, True)),
       ('Regret ₹/plan', lambda r: f(r.get('regret_per_job'), False, 0)),
       ('Waste %', lambda r: f(r.get('waste_share'), True)), ('Lost %', lambda r: f(r.get('lost_share'), True))]


def main():
    r = json.loads((V2 / 'results/robustness-v2.json').read_text())
    L = ['# Mithai Shop Challenge — robustness v2 tables', '',
         'SYNTHETIC, controlled research. Separately versioned; the original results (`../../results/`) are unchanged.',
         'Intervals: 95% bootstrap over simulated shops. Regret = hindsight-reference contribution − policy contribution (simulated ₹, not real savings).', '',
         '## Verification', '', '```', json.dumps(r['verification'], indent=1, default=float), '```', '']
    P1 = r['P1_policy_fairness']
    methods = [m for m in P1['base: festival season (all years)']['methods'] if m != 'Hindsight reference']
    for s in ('base: festival season (all years)', 'base: festival season, +19-day shift years (2023, 2026)', 'base: ordinary periods'):
        st = P1[s]
        L += [f"## P1 Decision-policy fairness — {s} ({st['n_jobs']} plans, {st['n_seeds']} shops, committed plan)", ''] + ftab(st, methods, ACC) + ['']
    for s in [k for k in P1 if k.startswith('stress')]:
        st = P1[s]
        L += [f"### P1 — {s} ({st['n_jobs']} plans)", ''] + ftab(st, methods, ACC[:1] + ACC[5:]) + ['']
    P2 = r['P2_calendar_dependence']
    L += ['## P2 Calendar dependence — festival season', '']
    for s in ('base (seeds 0-19): festival season', 'R1-shape: festival season', 'R2-shocks: festival season', 'R3-both: festival season'):
        st = P2[s]
        L += [f"### {s} ({st['n_jobs']} plans)", ''] + ftab(st, methods, ACC) + ['']
    for s in ('base (seeds 0-19): ordinary periods', 'R3-both: ordinary periods'):
        st = P2[s]
        L += [f"### {s} ({st['n_jobs']} plans)", ''] + ftab(st, methods, ACC[:3] + ACC[5:6]) + ['']
    L += ['### Calendar gain: WAPE(model) − WAPE(model + calendar), percentage points (positive = calendar helps)', '',
          '| Stratum | ' + ' | '.join(next(iter(r['P2_calendar_gain'].values())).keys()) + ' |', '|---' * 7 + '|']
    for s, g in r['P2_calendar_gain'].items():
        L.append(f'| {s} | ' + ' | '.join(f(v, True, 1, True) for v in g.values()) + ' |')
    P3 = r['P3_rolling']
    L += ['', f"## P3 Rolling weekly replanning vs 56-day committed plan ({P3['rolling']['n_jobs']} festival-season plans, 20 shops)", '',
          f"Design: {json.dumps(P3['design'])}", '',
          '| Method | Regret rolling | Regret committed | Waste % rolling (finished) | Ingredient waste % rolling | Lost % rolling | Lost % committed |',
          '|---|---|---|---|---|---|---|']
    for m in methods:
        a, b = P3['rolling']['methods'].get(m), P3['committed_same_jobs']['methods'].get(m)
        if a and b:
            L.append(f"| {m} | {f(a['regret_per_job'], False, 0)} | {f(b['regret_per_job'], False, 0)} | {f(a['waste_share'], True)} | "
                     f"{f(a.get('ingredient_waste_share'), True)} | {f(a['lost_share'], True)} | {f(b['lost_share'], True)} |")
    hr = P3['rolling']['methods']['Hindsight reference']
    L += [f"| Hindsight reference (rolling) | 0 | 0 | {f(hr['waste_share'], True)} | {f(hr.get('ingredient_waste_share'), True)} | {f(hr['lost_share'], True)} | — |", '']
    P4 = r['P4_yug_seeds']
    keys = list(next(iter(P4.values())).keys())
    L += ['## P4 Yug-1 sampling seeds (original jobs; 100 paths each)', '', '| Metric | ' + ' | '.join(f'seed {s}' for s in P4) + ' | range |',
          '|---' * (len(P4) + 2) + '|']
    for k in keys:
        vals = [P4[s][k] for s in P4]
        pct = 'regret' not in k
        fmt = (lambda v: f'{v*100:.1f}') if pct else (lambda v: f'{v:,.0f}')
        L.append(f'| {k} | ' + ' | '.join(fmt(v) for v in vals) + f' | {fmt(max(vals) - min(vals))} |')
    (V2 / 'results/tables-v2.md').write_text('\n'.join(L) + '\n')
    print('wrote tables-v2.md', len(L), 'lines')


if __name__ == '__main__':
    main()
