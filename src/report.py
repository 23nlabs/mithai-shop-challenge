"""Aggregate accuracy and business metrics (seed-bootstrap 95% intervals) → results/metrics.json + tables.md."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import mithai_core as mc  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def main(kaggle_out):
    cal = mc.load_calendar(HERE / 'data/festivals.json')
    jobs = mc.build_jobs(cal)
    sha = mc.histories_sha256(jobs)
    expected = (HERE / 'evidence/histories.sha256').read_text().strip()
    assert sha == expected, 'HISTORY_HASH_MISMATCH'
    F, gate = ev.load_forecasts(jobs, [HERE / 'runs/local-baselines', kaggle_out])
    missing = [m for m in ev.METHODS if m not in F]
    acc = ev.per_job_accuracy(jobs, F, cal)
    bus, tau = ev.business(jobs, F)
    bus_out, _ = ev.business(jobs, F, mode='order-up-to')
    rng = np.random.default_rng(23)
    res = dict(n_jobs=len(jobs), histories_sha256=sha, gate=gate, missing_methods=missing,
               tau={p: float(tau[[j['product'] == p for j in jobs]][0]) for p in mc.PRODUCTS}, strata={})
    for sname, mask in ev.strata(jobs).items():
        seeds, groups = ev.seed_index(jobs, mask)
        st = dict(n_jobs=int(mask.sum()), n_seeds=len(seeds), methods={})
        for name in list(F) + ['Hindsight reference']:
            r = {}
            if name in acc:
                a = acc[name]
                for h in mc.HORIZONS:
                    r[f'wape_{h}'] = ev.boot_ratio(a['abs_err'][h], a['actual'][h], groups, rng)
                    r[f'mae_{h}'] = float(a['abs_err'][h][mask].sum() / (mask.sum() * h))
                r['wape_by_week'] = [float(a['week_err'][mask, w].sum() / a['week_act'][mask, w].sum()) for w in range(8)]
                if 'pinball' in a:
                    r['wql'] = ev.boot_ratio(a['pinball'], a['actual'][56], groups, rng)
                    r['coverage80'] = ev.boot_ratio(a['covered'].astype(float), np.full(len(jobs), float(ev.H)), groups, rng)
                    r['width_rel'] = ev.boot_ratio(a['width'], a['actual'][56], groups, rng)
                r['bias'] = ev.boot_ratio(a['bias'] * a['actual'][56], a['actual'][56], groups, rng)
                for k in ('peak', 'surge', 'phantom', 'phantom_rel'):
                    x = a[k]
                    if np.isfinite(x[mask]).sum():
                        r[f'{k}_median'] = ev.boot_mean(x, groups, rng, stat=np.nanmedian)
                        if k == 'peak':
                            r['peak_abs_median'] = ev.boot_mean(np.abs(x), groups, rng, stat=np.nanmedian)
                            r['peak_early_share'] = ev.boot_mean((x <= -7).astype(float) + 0 * x, groups, rng)
                            r['no_peak_share'] = ev.boot_mean(a['no_peak'], groups, rng)
            b = bus[name]
            r['contribution_per_job'] = ev.boot_mean(b['contribution'], groups, rng)
            r['waste_share'] = ev.boot_ratio(b['waste'], b['made'], groups, rng)
            r['lost_share'] = ev.boot_ratio(b['lost'], b['demand'], groups, rng)
            r['stockout_rate'] = ev.boot_mean(b['stockout_rate'], groups, rng)
            r['waste_kg_per_job'] = float(b['waste'][mask].mean())
            r['lost_kg_per_job'] = float(b['lost'][mask].mean())
            bo = bus_out[name]
            r['order_up_to'] = dict(waste_share=ev.boot_ratio(bo['waste'], bo['made'], groups, rng),
                                    lost_share=ev.boot_ratio(bo['lost'], bo['demand'], groups, rng))
            if 'regret' in bo:
                r['order_up_to']['regret_per_job'] = ev.boot_mean(bo['regret'], groups, rng)
            if 'regret' in b:
                r['regret_per_job'] = ev.boot_mean(b['regret'], groups, rng)
                r['fva_vs_shopkeeper_per_job'] = ev.boot_mean(bus['Shopkeeper']['contribution'] * -1 + b['contribution'], groups, rng)
            st['methods'][name] = r
        res['strata'][sname] = st
    # economic sensitivity (base, festival season), one factor at a time
    mask = ev.strata(jobs)['base: festival season (all years)']
    seeds, groups = ev.seed_index(jobs, mask)
    sens = {}
    grid = [('base', {}), ('margin -20%', dict(price_scale=0.8)), ('margin +20%', dict(price_scale=1.2)),
            ('shelf life x0.5', dict(shelf_scale=0.5)), ('shelf life x1.5', dict(shelf_scale=1.5)),
            ('goodwill penalty 50% of price', dict(goodwill_share=0.5)), ('median policy (tau=0.5)', dict(policy='median'))]
    for label, kw in grid:
        b, _ = ev.business(jobs, F, **kw)
        sens[label] = {m: dict(regret_per_job=ev.boot_mean(b[m]['regret'], groups, rng),
                               fva_vs_shopkeeper=ev.boot_mean(b[m]['contribution'] - b['Shopkeeper']['contribution'], groups, rng))
                       for m in F}
    res['sensitivity_festival_season'] = sens
    # scale context: share of latent demand from random bulk orders and corporate orders (base, festival season)
    m = ev.strata(jobs)['base: festival season (all years)']
    res['hindsight_contribution_per_job'] = {s: float(bus['Hindsight reference']['contribution'][mk].mean())
                                             for s, mk in ev.strata(jobs).items()}
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results/metrics.json').write_text(json.dumps(res, indent=1))
    write_tables(res)
    print('gate:', json.dumps(gate))
    print('missing:', missing)


def fmt(x, pct=False, dec=1):
    if x is None:
        return 'n/a'
    s = 100 if pct else 1
    return f"{x['point']*s:.{dec}f} [{x['lo']*s:.{dec}f}, {x['hi']*s:.{dec}f}]"


def write_tables(res):
    lines = ['# Mithai Shop Challenge — verified result tables', '',
             f"Jobs: {res['n_jobs']}. History hash `{res['histories_sha256'][:16]}…`. Intervals: 95% bootstrap over simulated shops (seeds).", '']
    order = [m for m in res['strata']['base: festival season (all years)']['methods'] if m != 'Hindsight reference']
    for sname, st in res['strata'].items():
        lines += [f"## {sname} — {st['n_jobs']} product-forecasts, {st['n_seeds']} simulated shops", '',
                  '| Method | WAPE 28d % | WAPE 56d % | wQL 56d % | 80% cov % | Peak err (median days) | Surge err (median %) | Phantom surge (median %) | Waste % of made | Lost % of demand | Regret ₹/forecast | vs shopkeeper ₹/forecast |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for m in order + ['Hindsight reference']:
            r = st['methods'].get(m)
            if not r:
                continue
            lines.append('| ' + ' | '.join([
                m, fmt(r.get('wape_28'), True), fmt(r.get('wape_56'), True), fmt(r.get('wql'), True), fmt(r.get('coverage80'), True, 0),
                fmt(r.get('peak_median'), False, 0), fmt(r.get('surge_median'), True, 0), fmt(r.get('phantom_median'), True, 0),
                fmt(r.get('waste_share'), True), fmt(r.get('lost_share'), True),
                fmt(r.get('regret_per_job'), False, 0), fmt(r.get('fva_vs_shopkeeper_per_job'), False, 0)]) + ' |')
        lines.append('')
    lines += ['## Economic sensitivity (base, festival season): regret ₹ per product-forecast', '',
              '| Assumption | ' + ' | '.join(order) + ' |', '|---' * (len(order) + 1) + '|']
    for label, row in res['sensitivity_festival_season'].items():
        lines.append(f'| {label} | ' + ' | '.join(f"{row[m]['regret_per_job']['point']:.0f}" for m in order) + ' |')
    (HERE / 'results/tables.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
