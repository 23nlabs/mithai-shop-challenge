"""Product-level and horizon-week tables (base scenario), appended to results/tables.md and metrics.json."""
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
    F, _ = ev.load_forecasts(jobs, [HERE / 'runs/local-baselines', kaggle_out])
    acc = ev.per_job_accuracy(jobs, F, cal)
    bus, _ = ev.business(jobs, F)
    res = json.loads((HERE / 'results/metrics.json').read_text())
    rng = np.random.default_rng(23)
    lines = ['', '## Product level — base, festival season (WAPE 56d %, regret ₹ per plan, waste %, lost %)', '',
             '| Method | ' + ' | '.join(f'{mc.NAMES[p]} WAPE | {mc.NAMES[p]} regret' for p in mc.PRODUCTS) + ' |',
             '|---' * (1 + 2 * len(mc.PRODUCTS)) + '|']
    prod = {}
    base_fest = ev.strata(jobs)['base: festival season (all years)']
    for m in F:
        cells = []
        prod[m] = {}
        for p in mc.PRODUCTS:
            mask = base_fest & np.array([j['product'] == p for j in jobs])
            _, g = ev.seed_index(jobs, mask)
            w = ev.boot_ratio(acc[m]['abs_err'][56], acc[m]['actual'][56], g, rng)
            r = ev.boot_mean(bus[m]['regret'], g, rng)
            prod[m][p] = dict(wape_56=w, regret_per_job=r)
            cells += [f"{w['point']*100:.1f} [{w['lo']*100:.1f}, {w['hi']*100:.1f}]", f"{r['point']:,.0f} [{r['lo']:,.0f}, {r['hi']:,.0f}]"]
        lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## WAPE % by week into the horizon — base, festival season', '',
              '| Method | ' + ' | '.join(f'wk {w+1}' for w in range(8)) + ' |', '|---' * 9 + '|']
    st = res['strata']['base: festival season (all years)']['methods']
    for m in F:
        lines.append(f'| {m} | ' + ' | '.join(f'{x*100:.0f}' for x in st[m]['wape_by_week']) + ' |')
    lines += ['', '## Forecast failure counts (all 2,160 jobs per method)', '', '| Method | non-finite | crossed quantiles |', '|---|---|---|']
    for m, g in res['gate'].items():
        lines.append(f"| {m} | {g['nonfinite']} | {g['crossed_quantiles']} |")
    res['product_level_festival'] = prod
    (HERE / 'results/metrics.json').write_text(json.dumps(res, indent=1))
    with open(HERE / 'results/tables.md', 'a') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1])
