"""Mithai Shop Challenge figures — generated from results/metrics.json and regenerated simulator data only."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'src'))
sys.path.insert(0, str(HERE / 'src'))
import evaluate as ev  # noqa: E402
import mithai_core as mc  # noqa: E402
from make_figures import SURFACE, INK, INK2, MUTED, GRID, style  # noqa: E402

FIG = HERE / 'figures'
YUG, OTHER = '#2a78d6', '#7d7c78'
LINES = {'Yug-1': ('#2a78d6', '-'), 'Yug-1 + calendar': ('#2a78d6', '--'), 'TimesFM 2.5': ('#eb6834', '-'),
         'Seasonal-naive 52w': ('#1baf7a', '-'), 'Shopkeeper': ('#eda100', '-')}
SOURCE = '23N Labs · Mithai Shop Challenge · SYNTHETIC simulated retailer (not real sales) · 30 simulated shops'


def save(fig, name, title, subtitle):
    fig.suptitle(title, x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK, y=0.995)
    fig.text(0.01, 0.915, subtitle, ha='left', fontsize=9.5, color=INK2)
    fig.text(0.01, 0.01, SOURCE, ha='left', fontsize=7.5, color=MUTED)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=200)
    plt.close(fig)
    print('wrote', name)


def forest(ax, rows, xlabel, scale=1.0, ref=None, fmtstr='{:.1f}'):
    for y, (label, m, x) in enumerate(rows):
        if x is None:
            ax.text(0.02, y, 'no distinct peak forecast', transform=ax.get_yaxis_transform(), va='center', fontsize=8, color=MUTED)
            continue
        c = YUG if m.startswith('Yug') else OTHER
        hollow = '+' in m or m in ('Calendar regression', 'Shopkeeper')
        ax.plot([x['lo'] * scale, x['hi'] * scale], [y, y], color=c, linewidth=2, solid_capstyle='round')
        ax.plot([x['point'] * scale], [y], 'o', color=SURFACE if hollow else c, markeredgecolor=c,
                markersize=7.5, markeredgewidth=2)
        ax.text(x['hi'] * scale, y, '  ' + fmtstr.format(x['point'] * scale), va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(INK if r[1].startswith('Yug') else INK2)
        t.set_fontweight('bold' if r[1].startswith('Yug') else 'normal')
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if ref is not None:
        ax.axvline(ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlabel(xlabel)
    style(ax)


def method_order(res):
    m = res['strata']['base: festival season (all years)']['methods']
    a = [k for k in ev.METHODS if k in m and ev.METHODS[k][0] == 'A']
    b = [k for k in ev.METHODS if k in m and ev.METHODS[k][0] == 'B']
    return a, b


def label(m):
    return f"{m}  [{'A' if ev.METHODS[m][0] == 'A' else 'B'}]"


def fig1_shifts(cal):
    s = mc.simulate(0, cal)
    fig, axes = plt.subplots(5, 1, figsize=(10, 7.2), sharex=True)
    for ax, year in zip(axes, range(2022, 2027)):
        lo = mc.IDX[mc.dt.date(year, 9, 1)]
        x = np.arange(92)
        ax.plot(x, s['demand'][0, lo:lo + 92], color=INK2, linewidth=1.4)
        d = mc.IDX[cal['Diwali'][year]] - lo
        ax.axvline(d, color=YUG, linewidth=2)
        ax.text(d + 1, ax.get_ylim()[1] * 0.75, f"Diwali {cal['Diwali'][year]:%d %b}", color=YUG, fontsize=8.5)
        prev = mc.IDX[cal['Diwali'][year - 1]] + 364 - lo
        if 0 <= prev < 92:
            ax.axvline(prev, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_ylabel(str(year), rotation=0, ha='right', va='center', color=INK)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.grid(axis='y', color=GRID)
    axes[-1].set_xticks([0, 30, 61, 91])
    axes[-1].set_xticklabels(['1 Sep', '1 Oct', '1 Nov', '1 Dec'])
    fig.text(0.01, 0.5, 'Latent kaju katli demand (kg/day)', rotation=90, va='center', color=INK2, fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.08, hspace=0.25)
    save(fig, 'fig1-festival-shifts', 'Diwali moves: the same simulated shop, five festival seasons',
         'One simulated shop (seed 0). Blue line: Diwali; dotted grey: last year’s Diwali date. 2025→2026 and 2022→2023 both move +19 days.')


def fig2_example(cal):
    jobs = [j for j in mc.build_jobs(cal) if j['key'].startswith('base|0|2026|10-01|')]
    allj = mc.build_jobs(cal)
    F, _ = ev.load_forecasts(allj, [HERE / 'runs/local-baselines', HERE / 'runs/kaggle-01/output/mithai'])
    i = next(k for k, j in enumerate(allj) if j['key'] == 'base|0|2026|10-01|kaju-katli')
    j = allj[i]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(mc.HORIZON)
    ax.plot(x, j['truth'], color=INK, linewidth=2.2, label='Actual demand (simulated)')
    for m, (c, ls) in LINES.items():
        if m in F:
            ax.plot(x, F[m][i, :, 4], ls, color=c, linewidth=1.8, label=m)
    d = mc.IDX[cal['Diwali'][2026]] - j['ci']
    ly = mc.IDX[cal['Diwali'][2025]] + 364 - j['ci']
    ax.axvline(d, color=YUG, linewidth=1, alpha=0.6)
    ax.axvline(ly, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    top = ax.get_ylim()[1]
    ax.text(d + 0.5, top * 0.92, 'Diwali 8 Nov 2026', fontsize=8.5, color=YUG)
    ax.text(ly + 0.5, top * 0.92, 'last year’s date', fontsize=8.5, color=MUTED)
    ax.set_xticks([0, 14, 31, 45, 55])
    ax.set_xticklabels(['1 Oct', '15 Oct', '1 Nov', '15 Nov', '25 Nov'])
    ax.set_ylabel('Kaju katli, kg/day')
    ax.grid(axis='y', color=GRID)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8.5, ncol=6)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.12)
    save(fig, 'fig2-forecast-example', 'One shop, one forecast: planning kaju katli on 1 October 2026',
         'Median forecasts made with data up to 30 Sep 2026 (seed 0, illustrative — not representative of all 30 shops).')


def fig3_wape(res):
    a, b = method_order(res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, s, head in zip(axes, ('base: ordinary periods', 'base: festival season (all years)'), ('Ordinary periods', 'Festival season')):
        st = res['strata'][s]
        rows = [(label(m), m, st['methods'][m]['wape_56']) for m in a + b]
        forest(ax, rows, 'WAPE over 56 days, % (lower is better)', scale=100, fmtstr='{:.0f}')
        ax.set_title(f"{head}  ({st['n_jobs']} forecasts)", loc='left', fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.12, wspace=0.1)
    save(fig, 'fig3-wape', 'Forecast error: history-only [A] vs calendar-informed [B] methods',
         'Pooled WAPE, 56-day horizon, latent demand. Bars: 95% bootstrap over 30 simulated shops. Hollow markers: calendar-informed.')


def fig4_peak(res):
    a, b = method_order(res)
    s = 'base: festival season, +19-day shift years (2023, 2026)'
    st = res['strata'][s]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    rows = [(label(m), m, st['methods'][m].get('peak_median')) for m in a + b]
    forest(axes[0], rows, 'Median peak-timing error, days (negative = too early)', ref=0, fmtstr='{:.0f}')
    axes[0].set_title("Peak timing (jobs with a distinct actual and forecast peak)", loc='left', fontsize=10.5, color=INK)
    rows = [(label(m), m, st['methods'][m].get('phantom_rel_median')) for m in a + b]
    forest(axes[1], rows, 'Relative forecast excess on last year’s Diwali dates, %', scale=100, ref=0, fmtstr='{:+.0f}')
    axes[1].set_title(f"Bias-adjusted “phantom surge” (n={rows[0][2]['n']} per method)", loc='left', fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.12, wspace=0.1)
    save(fig, 'fig4-peak-timing', 'The Diwali trap in +19-day years (2023, 2026): early peaks and phantom surges',
         'Diwali moved 19 days later. Right: forecast/actual ratio on last year’s dates ÷ same ratio over the horizon (post-hoc; flat ETS ≈ +10%). Medians, 95% bootstrap.')


def fig5_waste(res):
    a, b = method_order(res)
    st = res['strata']['base: festival season (all years)']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    forest(axes[0], [(label(m), m, st['methods'][m]['waste_share']) for m in a + b], 'Waste, % of kg produced', scale=100)
    axes[0].set_title('Waste (expired stock)', loc='left', fontsize=10.5, color=INK)
    forest(axes[1], [(label(m), m, st['methods'][m]['lost_share']) for m in a + b], 'Lost sales, % of kg demanded', scale=100)
    axes[1].set_title('Unmet demand', loc='left', fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.12, wspace=0.1)
    save(fig, 'fig5-waste-stockout', 'Simulated production outcomes, festival season',
         f"Same order-up-to policy, capacity and shelf lives for every method; {st['n_jobs']} 56-day plans. 95% bootstrap over shops.")


def fig6_contribution(res):
    a, b = method_order(res)
    st = res['strata']['base: festival season (all years)']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    forest(axes[0], [(label(m), m, st['methods'][m]['regret_per_job']) for m in a + b],
           'Regret vs hindsight reference, ₹ per 56-day product plan', fmtstr='{:,.0f}')
    axes[0].set_title('Regret (lower is better)', loc='left', fontsize=10.5, color=INK)
    forest(axes[1], [(label(m), m, st['methods'][m]['fva_vs_shopkeeper_per_job']) for m in a + b],
           'Gross contribution minus shopkeeper’s, ₹ per plan', ref=0, fmtstr='{:,.0f}')
    axes[1].set_title('Value vs the shopkeeper heuristic', loc='left', fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.12, wspace=0.1)
    save(fig, 'fig6-contribution', 'Simulated gross contribution by forecasting strategy, festival season',
         'Simulated business outcomes, not documented savings. ₹ per product per 56-day plan; 95% bootstrap over simulated shops.')


def fig7_sensitivity(res):
    a, b = method_order(res)
    sens = res['sensitivity_festival_season']
    labels = list(sens)
    methods = a + b
    M = np.array([[sens[l][m]['regret_per_job']['point'] for l in labels] for m in methods])
    rank = np.argsort(np.argsort(M, axis=0), axis=0) + 1
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.imshow(rank, cmap='Blues_r', aspect='auto', vmin=1, vmax=len(methods) + 3)
    for i in range(len(methods)):
        for k in range(len(labels)):
            ax.text(k, i, f"{M[i, k]:,.0f}\n#{rank[i, k]}", ha='center', va='center', fontsize=7.5,
                    color=SURFACE if rank[i, k] <= 3 else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([label(m) for m in methods], fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.82, bottom=0.17)
    save(fig, 'fig7-sensitivity', 'Does the ranking survive different economics? Regret by assumption',
         'Regret ₹ per product-plan and rank (#1 = lowest regret), festival season, one assumption changed at a time. Darker = better rank.')


if __name__ == '__main__':
    cal = mc.load_calendar(HERE / 'data/festivals.json')
    res = json.loads((HERE / 'results/metrics.json').read_text())
    fig1_shifts(cal)
    fig2_example(cal)
    fig3_wape(res)
    fig4_peak(res)
    fig5_waste(res)
    fig6_contribution(res)
    fig7_sensitivity(res)
