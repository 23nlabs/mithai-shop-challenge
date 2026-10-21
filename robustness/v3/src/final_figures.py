"""Final publication charts → research/mithai-shop-challenge/final/figures/ (from v3 JSON + copies of v1/v2 charts)."""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

V3 = Path(__file__).resolve().parents[1]
STUDY = V3.parents[1]
FINAL = STUDY / 'final/figures'
sys.path.insert(0, str(STUDY / 'src'))
sys.path.insert(0, str(STUDY / 'src'))
import figures as f0  # noqa: E402
from make_figures import SURFACE, INK, INK2, MUTED, GRID, style  # noqa: E402

f0.FIG = FINAL
f0.SOURCE = '23N Labs · Mithai Shop Challenge · SYNTHETIC retail demand (simulated shop, not real transaction records)'
YUG, OTHER = f0.YUG, f0.OTHER
ORDER = ['Yug-1', 'TimesFM 2.5', 'Chronos-Bolt', 'ETS', 'Seasonal-naive 52w', 'Calendar regression',
         'Calendar regression (censoring-aware)', 'Shopkeeper', 'Shopkeeper (calibrated, τ policy)', 'ETS + calendar',
         'Yug-1 + calendar', 'TimesFM + calendar', 'Chronos + calendar']
LABEL = {'Shopkeeper': 'Shopkeeper (×1.10 rule)', 'Shopkeeper (calibrated, τ policy)': 'Shopkeeper (calibrated)',
         'Calendar regression (censoring-aware)': 'Calendar regr. (censoring-aware)'}


def lab(m):
    return LABEL.get(m, m)


def fig_calibration(r):
    st = r['families']['original']['regret']['base: festival season (all years)']
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for y, m in enumerate(ORDER):
        a, b = st[m]['none'], st[m]['bias']
        c = YUG if m.startswith('Yug') else OTHER
        ax.plot([a['point'], b['point']], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.plot([a['lo'], a['hi']], [y + 0.17, y + 0.17], color=c, alpha=0.45, linewidth=1.5)
        ax.plot([b['lo'], b['hi']], [y - 0.17, y - 0.17], color=c, linewidth=1.5)
        ax.plot(a['point'], y + 0.17, 'o', color=SURFACE, markeredgecolor=c, markersize=7, markeredgewidth=2)
        ax.plot(b['point'], y - 0.17, 'o', color=c, markersize=7, markeredgecolor=SURFACE)
        ax.text(max(a['hi'], b['hi']), y, f"  {a['point']/1000:.0f}k → {b['point']/1000:.0f}k", va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([lab(m) + (' (5 seeds)' if m.startswith('Yug') else '') for m in ORDER], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Regret vs hindsight reference, ₹ per 56-day product plan (simulated)')
    ax.plot([], [], 'o', color=SURFACE, markeredgecolor=OTHER, markeredgewidth=2, label='uncalibrated')
    ax.plot([], [], 'o', color=OTHER, label='training-only bias correction')
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    style(ax)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.83, bottom=0.1)
    f0.save(fig, 'F4-regret-before-after-calibration', 'Correcting each method’s historical bias changes the business ranking',
            'Base festival season, 1,080 plans, 30 shops. 95% bootstrap over shops (and over 5 sampling seeds for Yug-1).')


def fig_pairwise(r, add):
    fams = [('Original', r['families']['original']['pairwise_festival']['bias']),
            ('R1 shapes', add['bias_pairwise_calendar_dependence']['R1-shape: festival season']),
            ('R2 shocks', add['bias_pairwise_calendar_dependence']['R2-shocks: festival season']),
            ('R3 both', add['bias_pairwise_calendar_dependence']['R3-both: festival season']),
            ('Weekly replanning', add['bias_pairwise_rolling'])]
    others = ['Calendar regression (censoring-aware)', 'Calendar regression', 'TimesFM + calendar', 'Chronos + calendar',
              'ETS + calendar', 'Shopkeeper (calibrated, τ policy)', 'Shopkeeper']
    fig, axes = plt.subplots(1, len(fams), figsize=(14, 4.8), sharey=True)
    for ax, (title, d) in zip(axes, fams):
        for y, o in enumerate(others):
            k1, k2 = f'Yug-1 + calendar − {o}', f'{o} − Yug-1 + calendar'
            if k1 in d:
                v = d[k1]
            elif k2 in d:
                v = {kk: -d[k2][kk] for kk in ('point', 'lo', 'hi')}
                v['lo'], v['hi'] = v['hi'], v['lo']
            else:
                continue
            inconc = v['lo'] <= 0 <= v['hi']
            c = MUTED if inconc else (YUG if v['hi'] < 0 else OTHER)
            ax.plot([v['lo'] / 1000, v['hi'] / 1000], [y, y], color=c, linewidth=2)
            ax.plot(v['point'] / 1000, y, 'o', color=SURFACE if inconc else c, markeredgecolor=c, markersize=7, markeredgewidth=2)
        ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.set_xlabel('₹ thousand per plan')
        style(ax)
    axes[0].set_yticks(range(len(others)))
    axes[0].set_yticklabels([f'vs {lab(o)}' for o in others], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.17, right=0.99, top=0.74, bottom=0.14, wspace=0.08)
    f0.save(fig, 'F5-paired-regret-yug-calendar', 'Yug-1 + calendar vs other calendar-informed methods, after bias correction',
            'Paired regret difference (Yug-1 + cal minus other); <0 favours Yug-1 + cal. Hollow grey = inconclusive (95% CI spans 0). Shops and Yug-1 seeds resampled.')


def fig_seeds(r):
    rows = [('Original festival', r['families']['original']['regret']['base: festival season (all years)']),
            ('Stress: early gifting', r['families']['original']['regret']['stress: early-gifting']),
            ('Stress: weak festival', r['families']['original']['regret']['stress: weak-festival']),
            ('Stress: local shock', r['families']['original']['regret']['stress: local-shock']),
            ('R1 shapes', r['families']['calendar_dependence']['regret']['R1-shape: festival season']),
            ('R2 shocks', r['families']['calendar_dependence']['regret']['R2-shocks: festival season']),
            ('R3 both', r['families']['calendar_dependence']['regret']['R3-both: festival season']),
            ('Weekly replanning', r['families']['rolling']['regret']['base festival season, 20 shops'])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, m, ref in zip(axes, ('Yug-1', 'Yug-1 + calendar'), ('TimesFM 2.5', 'TimesFM + calendar')):
        for y, (name, st) in enumerate(rows):
            sr = np.array(st[m]['none']['seed_range']) / 1000
            ax.plot(sr, [y] * 5, 'o', color=YUG, alpha=0.7, markersize=6)
            ax.plot(sr[0], y, 'D', color=SURFACE, markeredgecolor=YUG, markersize=7, markeredgewidth=2)
            ax.plot(st[ref]['none']['point'] / 1000, y, '|', color=OTHER, markersize=14, markeredgewidth=2)
        ax.set_title(f'{m} (uncalibrated); grey bar = {ref}', loc='left', fontsize=10, color=INK)
        ax.set_xlabel('Regret, ₹ thousand per plan')
        style(ax)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([n for n, _ in rows])
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.15, right=0.98, top=0.78, bottom=0.12, wspace=0.08)
    f0.save(fig, 'F6-yug-sampling-seeds', 'Yug-1 business results vary with the sampling seed in every experiment family',
            'Dots: seeds 67067, 1, 2, 3, 4 (100 paths). Hollow diamond: seed 67067, used in v1/v2 — the most favourable seed in 15 of these 16 rows.')


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    r = json.loads((V3 / 'results/final-v3.json').read_text())
    add = json.loads((V3 / 'results/final-v3-addon.json').read_text())
    for src, dst in ((STUDY / 'figures/fig1-festival-shifts', 'F1-festival-shifts'),
                     (STUDY / 'robustness/v2/figures/fig12-peaks-with-distinct-share', 'F2-diwali-trap-peaks'),
                     (STUDY / 'robustness/v2/figures/fig8-calendar-gain', 'F3-calendar-gain-by-scenario')):
        for ext in ('png', 'svg'):
            shutil.copyfile(f'{src}.{ext}', FINAL / f'{dst}.{ext}')
    fig_calibration(r)
    fig_pairwise(r, add)
    fig_seeds(r)


if __name__ == '__main__':
    main()
