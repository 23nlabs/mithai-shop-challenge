"""Robustness v2 figures (from results/robustness-v2.json only) → robustness/v2/figures/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

V2 = Path(__file__).resolve().parents[1]
STUDY = V2.parents[1]
sys.path.insert(0, str(STUDY / 'src'))
sys.path.insert(0, str(STUDY / 'src'))
import figures as f0  # noqa: E402  (original figure helpers, unchanged)
from make_figures import SURFACE, INK, INK2, MUTED, GRID, style  # noqa: E402

f0.FIG = V2 / 'figures'
f0.SOURCE = '23N Labs · Mithai Shop Challenge · robustness v2 · SYNTHETIC simulated retailer (not real sales)'
YUG, OTHER = f0.YUG, f0.OTHER


def order(st):
    return [m for m in st['methods'] if m != 'Hindsight reference']


def fig8_calendar_gain(r):
    g = r['P2_calendar_gain']
    strata = ['base (seeds 0-19): festival season', 'R1-shape: festival season', 'R2-shocks: festival season', 'R3-both: festival season']
    titles = ['Base (calendar-aligned)', 'R1 varied festival shapes', 'R2 festival-independent shocks', 'R3 both']
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=True)
    for ax, s, t in zip(axes, strata, titles):
        rows = [(k.replace(' → ', ' →\n'), k, v) for k, v in g[s].items()]
        f0.forest(ax, [(lbl, k if k.startswith('Yug') else 'x', v) for lbl, k, v in rows], 'WAPE reduction, points', scale=100, ref=0, fmtstr='{:+.1f}')
        ax.set_title(t, loc='left', fontsize=10, color=INK)
        ax.tick_params(axis='y', labelsize=8)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.74, bottom=0.14, wspace=0.12)
    f0.save(fig, 'fig8-calendar-gain', 'Does the calendar advantage survive less calendar-friendly demand?',
            'WAPE(history-only) − WAPE(with calendar), festival season, 56 days; paired within shops, 95% bootstrap. Positive = calendar helps.')


def fig9_policy(r):
    st = r['P1_policy_fairness']['base: festival season (all years)']
    ms = order(st)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
    f0.forest(axes[0], [(m, m, st['methods'][m]['regret_per_job']) for m in ms], 'Regret, ₹ per 56-day product plan', fmtstr='{:,.0f}')
    axes[0].set_title('Committed plan: regret (lower is better)', loc='left', fontsize=10.5, color=INK)
    f0.forest(axes[1], [(m, m, st['methods'][m]['lost_share']) for m in ms], 'Lost sales, % of demand', scale=100)
    axes[1].set_title('Unmet demand', loc='left', fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.78, bottom=0.11, wspace=0.08)
    f0.save(fig, 'fig9-policy-fairness', 'Fair decision policy: shopkeeper with calibrated quantiles, and a censoring-aware baseline',
            f"Base festival season, {st['n_jobs']} plans. Original shopkeeper = ×1.10 point rule; calibrated = same τ policy as all models.")


def fig10_rolling(r):
    P3 = r['P3_rolling']
    ms = order(P3['rolling'])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for y, m in enumerate(ms):
        a = P3['rolling']['methods'][m]['regret_per_job']
        b = P3['committed_same_jobs']['methods'][m]['regret_per_job']
        c = YUG if m.startswith('Yug') else OTHER
        ax.plot([b['point'], a['point']], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.plot([b['lo'], b['hi']], [y + 0.18, y + 0.18], color=c, alpha=0.5, linewidth=1.5)
        ax.plot([a['lo'], a['hi']], [y - 0.18, y - 0.18], color=c, linewidth=1.5)
        ax.plot(b['point'], y + 0.18, 'o', color=SURFACE, markeredgecolor=c, markersize=7, markeredgewidth=2)
        ax.plot(a['point'], y - 0.18, 'o', color=c, markersize=7, markeredgecolor=SURFACE)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Regret vs hindsight reference under the SAME planning mode, ₹ per 56-day product plan')
    ax.plot([], [], 'o', color=OTHER, label='rolling weekly replanning (filled)')
    ax.plot([], [], 'o', color=SURFACE, markeredgecolor=OTHER, markeredgewidth=2, label='56-day committed plan (hollow)')
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    style(ax)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.82, bottom=0.11)
    f0.save(fig, 'fig10-rolling-vs-committed', 'Weekly replanning with procurement lead times vs a committed 56-day plan',
            f"Base festival season, {P3['rolling']['n_jobs']} plans (20 shops); Yug-1 rows use sampling seed 67067 only (the most favourable of 5, see fig 11). Each mode has its own hindsight reference.")


def fig11_seeds(r):
    P4 = r['P4_yug_seeds']
    P1 = r['P1_policy_fairness']
    panels = [('base: festival season (all years)', 'wape_56', 'Festival WAPE %', 100),
              ('base: ordinary periods', 'wape_56', 'Ordinary WAPE %', 100),
              ('base: festival season (all years)', 'regret', 'Festival regret ₹/plan', 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (s, k, lab, sc) in zip(axes, panels):
        for y, name in enumerate(('Yug-1', 'Yug-1 + calendar')):
            vals = [P4[seed][f'{name} | {s} | {k}'] * sc for seed in P4]
            ax.plot(vals, [y] * len(vals), 'o', color=YUG, alpha=0.75, markersize=7)
            ax.text(max(vals), y, f'  {min(vals):,.1f}–{max(vals):,.1f}' if sc == 100 else f'  {min(vals):,.0f}–{max(vals):,.0f}', va='center', fontsize=8, color=INK2)
        for name, ls in (('TimesFM 2.5', (0, (4, 3))), ('Chronos-Bolt', (0, (1, 2)))):
            key = 'wape_56' if k == 'wape_56' else 'regret_per_job'
            ax.axvline(P1[s]['methods'][name][key]['point'] * (sc if k == 'wape_56' else 1), color=OTHER, linestyle=ls, linewidth=1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Yug-1', 'Yug-1 + cal'])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(lab, loc='left', fontsize=10, color=INK)
        style(ax)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.7, bottom=0.12, wspace=0.35)
    f0.save(fig, 'fig11-yug-seeds', 'Yug-1 across five sampling seeds',
            'Each dot = one seed (67067, 1, 2, 3, 4; 100 paths). Dashed: TimesFM 2.5; dotted: Chronos-Bolt (history-only, same jobs).')


def fig12_peaks(r):
    st = r['P1_policy_fairness']['base: festival season, +19-day shift years (2023, 2026)']
    ms = order(st)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
    f0.forest(axes[0], [(m, m, st['methods'][m].get('peak_median')) for m in ms], 'Median peak-timing error, days', ref=0, fmtstr='{:+.0f}')
    axes[0].set_title('Peak timing (forecasts with a distinct peak)', loc='left', fontsize=10.5, color=INK)
    f0.forest(axes[1], [(m, m, st['methods'][m].get('distinct_peak_share')) for m in ms], 'Forecasts with a distinct peak, %', scale=100, fmtstr='{:.0f}')
    axes[1].set_title('How often a distinct peak is forecast', loc='left', fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.78, bottom=0.11, wspace=0.08)
    f0.save(fig, 'fig12-peaks-with-distinct-share', 'Peak timing must be read with how often a peak is forecast at all',
            '+19-day years (2023, 2026), festival season. Distinct actual peak required for both panels. 95% bootstrap over shops.')


if __name__ == '__main__':
    r = json.loads((V2 / 'results/robustness-v2.json').read_text())
    fig8_calendar_gain(r)
    fig9_policy(r)
    fig10_rolling(r)
    fig11_seeds(r)
    fig12_peaks(r)
