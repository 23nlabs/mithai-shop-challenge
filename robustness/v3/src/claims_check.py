"""Claims ledger: every key number quoted in the final public documents is checked against the raw result
JSON (value, rounded as printed) and must appear verbatim in the named document. Writes final/claims-ledger.json."""
import json
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
STUDY = V3.parents[1]
FINAL = STUDY / 'final'
J = {'v3': json.loads((V3 / 'results/final-v3.json').read_text())['families'],
     'add': json.loads((V3 / 'results/final-v3-addon.json').read_text()),
     'v2': json.loads((STUDY / 'robustness/v2/results/robustness-v2.json').read_text()),
     'ver': json.loads((V3 / 'results/final-verification.json').read_text())}
OF = ('v3', 'original', 'regret', 'base: festival season (all years)')
SH = ('v2', 'P1_policy_fairness', 'base: festival season, +19-day shift years (2023, 2026)', 'methods')
CG = ('v2', 'P2_calendar_gain')
PW = ('v3', 'original', 'pairwise_festival', 'bias')


def get(path):
    x = J[path[0]]
    for k in path[1:]:
        x = x[k]
    return x


def rupees(v):
    return f'{v:,.0f}'


def pct(v):
    return f'{round(v * 100):d}%'


def pts(v):
    return f'+{v * 100:.1f}'


C = []


def claim(cid, path, fmt, docs, text=None, transform=None):
    v = get(path)
    if transform:
        v = transform(v)
    s = fmt(v)
    C.append(dict(id=cid, value=v, printed=text or s, expected=s, source='/'.join(map(str, path)), docs=docs))


A, T = 'article.md', 'technical-appendix.md'
for m, lab in (('Calendar regression (censoring-aware)', 'censor'), ('Calendar regression', 'calreg'), ('TimesFM + calendar', 'tfm+cal'),
               ('Yug-1 + calendar', 'yug+cal'), ('Chronos + calendar', 'chr+cal'), ('Yug-1', 'yug'), ('TimesFM 2.5', 'tfm'), ('Chronos-Bolt', 'chr')):
    for v in ('none', 'bias'):
        claim(f'regret {lab} {v}', OF + (m, v, 'point'), rupees, [A, T])
claim('regret shopkeeper x1.10 none', OF + ('Shopkeeper', 'none', 'point'), rupees, [T])
claim('phantom yug', SH + ('Yug-1', 'phantom_rel_median', 'point'), lambda v: f'+{pct(v)}', [A, T])
claim('phantom timesfm', SH + ('TimesFM 2.5', 'phantom_rel_median', 'point'), lambda v: f'+{pct(v)}', [A, T])
claim('phantom chronos', SH + ('Chronos-Bolt', 'phantom_rel_median', 'point'), lambda v: f'+{pct(v)}', [A, T])
claim('phantom seasonal-naive', SH + ('Seasonal-naive 52w', 'phantom_rel_median', 'point'), lambda v: f'+{pct(v)}', [A, T])
claim('phantom ets', SH + ('ETS', 'phantom_rel_median', 'point'), lambda v: f'+{pct(v)}', [A, T])
claim('distinct peak yug', SH + ('Yug-1', 'distinct_peak_share', 'point'), pct, [A, T])
claim('distinct peak timesfm', SH + ('TimesFM 2.5', 'distinct_peak_share', 'point'), pct, [A, T])
claim('distinct peak chronos', SH + ('Chronos-Bolt', 'distinct_peak_share', 'point'), pct, [A, T])
claim('peak error yug', SH + ('Yug-1', 'peak_median', 'point'), lambda v: f'{int(-v)} days early', [A], text='17 days early')
for scen, key in (('base', 'base (seeds 0-19): festival season'), ('R3', 'R3-both: festival season')):
    for m, k in (('yug', 'Yug-1 → Yug-1 + calendar'), ('tfm', 'TimesFM 2.5 → TimesFM + calendar'), ('chr', 'Chronos-Bolt → Chronos + calendar')):
        claim(f'calendar gain {scen} {m}', CG + (key, k, 'point'), pts, [A, T])
claim('pair yug+cal - tfm+cal', PW + ('Yug-1 + calendar − TimesFM + calendar', 'point'), rupees, [A, T])
claim('pair yug+cal - tfm+cal lo', PW + ('Yug-1 + calendar − TimesFM + calendar', 'lo'), lambda v: rupees(-v), [A, T])
claim('pair yug+cal - tfm+cal hi', PW + ('Yug-1 + calendar − TimesFM + calendar', 'hi'), rupees, [A, T])
claim('yug - tfm after bias', ('v3', 'original', 'yug_vs_history_only', 'bias', 'Yug-1 − TimesFM 2.5', 'point'), rupees, [A, T])
claim('censor - yug+cal bias', PW + ('Calendar regression (censoring-aware) − Yug-1 + calendar', 'point'), lambda v: rupees(-v), [T])
claim('forecast rows', ('ver', 'counts'), rupees, [A, T], transform=lambda d: sum(x['rows'] for x in d.values()))
claim('hindsight contribution', ('v2', 'P1_policy_fairness', 'base: festival season (all years)', 'methods', 'Hindsight reference',
                                 'contribution_per_job', 'point'), lambda v: f'{v / 1000:.0f},000', [A], text='327,000')
seeds = J['v2']['P4_yug_seeds']
C.append(dict(id='yug festival coverage seed range', printed='68% and 80%', source='v2/P4_yug_seeds',
              expected=f"{min(s['Yug-1 | base: festival season (all years) | coverage80'] for s in seeds.values())*100:.0f}% and "
                       f"{max(s['Yug-1 | base: festival season (all years) | coverage80'] for s in seeds.values())*100:.0f}%", docs=[A]))


def main():
    texts = {d: (FINAL / d).read_text() for d in (A, T)}
    bad = []
    for c in C:
        c['value_matches_print'] = c['printed'] == c['expected']
        c['present_in'] = {d: c['printed'].replace('-', '−') in texts[d] or c['printed'] in texts[d] for d in c['docs']}
        if not c['value_matches_print'] or not all(c['present_in'].values()):
            bad.append(c)
    (FINAL / 'claims-ledger.json').write_text(json.dumps(dict(claims=C, failures=[b['id'] for b in bad]), indent=1, default=float))
    print(len(C), 'claims;', len(bad), 'failures')
    for b in bad:
        print('  FAIL', b['id'], '| printed', b['printed'], '| expected', b['expected'], '|', b['present_in'])


if __name__ == '__main__':
    main()
