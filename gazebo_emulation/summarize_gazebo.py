"""Summarise the disturbance sweep of the Gazebo case study.

    python3 summarize_gazebo.py [results/gazebo]

Reads results/gazebo/<condition>/rendezvous_metrics_*.csv, wind_*.csv and
launch.log, prints one row per condition and writes summary.csv plus a
LaTeX table body (summary_table.tex) for the paper.
"""

import csv
import glob
import os
import re
import statistics as st
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'results', 'gazebo')
ORDER = ['calm', 'wind3', 'wind6_gust2', 'wind9_gust3', 'gnss05', 'drop20', 'combined']
LABEL = {
    'calm': 'Nominal (calm, RTK)',
    'wind3': 'Wind 3 m/s steady',
    'wind6_gust2': 'Wind 6 m/s, gusts $\\sigma$=2',
    'wind9_gust3': 'Wind 9 m/s, gusts $\\sigma$=3',
    'gnss05': 'GNSS noise $\\sigma$=0.5 m',
    'drop20': 'Telemetry loss 20\\%',
    'combined': 'Wind 6/2 + GNSS 0.5 + loss 10\\%',
}


def ms(vals, nd=1):
    if not vals:
        return 'n/a'
    return f'{st.mean(vals):.{nd}f} $\\pm$ {st.stdev(vals) if len(vals) > 1 else 0:.{nd}f}'


def main():
    rows_out = []
    conds = [c for c in ORDER if os.path.isdir(os.path.join(ROOT, c))]
    conds += sorted(c for c in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, c)) and c not in conds)
    print(f"{'condition':<14}{'cycles':>7}{'approach s':>18}{'hover mean cm':>15}{'p95 cm':>9}{'max cm':>9}"
          f"{'<10cm %':>9}{'losses':>8}{'cycle s':>16}{'strand':>8}{'wind m/s':>10}")
    for c in conds:
        d = os.path.join(ROOT, c)
        cyc = []
        for f in glob.glob(os.path.join(d, 'rendezvous_metrics_*.csv')):
            cyc += list(csv.DictReader(open(f)))
        if not cyc:
            continue
        approach = [float(r['approach_time_s']) for r in cyc]
        hover = [100 * float(r['hover_mean_err_m']) for r in cyc]
        p95 = [100 * float(r.get('hover_p95_err_m', r['hover_max_err_m'])) for r in cyc]
        hmax = [100 * float(r['hover_max_err_m']) for r in cyc]
        tol = [100 * float(r.get('frac_within_tol', 0)) for r in cyc]
        losses = sum(int(r.get('capture_losses', 0)) for r in cyc)
        total = [float(r['cycle_total_s']) for r in cyc]
        log = open(os.path.join(d, 'launch.log')).read() if os.path.exists(os.path.join(d, 'launch.log')) else ''
        strands = len(re.findall(r'STRANDED', log))
        dispatches = len(re.findall(r'DISPATCH ', log))
        wind = 'calm'
        wf = os.path.join(d, f'wind_{c}.csv')
        if os.path.exists(wf):
            w = [float(r['speed']) for r in csv.DictReader(open(wf))]
            if w:
                wind = f'{st.mean(w):.1f} (max {max(w):.1f})'
        print(f"{c:<14}{len(cyc):>7}{ms(approach):>18}{ms(hover):>15}{ms(p95, 1):>9}{max(hmax):>9.0f}"
              f"{st.mean(tol):>9.0f}{losses:>8}{ms(total):>16}{strands:>8}{wind:>10}")
        rows_out.append({
            'condition': c, 'label': LABEL.get(c, c), 'cycles': len(cyc), 'dispatches': dispatches,
            'approach_mean_s': round(st.mean(approach), 2), 'approach_std_s': round(st.stdev(approach) if len(approach) > 1 else 0, 2),
            'hover_mean_cm': round(st.mean(hover), 1), 'hover_p95_cm': round(st.mean(p95), 1), 'hover_max_cm': round(max(hmax), 1),
            'within_tol_pct': round(st.mean(tol), 1), 'capture_losses': losses,
            'cycle_mean_s': round(st.mean(total), 1), 'cycle_std_s': round(st.stdev(total) if len(total) > 1 else 0, 1),
            'strandings': strands, 'wind': wind,
        })
    with open(os.path.join(ROOT, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    with open(os.path.join(ROOT, 'summary_table.tex'), 'w') as f:
        for r in rows_out:
            f.write(f"{r['label']} & {r['cycles']} & ${r['approach_mean_s']:.1f} \\pm {r['approach_std_s']:.1f}$ & "
                    f"${r['hover_mean_cm']:.1f}$ & ${r['hover_p95_cm']:.1f}$ & ${r['hover_max_cm']:.0f}$ & "
                    f"{r['within_tol_pct']:.0f} & {r['capture_losses']} & ${r['cycle_mean_s']:.1f} \\pm {r['cycle_std_s']:.1f}$ & {r['strandings']} \\\\\n")
    print(f'\nwrote {ROOT}/summary.csv and summary_table.tex')


if __name__ == '__main__':
    main()
