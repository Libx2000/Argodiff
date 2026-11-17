"""Replot observation-availability results with explicit paired relative errors."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter

HERE = Path(__file__).resolve().parent
SCENARIOS = ('nominal', 'columns50', 'outage15')
SEASONS = ('winter', 'summer')
VARIABLES = ('salinity', 'temperature')
COLORS = {'nominal': '#0072B2', 'columns50': '#D55E00', 'outage15': '#7B61A8'}
STYLES = {'nominal': '-', 'columns50': (0, (5, 2)), 'outage15': (0, (2, 1.6))}
LABELS = {'nominal': 'Nominal', 'columns50': '50% mapped columns', 'outage15': '15-day guidance gap'}


def pool(rows, metric='rmse'):
    weights = [float(r['weight']) for r in rows]
    values = [float(r[metric]) for r in rows]
    assert rows and all(w > 0 and math.isfinite(w) for w in weights)
    assert all(math.isfinite(v) for v in values)
    if metric == 'rmse':
        return math.sqrt(sum(w * v * v for w, v in zip(weights, values)) / sum(weights))
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)


def analyze(source, baseline_source, output):
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 2 * 2 * 60 * 2, len(rows)
    keys = [(r['scenario'], r['method'], r['season'], r['variable'], int(r['lead'])) for r in rows]
    assert len(set(keys)) == len(keys)
    lookup = dict(zip(keys, rows))
    with baseline_source.open(encoding='utf-8-sig', newline='') as handle:
        baseline_rows = list(csv.DictReader(handle))
    assert len(baseline_rows) == 240
    deterministic = {(r['season'],r['variable'],int(r['lead'])):r for r in baseline_rows}
    assert len(deterministic) == 240
    for scenario in SCENARIOS:
        for season in SEASONS:
            for variable in VARIABLES:
                for lead in range(1, 61):
                    target = lookup[scenario, 'guided_raw', season, variable, lead]
                    for other in (deterministic[season, variable, lead],
                                  lookup['nominal', 'guided_raw', season, variable, lead]):
                        assert all(target[f] == other[f] for f in ('date', 'count', 'weight', 'depth_index', 'domain'))
    curves = []
    summary = []
    windows = {'all': (1, 60), 'before_gap': (1, 30), 'during_gap': (31, 45), 'after_gap': (46, 60)}
    for variable in VARIABLES:
        for scenario in SCENARIOS:
            for season in ('pooled',) + SEASONS:
                for lead in range(1, 61):
                    chosen = SEASONS if season == 'pooled' else (season,)
                    matched = lambda sc, method: [lookup[sc, method, s, variable, lead] for s in chosen]
                    guided = pool(matched(scenario, 'guided_raw'))
                    control = pool([deterministic[s, variable, lead] for s in chosen])
                    nominal = pool(matched('nominal', 'guided_raw'))
                    assert min(guided, control, nominal) > 0
                    curves.append({'variable': variable, 'scenario': scenario, 'start': season, 'lead': lead,
                        'guided_rmse': guided, 'deterministic_rmse': control, 'nominal_guided_rmse': nominal,
                        'reduction_vs_deterministic_percent': 100 * (1 - guided / control),
                        'increase_vs_nominal_percent': 100 * (guided / nominal - 1)})
            for window, (lo, hi) in windows.items():
                chosen = [r for r in rows if r['scenario'] == scenario and r['variable'] == variable and lo <= int(r['lead']) <= hi]
                guided = pool([r for r in chosen if r['method'] == 'guided_raw'])
                control = pool([r for r in baseline_rows if r['variable'] == variable and lo <= int(r['lead']) <= hi])
                nominal = pool([r for r in rows if r['scenario'] == 'nominal' and r['method'] == 'guided_raw' and r['variable'] == variable and lo <= int(r['lead']) <= hi])
                summary.append({'scenario': scenario, 'variable': variable, 'window': window,
                    'guided_rmse': guided, 'deterministic_rmse': control,
                    'reduction_vs_deterministic_percent': 100 * (1 - guided / control),
                    'increase_vs_nominal_percent': 100 * (guided / nominal - 1),
                    'coverage90_percent': 100 * pool([r for r in chosen if r['method'] == 'guided_raw'], 'coverage90')})
    with (output / 'Figure8_plot_values.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    with (output / 'Figure8_summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    ranges = {}
    for variable in VARIABLES:
        for scenario in SCENARIOS:
            selected = [r for r in curves if r['variable'] == variable and r['scenario'] == scenario and r['start'] == 'pooled']
            ranges[f'{variable}/{scenario}'] = {
                'days_with_lower_rmse_than_deterministic': sum(r['reduction_vs_deterministic_percent'] > 0 for r in selected),
                'skill_min_max_percent': [min(r['reduction_vs_deterministic_percent'] for r in selected), max(r['reduction_vs_deterministic_percent'] for r in selected)],
                'sensitivity_min_max_percent': [min(r['increase_vs_nominal_percent'] for r in selected), max(r['increase_vs_nominal_percent'] for r in selected)]}
    report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_rows': len(rows), 'guidance_rows_used':720, 'deterministic_rows':len(baseline_rows),
        'baseline_sha256':hashlib.sha256(baseline_source.read_bytes()).hexdigest(),
        'cases':8, 'guidance_cases':6, 'shared_deterministic_controls':2, 'days_per_case':60,
        'case_frequencies': {str(k): v for k, v in Counter((r['scenario'], r['method'], r['season']) for r in rows).items()},
        'exclusions': [], 'comparator_change':'Six nudging runs replaced in the main comparison by two matched deterministic controls, each reused across three availability scenarios. Original source rows remain archived.',
        'paired_support': 'Date, count, latitude weight, depth selection and verification domain match exactly; native-grid and full-mask checks are recorded in deterministic_baseline_provenance.json.',
        'aggregation': 'Pool weighted squared errors across the two fixed starts before taking the square root.',
        'uncertainty': 'Bands span the two fixed-start descriptive curves, not confidence intervals or ensemble intervals.',
        'ranges': ranges, 'summary': summary}
    (output / 'Figure8_analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return curves, report


def draw(curves, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    from matplotlib.transforms import ScaledTranslation
    from audit_panel_alignment import require_matplotlib_panel_alignment

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'axes.titleweight': 'bold', 'axes.labelweight': 'bold', 'axes.linewidth': 1,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.145, right=0.975, bottom=0.10, top=0.87, wspace=0.35, hspace=0.43)
    days = np.arange(1, 61)
    panels = []
    for row in range(2):
        for col, variable in enumerate(VARIABLES):
            ax = axes[row, col]
            field = 'reduction_vs_deterministic_percent' if row == 0 else 'increase_vs_nominal_percent'
            scenarios = SCENARIOS if row == 0 else SCENARIOS[1:]
            extents = [0.0]
            for scenario in scenarios:
                collect = lambda start: np.array([r[field] for r in curves if r['variable'] == variable and r['scenario'] == scenario and r['start'] == start])
                center = collect('pooled')
                seasonal = np.stack([collect(s) for s in SEASONS])
                assert center.shape == (60,) and seasonal.shape == (2, 60)
                lower, upper = seasonal.min(axis=0), seasonal.max(axis=0)
                ax.fill_between(days, lower, upper, color=COLORS[scenario], alpha=0.075, linewidth=0, zorder=1)
                ax.plot(days, center, color=COLORS[scenario], ls=STYLES[scenario], lw=1.8, label=LABELS[scenario], zorder=4)
                extents.extend(lower.tolist() + upper.tolist() + center.tolist())
            ax.axvspan(30.5, 45.5, facecolor='#777777', alpha=0.09, edgecolor='none', zorder=0)
            ax.axhline(0, color='#525252', ls=(0, (2, 2)), lw=0.8, zorder=2)
            span = max(extents) - min(extents)
            pad = max(0.08 * span, 0.01)
            ax.set_ylim(min(extents) - pad, max(extents) + pad)
            ax.set_xlim(1, 60)
            ax.set_xticks([1, 15, 30, 45, 60])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if row == 1 else '%.1f'))
            ax.tick_params(direction='out', width=0.9, length=3, pad=3)
            ax.grid(axis='y', color='#E5E5E5', linewidth=0.5, zorder=0)
            ax.set_title(variable.capitalize() + (' skill' if row == 0 else ' sensitivity'), pad=10)
            if col == 0:
                ax.set_ylabel('RMSE reduction\nvs Deterministic (%)' if row == 0 else 'RMSE change\nvs nominal Argo-Diff (%)', labelpad=7)
            if row == 1:
                ax.set_xlabel('Cycle day', labelpad=6)
            letter = 'abcd'[row * 2 + col]
            ax.text(0, 1, letter, transform=ax.transAxes + ScaledTranslation(-22/72, 10/72, fig.dpi_scale_trans),
                    ha='left', va='bottom', fontsize=11, fontweight='bold')
            panels.append({'panel': letter, 'variable': variable, 'metric': field,
                'full_signed_range_displayed': True, 'uncertainty': 'two fixed-start range, not CI',
                'y_limits': list(ax.get_ylim())})
    handles = [Line2D([], [], color=COLORS[s], ls=STYLES[s], lw=1.8, label=LABELS[s]) for s in SCENARIOS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.982),
               ncol=3, frameon=False, handlelength=2.4, columnspacing=1.7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=output/'Figure8.alignment.json',
        overlay_svg=output/'Figure8.alignment.svg', tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(output/'Figure8.pdf', dpi=300)
    fig.savefig(output/'Figure8.svg', dpi=300)
    fig.savefig(output/'Figure8.png', dpi=300)
    (output/'Figure8_panels.json').write_text(json.dumps(panels, indent=2)+'\n', encoding='utf-8')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE/'source_data'/'availability_20261116'/'quick_daily.csv')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--baseline', type=Path, default=HERE/'source_data'/'availability_20261116'/'deterministic_daily_60.csv')
    parser.add_argument('--deps', type=Path)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.deps:
        sys.path.insert(0, str(args.deps))
    sys.path.insert(0, str(HERE/'figure8_support'))
    curves, report = analyze(args.source, args.baseline, args.output_dir)
    if not args.analyze_only:
        draw(curves, args.output_dir)
    print(json.dumps({'source_rows': report['source_rows'], 'ranges': report['ranges'],
        'pooled_60day': [r for r in report['summary'] if r['window'] == 'all']}, indent=2))


if __name__ == '__main__':
    main()
