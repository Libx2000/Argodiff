"""Reproduce archived Figure 2 values, adding explicitly labelled late-cycle zooms."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict

HERE = Path(__file__).resolve().parent
LOCAL_DEPS = HERE.parent / 'ems_results_plot_deps'
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

PROJECT = Path(r'C:\Users\Administrator\Desktop\EMS\els-cas-templates')
BLOCKS = {('winter', '20261006'), ('winter', '20261017'),
          ('spring', '20261006'), ('spring', '20261017'), ('summer', '20261006')}
METHODS = ('deterministic', 'guided_raw', 'selected', 'nudging_selected')
LABEL = dict(zip(METHODS, ('Deterministic', 'Argo-Diff', 'SA-Diff', 'Nudging')))
COLOR = dict(zip(METHODS, ('#222222', '#0072B2', '#D55E00', '#009E73')))
LINE = dict(zip(METHODS, ('--', '-', '-.', ':')))
MARKER = dict(zip(METHODS, ('s', 'o', '^', 'D')))
PANELS = (
    ('temperature', 'rmse', 'Temperature RMSE', r'$^{\circ}$C', 1.0,
     METHODS, 'All methods'),
    ('salinity', 'rmse', 'Salinity RMSE', 'Practical Salinity', 1.0,
     METHODS[:3], 'Det. / Argo / SA'),
    ('rho', 'crps', 'Density CRPS', r'kg m$^{-3}$', 1.0,
     ('guided_raw', 'selected'), 'Argo / SA'),
    ('n2', 'crps', r'$N^2$ CRPS', r'$10^{-5}$ s$^{-2}$', 1e5,
     ('deterministic', 'nudging_selected'), 'Det. / Nudging'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def prepare_source():
    source = PROJECT / 'source_data/validation_daily.csv'
    manifest = json.loads((PROJECT / 'figure_manifest_20261026.json').read_text())
    original = next(row for row in manifest['figures'] if row['figure'] == 'fig2_cycles')
    expected = next(value for key, value in original['sources'].items() if key.replace('\\', '/').endswith('validation_daily.csv'))
    assert sha(source) == expected, 'Archived daily source changed'
    groups = defaultdict(list)
    for row in readcsv(source):
        if (row['method'] in METHODS and (row['season'], row['seed']) in BLOCKS
                and row['candidate'] in ('raw', 'point')
                and row['domain'] == 'glorys' and row['support'] == 'common'
                and row['region'] == 'global' and row['depth_index'] == '-1'
                and row['variable'] in {panel[0] for panel in PANELS}):
            groups[row['method'], row['variable'], int(row['lead'])].append(row)
    output = []
    for variable, metric, title, unit, scale, zoom_methods, zoom_label in PANELS:
        for method in METHODS:
            for day in range(1, 91):
                rows = groups[method, variable, day]
                assert len(rows) == 5 and {(r['season'], r['seed']) for r in rows} == BLOCKS
                seeds = defaultdict(set)
                for row in rows:
                    seeds[row['season']].add(row['seed'])
                weights = np.array([float(row['weight']) / len(seeds[row['season']]) for row in rows])
                values = np.array([float(row[metric]) for row in rows])
                assert np.all(np.isfinite(values)) and np.all(weights > 0)
                value = float(np.sqrt(np.average(values ** 2, weights=weights)) if metric == 'rmse'
                              else np.average(values, weights=weights))
                output.append(dict(variable=variable, metric=metric, method=method, cycle_day=day,
                    value=value, display_scale=scale, blocks=5, starts=3))
    check = {(row['method'], row['variable'], int(row['cycle_day'])): float(row['rmse'])
             for row in readcsv(PROJECT / 'cycling_error_check_20261108.csv')}
    differences = [abs(row['value'] - check[row['method'], row['variable'], row['cycle_day']])
        for row in output if row['variable'] in ('temperature', 'salinity')
        and (row['method'], row['variable'], row['cycle_day']) in check]
    assert max(differences) < 1e-12
    dest = HERE / 'Figure2_source.csv'
    with dest.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    provenance = dict(raw_source=str(source), raw_sha256=sha(source),
        original_figure_manifest_sha256=sha(PROJECT / 'figure_manifest_20261026.json'),
        support='global GLORYS common finite support, depth_index=-1; five fixed season-seed blocks',
        blocks=[dict(season=s, seed=int(seed)) for s, seed in sorted(BLOCKS)],
        estimator='Divide daily spatial weights by seed count within each start; pool MSE then root for RMSE, weighted mean for CRPS.',
        reproduced_archived_ts_max_abs_difference=max(differences),
        source_rows=len(output), human_verified=False)
    (HERE / 'Figure2_source_audit.json').write_text(json.dumps(provenance, indent=2) + '\n')


def plot(force=False):
    outputs = [HERE / 'Figure2.pdf', HERE / 'Figure2.png', HERE / 'Figure2_manifest_20261112.json']
    if not force:
        assert not any(path.exists() for path in outputs), 'Use --force to replace generated figure outputs'
    rows = readcsv(HERE / 'Figure2_source.csv')
    curves = {(variable, method): np.array([float(row['value']) for row in sorted(
        (r for r in rows if r['variable'] == variable and r['method'] == method),
        key=lambda r: int(r['cycle_day']))]) for variable, *_ in PANELS for method in METHODS}
    assert all(len(values) == 90 and np.all(np.isfinite(values)) for values in curves.values())
    days = np.arange(1, 91)
    zoom_records = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8.5,
            'axes.titlesize': 9.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
            'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
            'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}):
        fig = plt.figure(figsize=(7.2, 6.2), layout='constrained')
        grid = fig.add_gridspec(3, 2, height_ratios=(.13, 1, 1), hspace=.08, wspace=.06)
        legend_ax = fig.add_subplot(grid[0, :])
        legend_ax.axis('off')
        handles = [Line2D([], [], color=COLOR[m], ls=LINE[m], marker=MARKER[m],
            markerfacecolor='white', markersize=3.2, linewidth=1.35, label=LABEL[m]) for m in METHODS]
        legend_ax.legend(handles=handles, loc='center', ncol=4, frameon=False,
            fontsize=8, handlelength=2.8, columnspacing=1.6)
        for index, (variable, metric, title, unit, scale, zoom_methods, zoom_label) in enumerate(PANELS):
            ax = fig.add_subplot(grid[1 + index // 2, index % 2])
            for method in METHODS:
                ax.plot(days, curves[variable, method] * scale, color=COLOR[method],
                    ls=LINE[method], lw=1.25, marker=MARKER[method], markevery=10,
                    markersize=2.7, markerfacecolor='white', markeredgewidth=.7)
            ymax = max(np.max(curves[variable, m] * scale) for m in METHODS)
            ax.set(title=title, xlabel='Cycle day', ylabel=unit, xlim=(1, 90), ylim=(0, ymax * 1.055),
                   xticks=(1, 30, 60, 90))
            ax.grid(alpha=.18, linewidth=.55)
            ax.text(-.14, 1.055, 'abcd'[index], transform=ax.transAxes, fontsize=10, fontweight='bold')
            inset_bounds = [.52, .085, .455, .34] if variable == 'n2' else [.52, .115, .455, .395]
            inset = ax.inset_axes(inset_bounds)
            selected = np.concatenate([curves[variable, m][79:90] * scale for m in zoom_methods])
            span = float(selected.max() - selected.min())
            padding = max(span * .15, max(abs(selected)) * .0005)
            low, high = float(selected.min() - padding), float(selected.max() + padding)
            # The global panels retain every method; each inset identifies its focused group.
            for method in zoom_methods:
                inset.plot(days, curves[variable, method] * scale, color=COLOR[method],
                    ls=LINE[method], lw=.95, marker=MARKER[method], markevery=2,
                    markersize=2.4, markerfacecolor='white', markeredgewidth=.65)
            inset.set(xlim=(80, 90), ylim=(low, high), xticks=(80, 85, 90))
            inset.set_title('Days 80-90\n' + zoom_label, fontsize=6.5, pad=2)
            inset.tick_params(labelsize=6.4, length=2, pad=1.5)
            inset.yaxis.set_major_locator(MaxNLocator(3))
            inset.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if variable in ('salinity', 'rho') else '%.2f'))
            inset.grid(alpha=.15, linewidth=.4)
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(.55)
                spine.set_color('#666666')
            ax.indicate_inset_zoom(inset, edgecolor='#666666', linewidth=.6, alpha=.65)
            zoom_records.append(dict(panel='abcd'[index], variable=variable, cycle_days=[80, 90],
                methods=list(zoom_methods), inset_axes_bounds=inset_bounds,
                displayed_y_limits=[low, high], all_methods_in_main_panel=True))
        fig.savefig(HERE / 'Figure2.pdf', dpi=300,
            metadata={'Title': 'Figure 2: 90-day ocean cycling with late-cycle zooms',
                      'CreationDate': None, 'ModDate': None})
        fig.savefig(HERE / 'Figure2.png', dpi=300)
        plt.close(fig)
    manifest = dict(source=dict(path='Figure2_source.csv', sha256=sha(HERE / 'Figure2_source.csv')),
        source_audit=dict(path='Figure2_source_audit.json', sha256=sha(HERE / 'Figure2_source_audit.json')),
        code=dict(path=Path(__file__).name, sha256=sha(__file__)),
        outputs=[dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size) for path in outputs[:2]],
        dimensions_inches=[7.2, 6.2], export_dpi=300, pdf_content='Vector lines, markers and embedded TrueType text; DPI only applies to raster content.',
        png_pixels=[2160, 1860], zooms=zoom_records, full_cycle=[1, 90],
        palette=COLOR, line_styles=LINE, markers=MARKER, smoothing=False, offsets=False,
        missing_data='All 90 days complete; no interpolation or imputation.',
        uncertainty='No new error bars; original descriptive pooled curves retained.',
        alt_text='Four full-cycle panels compare temperature/salinity RMSE and density/stratification CRPS. Labelled insets enlarge days 80-90 for overlapping method groups; main panels retain all four methods.',
        human_verified=False)
    (HERE / 'Figure2_manifest_20261112.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(pdf=str(HERE / 'Figure2.pdf'), png=str(HERE / 'Figure2.png'),
                         curves=16, days_per_curve=90, source_rows=len(rows), export_dpi=300), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-source', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.prepare_source:
        prepare_source()
    plot(args.force)
