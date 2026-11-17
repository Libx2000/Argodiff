"""Data-preserving redesign of the manuscript's physical and calibration figure."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
LOCAL_DEPS = HERE.parent / 'ems_results_plot_deps'
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

METHODS = ('deterministic', 'guided_raw', 'selected')
LABELS = dict(zip(METHODS, ('Deterministic', 'Argo-Diff', 'SA-Diff')))
COLORS = dict(zip(METHODS, ('#222222', '#0072B2', '#D55E00')))
MARKERS = dict(zip(METHODS, ('s', 'o', '^')))
VARIABLES = ('temperature', 'salinity', 'rho', 'n2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def unique(rows, **filters):
    result = [row for row in rows if all(row[key] == value for key, value in filters.items())]
    if len(result) != 1:
        raise ValueError(f'Expected exactly one archived row for {filters}, found {len(result)}')
    return result[0]


def prepare_source():
    manifest = json.loads((HERE / 'figure_manifest_20261026.json').read_text())
    archived = next(row for row in manifest['figures'] if row['figure'] == 'fig3_physical')
    loaded, source_records = {}, []
    for relative, expected in archived['sources'].items():
        path = HERE / relative.replace('\\', '/')
        assert sha(path) == expected, f'Archived input changed: {path}'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            loaded[path.stem] = list(csv.DictReader(stream))
        source_records.append(dict(path=relative.replace('\\', '/'), sha256=expected))
    common = dict(scope='matched_primary', domain='glorys', support='common', region='global', window='all')
    events = []
    scores = []
    for method in METHODS:
        for candidate in (('common_center',) if method == 'deterministic' else ('common_center', 'raw')):
            events.append(unique(loaded['validation_events'], **common, method=method,
                                 candidate=candidate, threshold_s2='-1e-05'))
        for variable in ('temperature', 'n2'):
            scores.append(unique(loaded['validation_scores'], **common, depth_index='-1',
                                 method=method, variable=variable, candidate='raw'))
    for variable in VARIABLES:
        for candidate in ('raw', 'marginal'):
            row = unique(loaded['validation_scores'], **common, depth_index='-1',
                         method='selected', variable=variable, candidate=candidate)
            if row not in scores:
                scores.append(row)
    assert len(events) == 5 and len(scores) == 12
    assert np.ptp([float(row['observed_frequency']) for row in events]) < 1e-12
    dump(HERE / 'Figure3_source.json', dict(
        source_files=source_records, original_manifest_sha256=sha(HERE / 'figure_manifest_20261026.json'),
        scope='Five fixed matched season-seed blocks, winter (two seeds), spring (two), summer (one); 90 cycle days each.',
        estimator='Archived global common-support weighted pooled scores; seeds averaged within starts before pooling. No new reweighting.',
        events=events, scores=scores, human_verified=False))


def plot(force=False):
    outputs = [HERE / f'Figure3.{fmt}' for fmt in ('pdf', 'png')]
    if not force and any(path.exists() for path in outputs):
        raise FileExistsError('Use --force to replace generated Figure3 outputs')
    source = json.loads((HERE / 'Figure3_source.json').read_text(encoding='utf-8'))
    events, scores = source['events'], source['scores']
    revision_path = HERE / 'Figure4_table2_alignment_20261117.json'
    revision = json.loads(revision_path.read_text(encoding='utf-8'))
    for row in scores:
        if row['variable'] == 'temperature' and row['candidate'] == 'raw':
            key = row['method'] + '/temperature'
            row['archived_rmse'] = row['rmse']
            row['rmse'] = revision['revised_rmse'][key]
    e = lambda method, candidate='common_center': unique(events, method=method, candidate=candidate)
    s = lambda method, variable, candidate='raw': unique(scores, method=method, variable=variable, candidate=candidate)
    values = []
    def record(panel, method, candidate, variable, metric, value, unit):
        assert np.isfinite(value)
        values.append(dict(panel=panel, method=method, candidate=candidate, variable=variable,
                           metric=metric, displayed_value=float(value), displayed_unit=unit))

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9.4, 'font.weight': 'medium',
            'axes.labelsize': 9.5, 'axes.labelweight': 'semibold',
            'axes.titlesize': 11, 'axes.titleweight': 'semibold', 'axes.titlepad': 8,
            'xtick.labelsize': 8.8, 'ytick.labelsize': 9.2,
            'xtick.major.width': 1, 'ytick.major.width': 1,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.linewidth': 1, 'pdf.fonttype': 42, 'ps.fonttype': 42,
            'savefig.facecolor': 'white'}):
        fig = plt.figure(figsize=(7.2, 5.4), layout='constrained')
        grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1), hspace=.14, wspace=.15)
        a = fig.add_subplot(grid[0, 0])
        b = fig.add_subplot(grid[0, 1])
        c = fig.add_subplot(grid[1, :])

        # a: log frequency is a position encoding, not a log-scaled bar length.
        for method, y in zip(METHODS, (2, 1, 0)):
            centre = 100 * float(e(method)['forecast_frequency'])
            assert centre > 0
            record('a', method, 'common_center', 'n2', 'severe_event_frequency', centre, '%')
            yc = y if method == 'deterministic' else y - .09
            if method != 'deterministic':
                member = 100 * float(e(method, 'raw')['forecast_frequency'])
                assert member > 0
                record('a', method, 'raw', 'n2', 'severe_event_frequency', member, '%')
                a.plot([centre, member], [yc, y + .09], color=COLORS[method], lw=1.2, alpha=.75, zorder=2)
                a.scatter(member, y + .09, s=70, marker='^', facecolors='white',
                          edgecolors=COLORS[method], linewidths=1.7, zorder=4)
                a.annotate(f'{member:.2f}%', (member, y + .09), xytext=(0, 10),
                           textcoords='offset points', ha='center', fontsize=8.7, color='#333333')
            a.scatter(centre, yc, s=64, marker='o', c=COLORS[method], edgecolors='white', linewidths=.55, zorder=4)
            text = f'{centre:.4f}%' if centre < .1 else f'{centre:.2f}%'
            centre_offset = (0, -11) if method == 'selected' else (-9, 0)
            a.annotate(text, (centre, yc), xytext=centre_offset, textcoords='offset points',
                       ha='center' if method == 'selected' else 'right',
                       va='top' if method == 'selected' else 'center', fontsize=8.7, color='#333333')
        reference = float(e('selected')['observed_frequency']) * 100
        record('a', 'glorys', 'reference', 'n2', 'severe_event_frequency', reference, '%')
        a.axvline(reference, color='#666666', ls=(0, (2, 2)), lw=1.25, zorder=1)
        a.text(.02, .97, f'GLORYS\n{reference:.5f}%', transform=a.transAxes, va='top',
               fontsize=8.5, fontweight='semibold', color='#444444',
               bbox=dict(facecolor='white', edgecolor='none', pad=.4))
        a.legend(handles=[Line2D([], [], marker='o', ls='none', color='#333333', markersize=5.8,
                                 label='Cycling centre'),
                          Line2D([], [], marker='^', ls='none', color='#333333', markerfacecolor='white',
                                 markersize=5.8, markeredgewidth=1.25, label='Ensemble members')],
                 loc='upper right', frameon=False, fontsize=8.2, borderaxespad=.12, handletextpad=.35,
                 labelspacing=.5)
        a.set(xscale='log', xlim=(1e-3, 50), ylim=(-.7, 3.1), yticks=(2, 1, 0),
              yticklabels=[LABELS[m] for m in METHODS], xlabel='Severe-instability frequency (%)',
              title='Static-instability events')
        a.set_xticks([.001, .01, .1, 1, 10], ['0.001', '0.01', '0.1', '1', '10'])
        a.grid(axis='x', which='major', color='#e3e3e3', lw=.7)
        a.tick_params(axis='y', length=0)

        # b: measured points only. No fitted frontier or inferred causal path.
        offsets = {'deterministic': (-8, 10, 'right'), 'guided_raw': (8, 7, 'left'),
                   'selected': (8, 2, 'left')}
        for method in METHODS:
            x = float(s(method, 'temperature')['rmse'])
            y = float(s(method, 'n2')['crps']) * 1e5
            record('b', method, 'raw', 'temperature', 'rmse', x, 'degC')
            record('b', method, 'raw', 'n2', 'crps', y, '1e-5 s^-2')
            b.scatter(x, y, s=95, marker=MARKERS[method], color=COLORS[method],
                      edgecolors='white', linewidths=.7, zorder=4)
            dx, dy, ha = offsets[method]
            b.annotate(LABELS[method], (x, y), xytext=(dx, dy), textcoords='offset points',
                       ha=ha, fontsize=9.3, fontweight='semibold', color='#222222')
        b.set(xlim=(.812, .855), ylim=(3.85, 5.65), xticks=(.815, .825, .835, .845, .855),
              yticks=(4.0, 4.5, 5.0, 5.5), xlabel=r'Temperature RMSE ($^{\circ}$C)',
              ylabel=r'$N^2$ CRPS ($10^{-5}$ s$^{-2}$)', title='Accuracy and stratification')
        b.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        b.grid(color='#e3e3e3', lw=.7)
        b.annotate('', xy=(.814, 3.96), xytext=(.819, 4.18),
                   arrowprops=dict(arrowstyle='->', lw=1.2, color='#555555'))
        b.text(.035, .02, 'Lower errors', transform=b.transAxes, fontsize=8.3, color='#555555', va='bottom')

        # c: raw/calibrated pairs on bounded 0-100% support; proximity to 90% matters.
        for variable, y in zip(VARIABLES, (3, 2, 1, 0)):
            raw = 100 * float(s('selected', variable, 'raw')['coverage90'])
            calibrated = 100 * float(s('selected', variable, 'marginal')['coverage90'])
            assert 0 <= raw <= 100 and 0 <= calibrated <= 100
            record('c', 'selected', 'raw', variable, 'coverage90', raw, '%')
            record('c', 'selected', 'marginal', variable, 'coverage90', calibrated, '%')
            c.plot([raw, calibrated], [y, y], color=COLORS['selected'], alpha=.8, lw=1.5, zorder=2)
            c.scatter(raw, y, s=72, marker='o', facecolors='white', edgecolors=COLORS['selected'],
                      linewidths=1.8, zorder=4)
            c.scatter(calibrated, y, s=64, marker='s', color=COLORS['selected'],
                      edgecolors='white', linewidths=.55, zorder=4)
            for value in (raw, calibrated):
                c.annotate(f'{value:.2f}', (value, y), xytext=(0, 9), textcoords='offset points',
                           ha='center', fontsize=8.7, color='#333333',
                           bbox=dict(facecolor='white', edgecolor='none', pad=.2))
        c.axvline(90, color='#666666', ls=(0, (4, 3)), lw=1.3, zorder=1)
        c.text(90, 3.72, 'Nominal 90%', ha='center', va='bottom',
               fontsize=8.6, fontweight='semibold', color='#444444',
               bbox=dict(facecolor='white', edgecolor='none', pad=.4))
        c.legend(handles=[Line2D([], [], marker='o', ls='none', color=COLORS['selected'],
                                  markerfacecolor='white', markersize=6, markeredgewidth=1.5, label='Raw SA-Diff'),
                          Line2D([], [], marker='s', ls='none', color=COLORS['selected'],
                                  markersize=5.8, label='Calibrated SA-Diff')],
                 loc='upper left', frameon=False, fontsize=8.6, borderaxespad=.3, handletextpad=.5)
        c.set(xlim=(0, 100), ylim=(-.55, 4.05), xticks=(0, 20, 40, 60, 80, 100),
              yticks=(3, 2, 1, 0), yticklabels=['Temperature', 'Salinity', 'Density', r'$N^2$'],
              xlabel='Central 90% prediction-interval coverage (%)', title='SA-Diff ensemble calibration')
        c.grid(axis='x', color='#e3e3e3', lw=.7)
        c.tick_params(axis='y', length=0)
        for label, ax, x in [('a', a, -.32), ('b', b, -.17), ('c', c, -.12)]:
            ax.text(x, 1.06, label, transform=ax.transAxes, fontsize=12.5, fontweight='bold')

        fig.savefig(outputs[0], dpi=300, metadata={
            'Title': 'Figure 3: static stability, forecast accuracy and ensemble calibration',
            'CreationDate': None, 'ModDate': None})
        fig.savefig(outputs[1], dpi=300)
        plt.close(fig)

    with (HERE / 'Figure3_plot_values.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    dump(HERE / 'Figure3_alignment_manifest_20261117.json', dict(
        figure='Figure3', input=dict(path='Figure3_source.json', sha256=sha(HERE / 'Figure3_source.json')),
        revised_temperature_rmse_input=dict(path=revision_path.name, sha256=sha(revision_path),
                                           panel='b', source='Author-selected revised Figure 4 summary'),
        code=dict(path=Path(__file__).name, sha256=sha(__file__)),
        plotted_values=dict(path='Figure3_plot_values.csv', sha256=sha(HERE / 'Figure3_plot_values.csv')),
        outputs=[dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size) for path in outputs],
        figsize_inches=[7.2, 5.4], png_pixels=[2160, 1620], dpi=300,
        style_revision='2026-11-17: panel b temperature coordinates aligned to revised Figure 4; panel a/c data, typography and axis limits retained. SA-Diff label moved right of its point to avoid the lower-error direction annotation.',
        typography_pt=dict(axis_label=9.5, title=11, x_tick=8.8, y_tick=9.2,
                           value_label=8.7, method_label=9.3, panel_label=12.5),
        marker_area_pt2=dict(event_centre=64, event_member=70, two_metric=95,
                             coverage_raw=72, coverage_calibrated=64),
        pdf='Vector lines, text and symbols with embedded TrueType fonts; DPI applies only to raster content.',
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
        palette=COLORS, smoothing=False, new_exclusions=False, new_inferential_statistics=False,
        transformations=['panel b temperature RMSE pooled from revised depth summary', 'event fraction multiplied by 100', 'N2 CRPS multiplied by 1e5', 'coverage fraction multiplied by 100'],
        axis_limits={'a_frequency_percent': [.001, 50], 'b_temperature_rmse': [.812, .855],
                     'b_n2_crps_scaled': [3.85, 5.65], 'c_coverage_percent': [0, 100]},
        uncertainty='Archived descriptive pooled summaries; connectors pair centre/member or raw/calibrated scores, not confidence intervals.',
        event_encoding='Base-10 logarithmic frequency axis; all plotted frequencies are strictly positive. Non-deterministic centre/member y offsets are categorical separation only.',
        design_references=['GenCast Nature 2025 Fig. 3b-f: calibration reference lines',
                           'NeuralGCM Nature 2024 Extended Data Fig. 3: separate physical diagnostics',
                           'Camps-Valls et al. Nature Communications 2026 Fig. 2: two-metric layout only, no Pareto fit'],
        publisher_rules='Provisional EMS CAS manuscript figure; live EMS guide returned HTTP 403 on 2026-11-12. No journal-compliance certification.',
        alt_text='Panel a compares severe negative-N2 frequencies of deterministic, Argo-Diff and SA-Diff cycling centres and diffusion members against GLORYS. SA-Diff strongly reduces centre instability, with a smaller reduction in members. Panel b shows the temperature-RMSE versus N2-CRPS trade-off without an inferred frontier. Panel c compares raw and marginally calibrated SA-Diff coverage against 90%; calibrated N2 coverage exceeds the target.',
        human_verified=False))
    print(json.dumps(dict(pdf=str(outputs[0]), png=str(outputs[1]), plotted_values=len(values),
                          source_event_rows=len(events), source_score_rows=len(scores)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-source', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.prepare_source:
        prepare_source()
    plot(args.force)
