"""Fixed-site density/N2 profiles, using Figure 4 typography and explicit zooms."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / 'gsw_deps', ROOT.parent / 'ems_results_plot_deps'):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter

METHODS = ('deterministic', 'guided_raw', 'selected')
COLORS = dict(zip(METHODS, ('#444444', '#0072B2', '#D55E00')))
STYLES = dict(zip(METHODS, ('--', '-', '-.')))
LABELS = dict(zip(METHODS, ('Deterministic', 'Argo-Diff', 'SA-Diff')))
REFERENCE_STYLE = (0, (1, 1.5))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    with (ROOT / 'Figure5_source.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 261
    curves = {}
    for variable in ('rho', 'n2'):
        for site in (1, 2, 3):
            for method in METHODS:
                selected = sorted([r for r in rows if r['variable'] == variable and
                                   int(r['site']) == site and r['method'] == method],
                                  key=lambda r: float(r['depth_m']))
                assert len(selected) == (15 if variable == 'rho' else 14)
                curves[variable, site, method] = {field: np.array([float(r[field]) for r in selected])
                    for field in ('depth_m', 'center', 'truth', 'member_q05', 'member_q95')}
    return rows, curves


def limits(values, padding=.06):
    values = np.asarray(values)
    low, high = np.min(values), np.max(values)
    spread = high - low
    assert np.isfinite(values).all() and spread > 0
    return float(low - padding * spread), float(high + padding * spread)


def draw(ax, curves, variable, site, scale=1., band=True, linewidth=2.35):
    if band:
        selected = curves[variable, site, 'selected']
        ax.fill_betweenx(selected['depth_m'], selected['member_q05'] * scale,
                         selected['member_q95'] * scale, color=COLORS['selected'], alpha=.13,
                         edgecolor='none', zorder=1)
    for method in METHODS:
        data = curves[variable, site, method]
        line, = ax.plot(data['center'] * scale, data['depth_m'], color=COLORS[method],
                       ls=STYLES[method], lw=linewidth, zorder=3)
        np.testing.assert_array_equal(line.get_xdata(), data['center'] * scale)
        np.testing.assert_array_equal(line.get_ydata(), data['depth_m'])
    reference = curves[variable, site, 'deterministic']
    ax.plot(reference['truth'] * scale, reference['depth_m'], color='#111111',
            ls=REFERENCE_STYLE, lw=linewidth * .9, zorder=4)


def render(sites, stem):
    assert sites in ((1, 3), (1, 2, 3))
    full = len(sites) == 3
    dimensions = (7.2, 6.8) if full else (7.2, 6.6)
    rows, curves = load()
    source_hash = sha(ROOT / 'Figure5_source.csv')
    recorded_axes, recorded_zooms = {}, []
    style = {
        'font.family': 'DejaVu Sans', 'font.size': 9.5, 'font.weight': 'bold',
        'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.labelsize': 9.5,
        'axes.labelweight': 'bold', 'axes.linewidth': 1.35,
        'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0,
        'xtick.major.width': 1.25, 'ytick.major.width': 1.25,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, len(sites), figsize=dimensions, sharey='row')
        fig.subplots_adjust(left=.085, right=.985, bottom=.095, top=.85,
                            hspace=.39, wspace=.30)
        handles = [Line2D([], [], color='#111111', ls=REFERENCE_STYLE, lw=2.1, label='GLORYS')]
        handles += [Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=2.35, label=LABELS[m]) for m in METHODS]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .995),
                   ncol=4, frameon=False, handlelength=2.4, columnspacing=1.1,
                   prop={'family': 'DejaVu Sans', 'weight': 'bold', 'size': 9.0})
        fig.legend(handles=[Patch(facecolor=COLORS['selected'], alpha=.13,
                                  label='SA-Diff empirical 5-95% member range')],
                   loc='upper center', bbox_to_anchor=(.52, .945), frameon=False,
                   prop={'family': 'DejaVu Sans', 'weight': 'bold', 'size': 8.5})
        for row_index, variable in enumerate(('rho', 'n2')):
            scale = 1e4 if variable == 'n2' else 1.
            all_values = []
            for site in sites:
                for method in METHODS:
                    data = curves[variable, site, method]
                    all_values.extend(data['center'] * scale)
                    all_values.extend(data['truth'] * scale)
                    if method == 'selected':
                        all_values.extend(data['member_q05'] * scale)
                        all_values.extend(data['member_q95'] * scale)
            bounds = limits(all_values)
            if variable == 'n2':
                bounds = float(4 * np.floor(bounds[0] / 4)), float(4 * np.ceil(bounds[1] / 4))
            for column, site in enumerate(sites):
                ax = axes[row_index, column]
                draw(ax, curves, variable, site, scale)
                ax.set(xlim=bounds, ylim=(520, -12), yticks=(0, 100, 200, 300, 400, 500))
                if variable == 'n2':
                    ax.set_xticks(np.arange(bounds[0], bounds[1] + .01, 4))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.set_xlabel(r'Density (kg $\mathbf{m}^{\mathbf{-3}}$)' if variable == 'rho'
                              else r'$\mathbf{N}^{\mathbf{2}}$ ($\mathbf{10}^{\mathbf{-4}}$ $\mathbf{s}^{\mathbf{-2}}$)')
                if column == 0:
                    ax.set_ylabel('Depth (m)')
                if row_index == 0:
                    metadata = next(r for r in rows if int(r['site']) == site)
                    ax.set_title(f"{float(metadata['latitude']):.2f}°N, {float(metadata['longitude']):.1f}°E", pad=9)
                else:
                    ax.axvline(0, color='#888888', lw=.8, ls=':', zorder=0)
                ax.grid(color='#e4e4e4', lw=.55)
                ax.text(-.17, 1.035, 'abcdef'[row_index * len(sites) + column], transform=ax.transAxes,
                        fontsize=12.5, fontweight='bold')
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                # Uniform, variable-specific depth windows are shown for all sites.
                shallow, deep = (0., 35.) if variable == 'rho' else (150., 480.)
                detail_values = []
                for method in METHODS:
                    data = curves[variable, site, method]
                    mask = (data['depth_m'] >= shallow) & (data['depth_m'] <= deep)
                    detail_values.extend(data['center'][mask] * scale)
                    detail_values.extend(data['truth'][mask] * scale)
                inset_bounds = limits(detail_values, padding=.12)
                box = [.10, .08, .51, .43] if variable == 'rho' else [.46, .08, .50, .43]
                inset = ax.inset_axes(box, facecolor='white', zorder=7)
                draw(inset, curves, variable, site, scale, band=False, linewidth=1.35)
                inset.set(xlim=inset_bounds, ylim=(deep, shallow))
                inset.xaxis.set_major_locator(MaxNLocator(2))
                inset.yaxis.set_major_locator(MaxNLocator(3))
                inset.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                inset.tick_params(labelsize=7.2, width=.8, length=2, pad=1)
                inset.set_title(f'{shallow:.0f}-{deep:.0f} m', fontsize=7.8, pad=3, fontweight='bold')
                for spine in inset.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(.85)
                for label in inset.get_xticklabels() + inset.get_yticklabels():
                    label.set_fontweight('bold')
                inset.grid(color='#eeeeee', lw=.4)
                ax.add_patch(Rectangle((inset_bounds[0], shallow), inset_bounds[1] - inset_bounds[0],
                                      deep - shallow, fill=False, edgecolor='#777777', lw=.75,
                                      ls=(0, (2, 2)), zorder=5))
                recorded_zooms.append(dict(panel='abcdef'[row_index * len(sites) + column], variable=variable,
                    site=site, xlim=list(inset_bounds), ylim=[deep, shallow],
                    range_rule='All four centre/reference curves in the common variable-specific depth window, with 12% padding.',
                    members='Member range shown in full in parent panels, not repeated in insets.'))
                assert all(line.get_marker() in ('None', '', None) for line in ax.lines)
                recorded_axes[variable, site] = dict(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
        fig.savefig(ROOT / (stem + '.pdf'), dpi=300, metadata={
            'Title': 'Density and stratification: ' + ('all three fixed sites' if full else 'two illustrative cases'),
            'Subject': 'All-site comparison retained in the appendix; source values unchanged.',
            'CreationDate': None, 'ModDate': None})
        fig.savefig(ROOT / (stem + '.png'), dpi=300)
        plt.close(fig)
    assert sha(ROOT / 'Figure5_source.csv') == source_hash
    manifest = dict(
        figure=stem, version='2026-11-15-focus', case='Winter 2019, cycle day 90, seed 20261006',
        fixed_sites=3, displayed_sites=list(sites), displayed_all_sites=full,
        no_outcome_based_site_selection=full,
        selection='All three fixed sites.' if full else 'Post hoc illustrative selection of sites 1 and 3 with reduced local errors; complete all-site comparison retained in the appendix.',
        excluded_from_study=[], all_results_retained_in_manuscript=True,
        sources=[dict(path=name, sha256=sha(ROOT / name)) for name in
                 ('Figure5_source.csv', 'Figure5_audit_20261115.json', 'Figure5_site_metrics.csv')],
        source_unchanged=True, no_smoothing=True, no_extrapolation=True, no_point_markers=True,
        no_value_offsets=True, density_is_in_situ=True, n2_display_scale=1e4,
        uncertainty='SA-Diff raw ensemble pointwise 5th-95th member quantiles; 50 members, not a confidence interval.',
        axes=[dict(variable=var, site=site, **record) for (var, site), record in recorded_axes.items()],
        horizontal_limits_common_within_row=True, zooms=recorded_zooms,
        typography='Figure 4 palette, line-style mapping, bold sans-serif text, 2.35 pt curves and 1.35 pt axes.',
        curve_linewidth_pt=2.35, axes_linewidth_pt=1.35, dimensions_inches=list(dimensions),
        export_dpi=300, png_pixels=[2160, round(dimensions[1] * 300)], vector_pdf=True, human_verified=False,
        code=dict(path=Path(__file__).name, sha256=sha(Path(__file__))),
        outputs=[dict(path=name, sha256=sha(ROOT / name), bytes=(ROOT / name).stat().st_size)
                 for name in (stem + '.pdf', stem + '.png')], matplotlib=matplotlib.__version__)
    (ROOT / (stem + '_manifest_20261115.json')).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(outputs=manifest['outputs'], zooms=recorded_zooms), indent=2))


def main():
    render((1, 3), 'Figure5')
    render((1, 2, 3), 'FigureS2_all_sites')


if __name__ == '__main__':
    main()
