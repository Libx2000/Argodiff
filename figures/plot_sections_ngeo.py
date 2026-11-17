"""Redraw the two archived T/S sections with a restrained section-plot style.

The filled meshes retain all native-grid values and masks. White isolines use
Matplotlib's linear cell interpolation on adjacent unmasked nodes only.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, default=HERE / 'source_data' / 'sections_20261112')
parser.add_argument('--output-dir', type=Path, default=HERE)
parser.add_argument('--deps', type=Path)
parser.add_argument('--force', action='store_true')
args = parser.parse_args()
for candidate in (args.deps, HERE.parent / 'ems_results_plot_deps'):
    if candidate is not None and candidate.is_dir():
        sys.path.insert(0, str(candidate))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe
from plot_ems_sections_2d import check_pair, edges, load_verified


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate_label(value, position, longitude=False):
    if abs(value) < 1e-8:
        return '0°'
    suffix = ('E' if value > 0 else 'W') if longitude else ('N' if value > 0 else 'S')
    return f'{abs(value):g}°{suffix}'


def main():
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ['Figure6.pdf', 'Figure6.png', 'Figure7.pdf', 'Figure7.png',
             'section_style_manifest_20261116.json']
    if not args.force and any((out / name).exists() for name in names):
        raise FileExistsError('Refusing to overwrite previous outputs without --force')
    loaded = [load_verified(args.source_root, method, 90) for method in ('deterministic', 'selected')]
    arrays = [item[0] for item in loaded]
    check_pair(arrays, [item[2] for item in loaded])
    original_hashes = [sha(item[1]) for item in loaded]
    with (args.source_root / 'section_comparison_audited.csv').open(encoding='utf-8', newline='') as f:
        audited = {(row['section'], row['variable']): row for row in csv.DictReader(f)}

    # Reuse exactly the original shared, full-range normalization across both sections.
    specs = {}
    for variable_index, variable in enumerate(('temperature', 'salinity')):
        fields, changes = [], []
        for section in ('zonal', 'meridional'):
            ref = arrays[0][section + '_truth'][variable_index]
            valid = arrays[0][section + '_valid'][variable_index]
            predictions = [a[section + '_prediction'][variable_index] for a in arrays]
            fields.extend([ref[valid]] + [a[valid] for a in predictions])
            change = abs(predictions[1] - ref) - abs(predictions[0] - ref)
            changes.append(change[valid])
        all_values = np.concatenate(fields)
        specs[variable] = dict(vmin=float(all_values.min()), vmax=float(all_values.max()),
                               error_change_limit=float(np.max(abs(np.concatenate(changes)))))

    contour_levels = {'temperature': [5., 10., 15., 20., 25., 30.],
                      'salinity': [33., 33.5, 34., 34.5, 35., 35.5]}
    rc = {
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 9.2, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': .7, 'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'axes.facecolor': '#adadad',
    }
    outputs, panels, metrics = [], [], []
    with plt.rc_context(rc):
        for section, stem in (('zonal', 'Figure6'), ('meridional', 'Figure7')):
            x = arrays[0]['zonal_longitude' if section == 'zonal' else 'meridional_latitude']
            depth = arrays[0]['depth_m']
            xe, de = edges(x), edges(depth)
            de[0], de[-1] = max(0., de[0]), depth[-1]
            fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.1))
            fig.subplots_adjust(left=.08, right=.88, top=.90, bottom=.08,
                                wspace=.38, hspace=.36)
            contour_labels = []
            fig.suptitle('17.25°N | Longitude-depth section' if section == 'zonal'
                         else '135°E | Latitude-depth section',
                         y=.99, fontsize=11, fontweight='semibold')
            for v, variable in enumerate(('temperature', 'salinity')):
                ref = arrays[0][section + '_truth'][v]
                valid = arrays[0][section + '_valid'][v]
                deterministic, selected = [a[section + '_prediction'][v] for a in arrays]
                change = abs(selected - ref) - abs(deterministic - ref)
                values = [ref, deterministic, selected, change]
                names = ['GLORYS', 'Deterministic', 'SA-Diff', 'Absolute-error change']
                spec = specs[variable]
                field_norm = Normalize(spec['vmin'], spec['vmax'])
                limit = spec['error_change_limit']
                change_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
                images = []
                position = axes[0, v].get_position()
                fig.text((position.x0 + position.x1) / 2, .952,
                         'Temperature' if v == 0 else 'Practical Salinity',
                         ha='center', fontsize=10, fontweight='semibold')
                for row in range(4):
                    ax = axes[row, v]
                    masked = np.ma.array(values[row], mask=~valid)
                    cmap = plt.get_cmap('RdBu_r' if row == 3 else 'viridis').with_extremes(bad='#adadad')
                    mesh = ax.pcolormesh(xe, de, masked, shading='flat', cmap=cmap,
                                         norm=change_norm if row == 3 else field_norm,
                                         antialiased=False, rasterized=True)
                    np.testing.assert_array_equal(mesh.get_array().compressed(), masked.compressed())
                    images.append(mesh)
                    label = chr(97 + 2 * row + v)
                    ax.text(-.06, 1.04, label, transform=ax.transAxes,
                            fontsize=10.8, fontweight='bold', va='bottom')
                    ax.set_title(names[row], loc='left', pad=4, x=.03)
                    ax.set(xlim=(x[0], x[-1]), ylim=(depth[-1], 0),
                           yticks=[0, 100, 200, 300, 400, 500])
                    if v == 0:
                        ax.set_ylabel('Depth (m)', labelpad=3)
                    else:
                        ax.tick_params(labelleft=False)
                    if section == 'zonal':
                        ax.set_xticks([115, 130, 145, 160, 175])
                        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: coordinate_label(val, pos, True)))
                    else:
                        ax.set_xticks([-5, 10, 25, 40])
                        ax.xaxis.set_major_formatter(FuncFormatter(coordinate_label))
                    if row < 3:
                        ax.tick_params(labelbottom=False)
                        # White isolines follow the reference's visual language.
                        # No high-resolution field or smoothing is constructed.
                        levels = [level for level in contour_levels[variable]
                                  if masked.min() < level < masked.max()]
                        contour = ax.contour(x, depth, masked, levels=levels,
                                             colors='white', linewidths=.55,
                                             corner_mask=False, alpha=.95)
                        texts = ax.clabel(contour, inline=True, inline_spacing=2,
                                          fontsize=6.7, fmt='%g', colors='white')
                        for text in texts:
                            text.set_path_effects([pe.withStroke(linewidth=.8,
                                                                foreground='#33424b', alpha=.6)])
                            contour_labels.append((ax, text))
                    else:
                        ax.set_xlabel('Longitude' if section == 'zonal' else 'Latitude', labelpad=3)
                    panels.append(dict(figure=stem, panel=label, section=section,
                                       variable=variable, field=names[row], finite_count=int(masked.count())))

                p_top, p_bottom = axes[0, v].get_position(), axes[2, v].get_position()
                cax = fig.add_axes([p_top.x1 + .012, p_bottom.y0, .011,
                                    p_top.y1 - p_bottom.y0])
                cb = fig.colorbar(images[0], cax=cax,
                                  ticks=[0, 5, 10, 15, 20, 25, 30] if v == 0
                                  else [32.5, 33, 33.5, 34, 34.5, 35, 35.5])
                cb.set_label('Temperature (°C)' if v == 0 else 'Practical Salinity',
                             labelpad=5, fontsize=8.3)
                cb.ax.tick_params(labelsize=7.3, length=2.5, pad=2)
                cb.outline.set_linewidth(.6)
                p = axes[3, v].get_position()
                cax = fig.add_axes([p.x1 + .012, p.y0, .011, p.height])
                cb = fig.colorbar(images[3], cax=cax,
                                  ticks=[-3, 0, 3] if v == 0 else [-.5, 0, .5])
                cb.set_label(r'$\Delta|\mathrm{error}|$ (°C)' if v == 0 else r'$\Delta|\mathrm{error}|$',
                             labelpad=5, fontsize=8.1)
                cb.ax.tick_params(labelsize=7.3, length=2.5, pad=2)
                cb.outline.set_linewidth(.6)

                # Reproduce the established section scores, including mixed outcomes.
                for method, prediction in [('deterministic', deterministic), ('selected', selected)]:
                    diff = (prediction - ref)[valid].astype(float)
                    record = dict(section=section, variable=variable, method=method,
                                  count=len(diff), rmse=float(np.sqrt(np.mean(diff ** 2))),
                                  mae=float(np.mean(abs(diff))), bias=float(np.mean(diff)))
                    previous = audited[section, variable]
                    assert int(previous['count']) == len(diff)
                    for metric in ('rmse', 'mae', 'bias'):
                        np.testing.assert_allclose(record[metric], float(previous[method + '_' + metric]),
                                                   atol=1e-12, rtol=1e-12)
                    metrics.append(record)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            # Hide only partially clipped annotations, never field data or isolines.
            for ax, text in contour_labels:
                box = text.get_window_extent(renderer)
                bounds = ax.get_window_extent(renderer)
                if (box.x0 < bounds.x0 + 2 or box.x1 > bounds.x1 - 2 or
                        box.y0 < bounds.y0 + 2 or box.y1 > bounds.y1 - 2):
                    text.set_visible(False)
            # Detect clipped figure-level text and axes/colorbar labels at export size.
            for ax in fig.axes:
                for text in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
                    if text.get_visible() and text.get_text():
                        box = text.get_window_extent(renderer)
                        if box.x0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y0 < -1 or box.y1 > fig.bbox.y1 + 1:
                            raise RuntimeError(f'Clipped label: {text.get_text()}')
            for extension in ('pdf', 'png'):
                destination = out / (stem + '.' + extension)
                metadata = {'Title': stem + ': ' + section + ' temperature-salinity section',
                            'Subject': 'Native-grid fields and signed absolute-error changes; isolines are linearly interpolated within valid cells.',
                            'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None
                fig.savefig(destination, dpi=300, metadata=metadata)
                outputs.append(dict(path=destination.name, sha256=sha(destination), bytes=destination.stat().st_size))
            plt.close(fig)

    assert [sha(item[1]) for item in loaded] == original_hashes
    manifest = dict(
        revision_date='2026-11-16',
        style_reference='Knorr et al. (2021), Nature Geoscience, Figure 2; https://doi.org/10.1038/s41561-021-00857-3',
        original_figure_reused=False,
        visual_changes=['Viridis native-grid fields with thin white isolines',
                        'Compact aligned panels with separate variable headings',
                        'Shared slim colourbars; grey unavailable-reference mask',
                        'Two-sided blue-white-red error-change scale retained'],
        inputs=[dict(path=str(item[1].relative_to(HERE)), sha256=sha(item[1])) for item in loaded],
        outputs=outputs, normalization=specs,
        dimensions_inches=[7.2, 7.1], export_dpi=300,
        pdf_text_and_contours='vector with embedded TrueType fonts',
        pdf_meshes='rasterized at 300 dpi',
        field_values_unchanged=True, all_stored_levels_retained=True,
        smoothing=False, imputation=False, quantile_clipping=False,
        contours=dict(levels=contour_levels,
                      interpolation='Linear contour interpolation within adjacent valid nodes; corner_mask=False prevents crossing missing nodes.',
                      filled_fields='Native cells, without interpolation or smoothing.'),
        error_change='abs(SA-Diff minus GLORYS) minus abs(deterministic minus GLORYS); negative indicates improvement.',
        shared_limits_across_both_sections=True, panels=panels,
        reproduced_metrics=metrics, all_archived_section_metrics_reproduced=True,
        code=dict(path=Path(__file__).name, sha256=sha(__file__)),
        human_verified=False)
    (out / 'section_style_manifest_20261116.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(outputs=outputs, metrics=metrics), indent=2))


if __name__ == '__main__':
    main()
