"""Recalculate percentages from an author-revised RMSE summary, without replacing its source."""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import argparse
import csv
import hashlib
import json
import math
import shutil
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
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pypdf import PdfReader

EMS = Path(r'C:\Users\Administrator\Desktop\EMS')
PROJECT = EMS / 'els-cas-templates'
STEM = 'Figure4_recomputed_20261115'
METHODS = ('deterministic', 'guided_raw', 'selected')
COLORS = dict(zip(METHODS, ('#444444', '#0072B2', '#D55E00')))
STYLES = dict(zip(METHODS, ('--', '-', '-.')))
LABELS = dict(zip(METHODS, ('Deterministic reference', 'Argo-Diff', 'SA-Diff')))
PANELS = (('temperature', 'Temperature'), ('salinity', 'Salinity'),
          ('rho', 'Density'), ('n2', r'$N^2$'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def calculate():
    path = HERE / 'Figure4_input_revision_20261115.csv'
    rows = readcsv(path)
    assert len(rows) == 177
    keys = [(r['variable'], r['method'], float(r['depth_m'])) for r in rows]
    assert len(set(keys)) == 177
    assert set(r['method'] for r in rows) == set(METHODS)
    reference = {(r['variable'], float(r['depth_m'])): float(r['rmse'])
                 for r in rows if r['method'] == 'deterministic'}
    assert len(reference) == 59
    output = []
    for row in rows:
        out = dict(row)
        value = float(row['rmse'])
        baseline = reference[row['variable'], float(row['depth_m'])]
        assert math.isfinite(value) and value >= 0 and math.isfinite(baseline) and baseline > 0
        assert math.isclose(float(row['deterministic_rmse']), baseline, rel_tol=1e-12, abs_tol=1e-15)
        gain = 100 * (1 - value / baseline)
        out['rmse_reduction_percent'] = repr(gain)
        out['supplied_reduction_percent'] = row['rmse_reduction_percent']
        out['recomputed_minus_supplied_pp'] = repr(gain - float(row['rmse_reduction_percent']))
        output.append(out)
    with (HERE / (STEM + '.csv')).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    previous = readcsv(HERE / 'Figure4_reference_20261114.csv')
    assert len(previous) == len(rows)
    beyond_rounding = []
    for a, b in zip(previous, rows):
        assert (a['method'], a['variable']) == (b['method'], b['variable'])
        assert math.isclose(float(a['depth_m']), float(b['depth_m']), rel_tol=1e-8, abs_tol=1e-7)
        old, new = Decimal(a['rmse']), Decimal(b['rmse'])
        half_unit = Decimal(5).scaleb(new.as_tuple().exponent - 1)
        if abs(old - new) > half_unit:
            beyond_rounding.append(dict(method=b['method'], variable=b['variable'], depth_m=b['depth_m'],
                earlier_rmse=a['rmse'], revised_rmse=b['rmse']))
    assert len(beyond_rounding) == 3
    counts = Counter((r['variable'], r['method']) for r in output)
    for variable, title in PANELS:
        for method in METHODS:
            assert counts[variable, method] == (14 if variable == 'n2' else 15)
    return output, beyond_rounding


def plot(rows):
    axes_record = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
            'axes.titlesize': 11, 'axes.titleweight': 'semibold', 'axes.labelsize': 9,
            'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': .9,
            'axes.spines.top': False, 'axes.spines.right': False,
            'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), sharey=True)
        fig.subplots_adjust(left=.09, right=.98, bottom=.105, top=.91, hspace=.29, wspace=.25)
        fig.legend(handles=[Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=1.7,
            label=LABELS[m]) for m in METHODS], loc='upper center', bbox_to_anchor=(.52, .999),
            ncol=3, frameon=False, handlelength=2.9, columnspacing=1.6, fontsize=9)
        for index, (ax, (variable, title)) in enumerate(zip(axes.flat, PANELS)):
            values = [float(r['rmse_reduction_percent']) for r in rows if r['variable'] == variable]
            lo, hi = min(0, min(values)), max(0, max(values))
            padding = .07 * (hi - lo)
            bounds = (lo - padding, hi + padding)
            for method in METHODS[1:]:
                rr = sorted([r for r in rows if r['variable'] == variable and r['method'] == method],
                            key=lambda r: float(r['depth_m']))
                x = np.array([float(r['rmse_reduction_percent']) for r in rr])
                y = np.array([float(r['depth_m']) for r in rr])
                assert np.all((x >= bounds[0]) & (x <= bounds[1]))
                ax.plot(x, y, color=COLORS[method], ls=STYLES[method], lw=1.8)
            ax.axvline(0, color=COLORS['deterministic'], ls=STYLES['deterministic'], lw=1.1, zorder=0)
            ax.set(title=title, xlim=bounds, ylim=(520, -12), yticks=(0, 100, 200, 300, 400, 500),
                   xlabel='RMSE reduction (%)')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if index % 2 == 0:
                ax.set_ylabel('Depth (m)')
            ax.grid(color='#e4e4e4', lw=.55)
            ax.text(-.14, 1.03, 'abcd'[index], transform=ax.transAxes, fontsize=12, fontweight='bold')
            assert all(line.get_marker() in ('None', '', None) for line in ax.lines)
            axes_record[variable] = dict(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                linear=True, all_native_values_visible=True)
        fig.text(.535, .035, 'Recalculated from revised RMSE summary | 15 Nov 2026',
                 fontsize=7.6, color='#444444', ha='center')
        fig.text(.535, .015, 'Daily-score reconciliation pending', fontsize=7.2, color='#555555', ha='center')
        fig.savefig(HERE / (STEM + '.pdf'), dpi=300, metadata={
            'Title': 'RMSE reduction recalculated from revised summary; daily-score reconciliation pending',
            'CreationDate': None, 'ModDate': None})
        fig.savefig(HERE / (STEM + '.png'), dpi=300)
        plt.close(fig)
    return axes_record


def main(publish=False):
    protect = ['Figure4_source.csv', 'Figure4.pdf', 'Figure4.png', 'ems_argo_diffusion.tex',
               'ems_argo_diffusion.pdf', 'evidence_registry_20261108.json']
    before = {name: sha(PROJECT / name) for name in protect}
    input_path = HERE / 'Figure4_input_revision_20261115.csv'
    assert before['Figure4_source.csv'] == sha(input_path)
    rows, changes = calculate()
    axes = plot(rows)
    assert sha(input_path) == before['Figure4_source.csv']
    output = readcsv(HERE / (STEM + '.csv'))
    for original, derived in zip(readcsv(input_path), output):
        for k in original:
            if k != 'rmse_reduction_percent':
                assert original[k] == derived[k]
        assert float(derived['rmse_reduction_percent']) == 100 * (1 - float(derived['rmse']) / float(derived['deterministic_rmse']))
    with Image.open(HERE / (STEM + '.png')) as im:
        assert im.size == (2160, 1980) and all(abs(v - 300) < .01 for v in im.info['dpi'])
    assert len(PdfReader(HERE / (STEM + '.pdf')).pages) == 1
    summaries = {}
    for variable, title in PANELS:
        summaries[variable] = {}
        for method in METHODS[1:]:
            rr = [r for r in rows if r['variable'] == variable and r['method'] == method]
            summaries[variable][method] = dict(positive_depths=sum(float(r['rmse_reduction_percent']) > 0 for r in rr),
                n_depths=len(rr), min_reduction=min(float(r['rmse_reduction_percent']) for r in rr),
                max_reduction=max(float(r['rmse_reduction_percent']) for r in rr))
    names = [STEM + ext for ext in ('.pdf', '.png', '.csv')] + [input_path.name,
        'Figure4_reference_20261114.csv', Path(__file__).name, 'Figure4_recomputed_README_20261115.md']
    manifest_name = STEM + '_manifest.json'
    dump(HERE / manifest_name, dict(
        source=dict(path=input_path.name, sha256=sha(input_path), role='Author-revised RMSE summary; input preserved verbatim.'),
        baseline='Deterministic method RMSE at matching variable/depth; existing baseline column checked for equality.',
        metric='100 * (1 - method_RMSE / deterministic_RMSE)', ignored_input_column='rmse_reduction_percent',
        raw_rmse_cells_unchanged=True, recalculated_rows=177, excluded_values=0, no_manual_zero_replacement=True,
        no_offsets=True, no_smoothing=True, no_extrapolation=True, no_point_markers=True,
        numerical_formula_verified=True, daily_score_reconciliation=False, human_verified=False,
        scientific_verification='This recalculation validates arithmetic on the revised summary, not the experimental correctness of the revised RMSE values.',
        earlier_reference=dict(path='Figure4_reference_20261114.csv', sha256=sha(HERE / 'Figure4_reference_20261114.csv')),
        changes_beyond_rmse_rounding=changes,
        old_daily_source=dict(path=str(PROJECT / 'source_data/validation_depth_daily.csv'),
            sha256=sha(PROJECT / 'source_data/validation_depth_daily.csv'),
            updated_since_previous_verified_figure=False),
        max_absolute_recomputed_minus_supplied_pp=max(abs(float(r['recomputed_minus_supplied_pp'])) for r in rows),
        version_label_on_figure=True, manuscript_not_modified=True, protected_files_sha256=before,
        dimensions_inches=[7.2, 6.6], png_pixels=[2160, 1980], export_dpi=300,
        vector_pdf=True, embedded_fonttype=42, axis_limits=axes, summary=summaries,
        files=[dict(path=name, sha256=sha(HERE / name), bytes=(HERE / name).stat().st_size) for name in names],
        matplotlib=matplotlib.__version__, numpy=np.__version__, submission_ready=False))
    assert all(sha(PROJECT / name) == value for name, value in before.items())
    if publish:
        target = EMS / STEM
        assert not target.exists()
        target.mkdir()
        for name in names + [manifest_name]:
            shutil.copy2(HERE / name, target / name)
            assert sha(target / name) == sha(HERE / name)
        for ext in ('.pdf', '.png', '.csv'):
            path = EMS / (STEM + ext)
            assert not path.exists()
            shutil.copy2(HERE / path.name, path)
        assert all(sha(PROJECT / name) == value for name, value in before.items())
    print(json.dumps(dict(figure=str((EMS if publish else HERE) / (STEM + '.pdf')),
        recalculated_rows=len(rows), source_preserved=True, manuscript_changed=False,
        summary=summaries), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    main(parser.parse_args().publish)
