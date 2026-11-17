"""Reproduce the current draft Figure 4 from the frozen author-revised RMSE table.

The arithmetic is checked, but the revised RMSE values have not been reconciled
with the archived daily scores. Keep this status in the caption and manifest.
"""
from pathlib import Path
import json

import plot_figure4_recomputed_20261115 as frozen
from plot_figure4_typography import plot


def main():
    root = Path(__file__).resolve().parent
    frozen.HERE = root
    frozen.PROJECT = root
    source = root / 'Figure4_input_revision_20261115.csv'
    assert frozen.sha(source) == frozen.sha(root / 'Figure4_source.csv'), (
        'Source revision changed. Review and version it before regenerating Figure 4.'
    )
    rows, changes = frozen.calculate()
    axes = plot(rows)
    manifest = json.loads((root / 'Figure4_manifest_20261115.json').read_text(encoding='utf-8'))
    assert manifest['source']['sha256'] == frozen.sha(source)
    manifest['axes'] = axes
    manifest['changes_beyond_rmse_rounding'] = changes
    manifest['typography_revision'] = dict(
        date='2026-11-15', footer_annotations_removed=True,
        source_status_retained_in_caption=True, all_data_and_limits_unchanged=True,
        curve_linewidth_pt=2.35, zero_linewidth_pt=1.45, axis_linewidth_pt=1.35,
        tick_linewidth_pt=1.25, font_weight='bold',
        label_and_tick_fontsize_pt=9.5, title_fontsize_pt=11.5,
        original_figure_backup='../Figure4_typography_20261115_before_revision/Figure4.pdf')
    manifest['outputs'] = [
        dict(path=name, sha256=frozen.sha(root / name), bytes=(root / name).stat().st_size)
        for name in ('Figure4.pdf', 'Figure4.png', frozen.STEM + '.csv')
    ]
    manifest['code'] = [
        dict(path=name, sha256=frozen.sha(root / name))
        for name in ('plot_figure4.py', Path(__file__).name, 'plot_figure4_typography.py',
                     'plot_figure4_recomputed_20261115.py')
    ]
    frozen.dump(root / 'Figure4_manifest_20261115.json', manifest)
    registry_path = root / 'evidence_registry_20261108.json'
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
        evidence = next(r for r in registry['sources'] if r['evidence_id'] == 'E446')
        evidence['sha256'] = frozen.sha(root / 'Figure4_manifest_20261115.json')
        evidence['locator'] = (
            'Recalculated reductions from RMSE columns; complete depth counts and output/code hashes. '
            'Typography revision removes in-figure provenance text and boldens strokes/text only. '
            'Pending daily-score reconciliation remains explicit in the unchanged manuscript caption.')
        assert evidence['daily_score_reconciliation'] is False and evidence['human_verified'] is False
        frozen.dump(registry_path, registry)
    print('Figure 4 regenerated from revised RMSE columns; daily-score reconciliation pending.')


if __name__ == '__main__':
    main()
