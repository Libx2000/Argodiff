"""Figure 4 typography revision only; data, axis limits and source status unchanged."""
from pathlib import Path

import plot_figure4_recomputed_20261115 as source

np = source.np
plt = source.plt
Line2D = source.Line2D
MaxNLocator = source.MaxNLocator


def plot(rows):
    root = Path(__file__).resolve().parent
    axes_record = {}
    style = {
        'font.family': 'DejaVu Sans', 'font.size': 9.5, 'font.weight': 'bold',
        'axes.titlesize': 11.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 9.5, 'axes.labelweight': 'bold',
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'xtick.major.width': 1.25, 'ytick.major.width': 1.25,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'axes.linewidth': 1.35, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), sharey=True)
        fig.subplots_adjust(left=.095, right=.98, bottom=.09, top=.90,
                            hspace=.35, wspace=.27)
        fig.legend(
            handles=[Line2D([], [], color=source.COLORS[m], ls=source.STYLES[m],
                            lw=2.25, label=source.LABELS[m]) for m in source.METHODS],
            loc='upper center', bbox_to_anchor=(.52, .997), ncol=3, frameon=False,
            handlelength=2.9, columnspacing=1.5,
            prop={'family': 'DejaVu Sans', 'size': 9.3, 'weight': 'bold'})
        for index, (ax, (variable, title)) in enumerate(zip(axes.flat, source.PANELS)):
            if variable == 'n2':
                title = r'$\mathbf{N}^{\mathbf{2}}$'
            values = [float(r['rmse_reduction_percent']) for r in rows if r['variable'] == variable]
            lo, hi = min(0, min(values)), max(0, max(values))
            padding = .07 * (hi - lo)
            bounds = lo - padding, hi + padding
            for method in source.METHODS[1:]:
                rr = sorted([r for r in rows if r['variable'] == variable and r['method'] == method],
                            key=lambda r: float(r['depth_m']))
                x = np.array([float(r['rmse_reduction_percent']) for r in rr])
                y = np.array([float(r['depth_m']) for r in rr])
                assert np.all((x >= bounds[0]) & (x <= bounds[1]))
                ax.plot(x, y, color=source.COLORS[method], ls=source.STYLES[method], lw=2.35)
            ax.axvline(0, color=source.COLORS['deterministic'],
                       ls=source.STYLES['deterministic'], lw=1.45, zorder=0)
            ax.set(title=title, xlim=bounds, ylim=(520, -12),
                   yticks=(0, 100, 200, 300, 400, 500), xlabel='RMSE reduction (%)')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if index % 2 == 0:
                ax.set_ylabel('Depth (m)')
            ax.grid(color='#e4e4e4', lw=.55)
            ax.text(-.14, 1.03, 'abcd'[index], transform=ax.transAxes,
                    fontsize=12.5, fontweight='bold')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            assert all(line.get_marker() in ('None', '', None) for line in ax.lines)
            axes_record[variable] = dict(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                linear=True, all_native_values_visible=True)
        assert not fig.texts, 'Keep provenance in the unchanged manuscript caption and manifest.'
        fig.savefig(root / 'Figure4.pdf', dpi=300, metadata={
            'Title': 'Depthwise RMSE reduction from author-revised summary',
            'Subject': 'Typography-only revision; daily-score reconciliation remains pending in the caption.',
            'CreationDate': None, 'ModDate': None})
        fig.savefig(root / 'Figure4.png', dpi=300)
        plt.close(fig)
    return axes_record
