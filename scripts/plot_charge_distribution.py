#!/usr/bin/env python3
"""Plot charge distribution for a specified channel from ROOT file."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import uproot as ur
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_charge_distribution(root_path: str, channel_id: int,
                              bins: int = 200, save_path: str = None):
    """Plot charge distribution for a specific channel.

    Parameters
    ----------
    root_path : str
        Path to ROOT file with TTree 'Gain' containing ch{id} branches.
    channel_id : int
        Channel ID to plot.
    bins : int
        Number of histogram bins.
    save_path : str, optional
        Path to save the figure. Default: charge_ch{id}.pdf
    """
    with ur.open(root_path) as f:
        tree = f['Gain']
        branch_name = f'ch{channel_id}'
        if branch_name not in tree:
            raise KeyError(f'{branch_name} not found in {root_path}')
        arr = tree[branch_name].array(library='np')

    raw = np.asarray(arr[0], dtype=np.float64)
    positive = raw[np.isfinite(raw) & (raw > 0)]

    from scipy.stats import gaussian_kde

    mean = np.mean(positive)
    median = np.median(positive)
    mode = _find_mode(positive)

    # KDE
    lo, hi = np.min(positive), np.max(positive)
    pad = 0.1 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 1000)
    kde = gaussian_kde(positive, bw_method=0.05)
    kde_vals = kde(grid)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(positive, bins=bins, density=True, alpha=0.6, color='steelblue',
            edgecolor='white', linewidth=0.5, label=f'n={len(positive)}')
    ax.plot(grid, kde_vals, 'k-', linewidth=1.5, alpha=0.8, label='gaussian KDE')

    ax.axvline(mean, color='C3', linestyle='--', linewidth=2,
               label=f'mean={mean:.1f}')
    ax.axvline(median, color='C2', linestyle='--', linewidth=2,
               label=f'median={median:.1f}')
    ax.axvline(mode, color='C4', linestyle='--', linewidth=2,
               label=f'mode={mode:.1f}')
    ax.set_xlim(0, 800)

    ax.set_xlabel('Charge')
    ax.set_ylabel('Density')
    ax.set_title(f'Charge distribution — ch{channel_id}')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save = save_path or f'charge_ch{channel_id}.pdf'
    fig.savefig(save, bbox_inches='tight')
    print(f'Saved {save}')
    plt.close(fig)


def _find_mode(positive: np.ndarray, bins: int = 400) -> float:
    """Estimate mode via histogram peak."""
    counts, edges = np.histogram(positive, bins=bins)
    i = np.argmax(counts)
    return float(0.5 * (edges[i] + edges[i + 1]))


if __name__ == '__main__':
    ROOT = os.path.join(os.path.dirname(__file__), '..')
    channel_id = 28
    plot_charge_distribution(
        os.path.join(ROOT, '../data/ser_ls_liuling/cut_preAna_Gain3_49796-51270.root'),
        channel_id,
        save_path=os.path.join(ROOT, f'plots/script_plots/charge_distribution_ch{channel_id}.pdf'),
    )
