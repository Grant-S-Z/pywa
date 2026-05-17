#!/usr/bin/env python3
"""Plot cpe vs ppe for all channels across events."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wavelike.pmtparam import load_all_pmt_params
from wavelike.preprocess import lucyddm, prior, get_pedestal, get_charge, ser_length, window_size
from wavelike.io import DataReader


def main(ser_path, gain_path, data_path, max_batches=20, plot_path=None):
    params = load_all_pmt_params(ser_path, gain_path, charge_model='DH')

    reader = DataReader(data_path, allowed_pmts=set(params.keys()))
    gen = reader.get_event_generator(batch_size=1, max_batches=max_batches)

    cpes, ppes = [], []
    for _, evt in gen:
        for ch_id, wf in evt.items():
            pmt = params[ch_id]
            gain = pmt.gm
            ser = pmt.ser.numpy() / np.sum(pmt.ser.numpy())
            ped = get_pedestal(wf)
            charge = get_charge(wf, ped)
            cpe = charge / gain
            wf_norm = (ped - wf) / gain
            S = ser_length
            d = lucyddm(wf_norm, ser)
            d = d[S - 1: S - 1 + window_size]
            ppe = prior(d).shape[0]
            if cpe >= 0:
                cpes.append(cpe)
                ppes.append(ppe)

    plt.figure(figsize=(8, 6))
    plt.scatter(cpes, ppes, alpha=0.5, edgecolors='none')
    mx = max(max(cpes), max(ppes)) * 1.05
    plt.plot([0, mx], [0, mx], 'k--', alpha=0.3, label='y=x')
    plt.xlabel('cpe (charge / gain)')
    plt.ylabel('ppe (deconvolution peaks)')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.title(f'cpe vs ppe ({len(cpes)} waveforms, {max_batches} events)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f'Saved {plot_path} ({len(cpes)} points)')


if __name__ == '__main__':
    ROOT = os.path.join(os.path.dirname(__file__), '..', '..')  # up to reconstruction/
    main(
        ser_path=os.path.join(ROOT, 'data/ser_ls_liuling/cut_preAna_LSW_ser.csv'),
        gain_path=os.path.join(ROOT, 'data/ser_ls_liuling/cut_preAna_Gain3_49796-51270.root'),
        data_path=os.path.join(ROOT, 'data/raw/run00050833/Jinping_1ton_Phy_20251126_00050833.root'),
        plot_path=os.path.join(os.path.dirname(__file__), '..', 'plots/script_plots/cpe_vs_ppe.pdf'),
    )
