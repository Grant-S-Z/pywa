#!/usr/bin/env python3
import sys
import argparse

import numpy as np

from wavelike import *
from wavelike.config import *
from wavelike.fit import WaveformFitter
# from wavelike.physics import ser_waveform_jit
from wavelike.plot import *
# from wavelike.preprocess import (
#     find_missed_peaks,
#     remove_coincident_peaks,
#     preprocess_waveform,
# )
from wavelike.preprocess import *
from wavelike.pmtparam import load_all_pmt_params


# Log
log_file = open('log/main_serial.log', 'w', buffering=1)
sys.stdout = log_file
sys.stderr = log_file


# Plot dir
prior_plot_dir = 'plots/prior_plots/'
fit_plot_dir = 'plots/fit_plots/'
plot_fmt = 'pdf'

# Output dir
output_dir = 'output'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run pywa serial fit with raw + ser only')
    p.add_argument('--data-path', default='../data/raw/run00050833/Jinping_1ton_Phy_20251126_00050833.root', help='Input raw ROOT waveform file path')
    p.add_argument('--ser-path', default='../data/ser_ls_liuling/cut_preAna_LSW_ser.csv', help='Input ser fit parameter file path')
    p.add_argument('--charge-model', default='DH', help='Input gain calibration shape (MG, GT or DH)')
    p.add_argument('--gain-path', default='../data/ser_ls_liuling/cut_preAna_Gain3_49796-51270.root', help='Input gain hist file path')
    p.add_argument('--max-batches', type=int, default=1, help='Max batches to process (default: 1)')
    p.add_argument('--batch-size', type=int, default=1, help='Events per batch for DataReader (default: 1)')
    p.add_argument('--min-charge-samples', type=int, default=100,
                   help='Skip channels with fewer positive charge samples (default: 100)')
    return p.parse_args()


@timer
def main():
    args = parse_args()

    print('--- Step 1: Load PMT Parameters ---')
    pmt_params = load_all_pmt_params(args.ser_path, args.gain_path, args.charge_model)
    if not pmt_params:
        raise RuntimeError('No PMT parameters loaded.')
    print(f"Loaded parameters for {len(pmt_params)} PMTs from {args.ser_path} and corresponding gain hist from {args.gain_path}.")
    print(f"PMT IDs: {list(pmt_params.keys())}")

    print('--- Step 2: Initialize Data Reader and Data Writer ---')
    data_reader = DataReader(args.data_path, allowed_pmts=set(pmt_params.keys()))
    data_writer = DataWriter(f'{output_dir}/test_serial.root')

    print('--- Step 3: Starting main analysis loop ---')
    for trigger_no, event_dict in data_reader.get_event_generator(batch_size=args.batch_size, max_batches=args.max_batches):
        print(f"Processing an event of {len(event_dict)} waveforms.")
        print('event_dict.keys():', event_dict.keys())
        for ch_id, waveform in event_dict.items():
            print(f"Processing PMT ID: {ch_id}")
            gain = pmt_params[ch_id].gain # mean
            gm = pmt_params[ch_id].gm # mode
            ser = pmt_params[ch_id].ser.numpy()
            ser /= np.sum(ser)
            waveform, _, deconv, pe_prior, cpe, noise = preprocess_waveform(waveform, ser, gain)
            ppe = len(pe_prior)

            plot_prior_comparison(waveform, deconv, pe_prior, cpe, ch_id, trigger_no, gain, prior_plot_dir)
            print(' Prior PE:\n', pe_prior)

            # # Build the analytical SER template for model construction
            # # (same formula used by _nll_core_dh)
            # x_axis = np.arange(len(waveform), dtype=np.float64)
            # tau = pmt_params[ch_id].decay_time
            # sig = pmt_params[ch_id].sigma

            print(f"Preprocessing completed, PPE={ppe}, start fitting...")
            fitter = WaveformFitter(waveform, pmt_params[ch_id], noise)
            current_prior = pe_prior

            nll, params, valid = fitter.fit(ppe, current_prior)
            print(f"  n={ppe}: NLL={nll:.2f}, Valid={valid}")

            amps = params[0::2]
            times = params[1::2]
            print('  Amplitudes:', amps)
            print('  Times:', times)
            data_writer.write_event(trigger_no, ch_id, amps, times)

            plot_fit_result(waveform, amps, times, pmt_params[ch_id],
                            ch_id, trigger_no, gm, fit_plot_dir, plot_fmt)

    print('--- Step 4: Finalizing Data Writer ---')
    data_writer.close()


if __name__ == '__main__':
    main()
