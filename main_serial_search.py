from wavelike import *
from wavelike.preprocess import *
from wavelike.config import *
from wavelike.plot import *
from wavelike.fit import WaveformFitter
import numpy as np

# Data path
data_path = 'data/run00045887/Jinping_1ton_Phy_20250302_00045887.root'

# Parameter path
dn_csv = 'data/DN_fit_results_simple.csv'
gain_csv = 'data/LSGainList_fsmp.csv'
gauss_csv = 'data/1.csv'
time_csv = 'data/TimeCalib.csv'

# Plot dir
original_plot_dir = 'plots/original_plots/'
ser_plot_dir = 'plots/ser_plots/'
sub_plot_dir = 'plots/sub_plots/'
deconv_plot_dir = 'plots/deconv_plots/'
prior_comparison_plot_dir = 'plots/prior_comparison_plots/'
fit_plot_dir = 'plots/fit_plots/'
plot_fmt = 'pdf'

# Output dir
output_dir = 'output/'


@timer
def main():
    print("--- Step 1: Load PMT Parameters and Precompute Templates ---")
    pmt_params = load_all_pmt_params(dn_csv, gain_csv, gauss_csv, time_csv,
                                     template_range, gauss_no)
    print(f"Loaded parameters for {len(pmt_params)} PMTs.")
    print(f"PMT IDs: {list(pmt_params.keys())}")

    print("--- Step 2: Initialize Data Reader and Data Writer ---")
    data_reader = DataReader(data_path, allowed_pmts=set(pmt_params.keys()))
    data_writer = DataWriter(f'{output_dir}/test_serial.root')

    print("--- Step 3: Starting main analysis loop ---")
    for event_dict in data_reader.get_event_generator(batch_size=1, max_batches=1):
        print(f"Processing a event of {len(event_dict)} waveforms.")
        print("event_dict.keys():", event_dict.keys())
        for ch_id, waveform in event_dict.items():
            print(f"Processing PMT ID: {ch_id}")
            gain = pmt_params[ch_id].gain
            ser = pmt_params[ch_id].ser.numpy()
            ser /= np.sum(ser)  # ser normalization
            waveform, waveform_norm, deconv, pe_prior, cpe, noise = preprocess_waveform(
                waveform, ser, gain)
            # print("PE prior shape:", pe_prior.shape)
            # print("PE prior example:\n", pe_prior[0:3])
            print(
                f"Preprocessing completed, PPE={np.count_nonzero(np.any(pe_prior != 0, axis=1))}")
            # plot_waveform(deconv, ch_id, 0, deconv_plot_dir, plot_fmt)

            # Fitting
            fitter = WaveformFitter(waveform, pmt_params[ch_id], noise)

            n_center = int(round(cpe))
            n_min = max(1, n_center - 1)
            n_max = n_center + 1

            best_nll = float('inf')
            best_params = None
            best_n = 0

            print(f"Fitting range: {n_min} to {n_max} PEs (CPE={cpe:.2f})")

            for n in range(n_min, n_max + 1):
                if n > len(pe_prior):
                    print(f"  Skipping n={n} as it exceeds prior length {len(pe_prior)}")
                    continue

                # Initial guess from pe_prior (top n)
                current_prior = pe_prior[:n]

                nll, params, valid = fitter.fit(n, current_prior)
                print(f"  n={n}: NLL={nll:.2f}, Valid={valid}")

                if valid and nll < best_nll:
                    best_nll = nll
                    best_params = params
                    best_n = n

            print(f"Best fit: n={best_n}, NLL={best_nll:.2f}")
            if best_params is not None:
                amps = best_params[0::2]
                times = best_params[1::2]
                print("  Amplitudes:", amps)
                print("  Times:", times)
                data_writer.write_event(0, ch_id, amps, times)

                plot_fit_result(waveform, amps, times, pmt_params[ch_id],
                                ch_id, 0, fit_plot_dir, plot_fmt)
            else:
                print("  No valid fit found.")

    print("--- Step 4: Finalizing Data Writer ---")
    data_writer.close()


if __name__ == '__main__':
    main()
