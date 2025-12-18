from wavelike import *
from wavelike.preprocess import *
from wavelike.config import *
from wavelike.plot import *
from wavelike.fit import WaveformFitter
import numpy as np
import sys

# Log
log_file = open('log/main_serial.log', 'w', buffering=1)
sys.stdout = log_file
sys.stderr = log_file

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
output_dir = 'output'


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
    # 看一下 charge/ppe 的增益谱
    print("--- Step 3: Starting main analysis loop ---")
    for trigger_no, event_dict in data_reader.get_event_generator(batch_size=1, max_batches=1):
        print(f"Processing a event of {len(event_dict)} waveforms.")
        print("event_dict.keys():", event_dict.keys())
        for ch_id, waveform in event_dict.items():
            print(f"Processing PMT ID: {ch_id}")
            gain = pmt_params[ch_id].gain
            ser = pmt_params[ch_id].ser.numpy()
            ser /= np.sum(ser)  # ser normalization
            waveform, _, _, pe_prior, _, noise = preprocess_waveform(
                waveform, ser, gain)            
            ppe = len(pe_prior)  # pe_prior is now variable length
            print(" Prior PE:\n", pe_prior)
            
            print(f"Preprocessing completed, PPE={ppe}, start fitting...")
            fitter = WaveformFitter(waveform, pmt_params[ch_id], noise)
            current_prior = pe_prior

            nll, params, valid = fitter.fit(ppe, current_prior)
            print(f"  n={ppe}: NLL={nll:.2f}, Valid={valid}")

            amps = params[0::2]
            times = params[1::2]
            print("  Amplitudes:", amps)
            print("  Times:", times)
            data_writer.write_event(trigger_no, ch_id, amps, times)

            plot_fit_result(waveform, amps, times, pmt_params[ch_id],
                            ch_id, trigger_no, fit_plot_dir, plot_fmt)

    print("--- Step 4: Finalizing Data Writer ---")
    data_writer.close()


if __name__ == '__main__':
    main()
