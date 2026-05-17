from wavelike import *
from wavelike.preprocess import *
from wavelike.config import *
from wavelike.plot import *
from wavelike.fit import WaveformFitter
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=50833)
parser.add_argument("--index", type=int, default=0)
args = parser.parse_args()

# Data path
data_path = f'/JNE/JinpingData/Jinping_1ton_Data/01_RawData/60PMTLS/Phy/run00050833/Jinping_1ton_Phy_20251126_00050833.root'

# Parameter path
dn_csv = 'data/1tdata/LSW_dn_df.csv'
gain_csv = 'data/1tdata/LSW_gain_df.csv'
time_csv = 'data/1tdata/Water_time_df.csv'

# Charge model configuration (choose one)
ChargeModel = 'GammaTWeedie'  # 'MultiGaussian', 'GammaTWeedie', or 'DataHist'
charge_file = 'data/1tdata/LSW_gt_df.csv'  # For MultiGaussian or GammaTWeedie or Rootfilename

# Plot dir
original_plot_dir = 'plots/original_plots/'
ser_plot_dir = 'plots/ser_plots/'
sub_plot_dir = 'plots/sub_plots/'
deconv_plot_dir = 'plots/deconv_plots/'
prior_comparison_plot_dir = 'plots/prior_comparison_plots/'
fit_plot_dir = 'plots/fit_plots/'
plot_fmt = 'png'

# Output dir
output_dir = f'output'


@timer
def main():
    total_start_time = time.time()
    print("--- Step 1: Load PMT Parameters and Precompute Templates ---")
    pmt_params = load_all_pmt_params(dn_csv, gain_csv, time_csv, template_range, ChargeModel, charge_file)
    print(f"Loaded parameters for {len(pmt_params)} PMTs using {ChargeModel} model.")
    print(f"PMT IDs: {list(pmt_params.keys())}")

    print("--- Step 2: Initialize Data Reader and Data Writer ---")
    data_reader = DataReader(data_path, allowed_pmts=set(pmt_params.keys()))
    data_writer = DataWriter(f'{output_dir}/run50833.root')

    print("--- Step 3: Starting main analysis loop ---")
    for trigger_no, event_dict in data_reader.get_event_generator():
        if trigger_no > 10:
            continue;
        print(f"Processing a event of {len(event_dict)} waveforms.")
        print("event_dict.keys():", event_dict.keys())
        for ch_id, waveform in event_dict.items():
            # if ch_id != 10:
            #     continue;
            print(f"Processing PMT ID: {ch_id}")
            # print(pmt_params[ch_id])
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
            if not fitter.ser_valid:
                print(f"Warning: Fitter invalid for channel {ch_id}, skipping")
                continue
            n_center = int(round(cpe))
            n_min = max(1, n_center - 3)
            n_max = n_center + 3

            best_nll = float('inf')
            best_params = None
            best_n = 0

            print(f"Fitting range: {n_min} to {n_max} PEs (CPE={cpe:.2f})")

            nll_history = []
            n_history = []
            stop_search = False
            for offset in range(0, max(n_center - n_min, n_max - n_center) + 1):
                if stop_search:
                    break

                for n in [n_center + offset, n_center - offset]:
                    if n < n_min or n > n_max or n in n_history:
                        continue

                    n_history.append(n)
                    if n > len(pe_prior):
                        extra_count = n - len(pe_prior)
                        if len(pe_prior) > 0:
                            peak_idx = np.argmax(pe_prior[:, 1])
                            base_time = pe_prior[peak_idx, 0]
                            offsets = np.concatenate([np.arange(1, extra_count // 2 + 1),
                                                    -np.arange(1, (extra_count + 1) // 2 + 1)])[:extra_count]
                            extra_times = base_time + offsets * 1.0
                        else:
                            base_time = len(waveform) // 2
                            extra_times = base_time + np.arange(extra_count) * 5.0

                        extra_pe = np.zeros((extra_count, 2))
                        extra_pe[:, 0] = np.clip(extra_times, 0, len(waveform) - 1)
                        extra_pe[:, 1] = 1.0
                        current_prior = np.vstack([pe_prior, extra_pe]) if len(pe_prior) > 0 else extra_pe
                    else:
                        current_prior = pe_prior[:n] if n > 0 else np.array([])

                    if len(current_prior) == 0 and n > 0:
                        print(f"  n={n}: No prior PE, skipping")
                        continue

                    nll, params, valid = fitter.fit(n, current_prior)
                    print(f"  n={n}: NLL={nll:.2f}, Valid={valid}")

                    if valid and not np.isinf(nll):
                        nll_history.append((n, nll))

                        if nll < best_nll:
                            best_nll = nll
                            best_params = params
                            best_n = n

                        # 检查是否已经过了极值点
                        if len(nll_history) >= 3:
                            # 获取最近的三个点（按 n 排序）
                            recent = sorted(nll_history[-3:], key=lambda x: x[0])
                            n1, nll1 = recent[0]
                            n2, nll2 = recent[1]
                            n3, nll3 = recent[2]

                            # 如果中间的点是最小值，且已经向两边扩展，则停止
                            if n2 == n_center + offset or n2 == n_center - offset:
                                if nll2 < nll1 and nll2 < nll3:
                                    print(f"  Found local minimum at n={n2}, stopping search")
                                    stop_search = True
                                    break
                    else:
                        # 如果遇到无效拟合，也停止继续搜索
                        print(f"  Invalid fit at n={n}, stopping search")
                        break
                else:
                    continue
                break

            print(f"Best fit: n={best_n}, NLL={best_nll:.2f}")

            if best_params is not None:
                amps = best_params[0::2]
                times = best_params[1::2]
                # print("  Amplitudes:", amps)
                # print("  Times:", times)
                data_writer.write_event(trigger_no, ch_id, amps, times)

                # plot_fit_result(waveform, amps, gain, times, pmt_params[ch_id],ch_id, trigger_no, args.run, fit_plot_dir, plot_fmt)
            else:
                print("  No valid fit found.")

    print("--- Step 4: Finalizing Data Writer ---")
    data_writer.close()
    total_end_time = time.time()  # 添加开始
    print(f"=== Total execution time: {total_end_time - total_start_time:.2f} seconds ===")

if __name__ == '__main__':
    main()
