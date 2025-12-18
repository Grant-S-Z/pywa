from wavelike import *
from wavelike.preprocess import *
from wavelike.config import *
from wavelike.plot import *

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
# deconv_plot_dir = 'plots/deconv_plots/'
# deconv_comparison_plot_dir = 'plots/deconv_comparison_plots/'
prior_comparison_plot_dir = 'plots/prior_comparison_plots/'
plot_fmt = 'pdf'
device_type = 'cuda'

if __name__ == '__main__':
    print("--- Step 1: Load PMT Parameters and Precompute Templates ---")
    pmt_params = load_all_pmt_params(dn_csv, gain_csv, gauss_csv, time_csv,
                                          template_range, gauss_no)
    print(f"Loaded parameters for {len(pmt_params)} PMTs.")
    print(f"PMT IDs: {list(pmt_params.keys())}")
    
    print("--- Step 2: Initialize Data Reader ---")
    data_reader = DataReader(data_path, allowed_pmts=set(pmt_params.keys()))

    print("--- Step 3: Starting main analysis loop ---")
    for ids_batch, waveform_batch in data_reader.get_batch_generator(batch_size=1, max_batches=1):
        print(f"Processing a batch of {len(waveform_batch)} triggers")
        print("ids_batch.shape:", ids_batch.shape)
        print("waveform_batch.shape:", waveform_batch.shape)
        print("ids_batch example:", ids_batch[0:5])
        waveform_batch, waveform_norm_batch, deconv_batch, pe_prior_batch, cpe_batch, noise_batch = preprocess_waveform_batch(waveform_batch, ids_batch, pmt_params, device=device_type)
        print("PE prior batch shape:", pe_prior_batch.shape)
        print("PE prior batch example:", pe_prior_batch[0, 0:5])
        print("Preprocessing completed.")

        # for trigger_no, channel_id in ids_batch:
        #     if trigger_no == 1:
        #         pmt_param = pmt_params.get(channel_id)
        #         if pmt_param is None:
        #             print(f"Warning: PMT ID {channel_id} parameters not found. Skipping plot.")
        #             continue
        #         print(f"Plotting preprocessed results for trigger {trigger_no}, PMT {channel_id}")
        #         waveform = waveform_norm_batch[ (ids_batch[:,0] == trigger_no) & (ids_batch[:,1] == channel_id) ][0].cpu().numpy()
        #         deconv_waveform = deconv_batch[ (ids_batch[:,0] == trigger_no) & (ids_batch[:,1] == channel_id) ][0].cpu().numpy()
        #         pe_prior = pe_prior_batch[ (ids_batch[:,0] == trigger_no) & (ids_batch[:,1] == channel_id) ][0].cpu().numpy()
        #         cpe = cpe_batch[ (ids_batch[:,0] == trigger_no) & (ids_batch[:,1] == channel_id) ][0].cpu().numpy()
        #         # plot_waveform(waveform, channel_id, 0, sub_plot_dir, plot_fmt)
        #         # plot_waveform(deconv_waveform, channel_id, trigger_no, deconv_plot_dir, plot_fmt)
        #         plot_prior_comparison(waveform, deconv_waveform, pe_prior, cpe, channel_id, trigger_no, prior_comparison_plot_dir, plot_fmt)
        #     continue

        
        
