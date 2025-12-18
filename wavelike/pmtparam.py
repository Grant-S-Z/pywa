import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict
from .physics import ser_waveform

@dataclass
class PMTParam:
    channel_id: int
    # SPE parameters
    amplitude: float
    decay_time: float
    sigma: float
    # Gain parameters
    gain: float
    gauss_no: int
    ratio: np.ndarray
    mean: np.ndarray
    sig2: np.ndarray
    # Time
    time_offset: float
    # Template
    ser: torch.Tensor = field(init=False)
    ser_length: int

    def __post_init__(self):
        print(f'Initializing PMT {self.channel_id}: Building template...')
        center = self.ser_length
        x = torch.arange(2 * self.ser_length + 1) - center

        template_tensor = ser_waveform(x,
                                    self.amplitude,
                                    self.decay_time,
                                    self.sigma,
                                    0.0
                                    )
        threshold = 0.01 * torch.max(template_tensor)
        mask = template_tensor > threshold
        true_indices = torch.where(mask)[0]
        self.ser = template_tensor[true_indices[0]: true_indices[0] + self.ser_length]
        

def load_all_pmt_params(dn_csv: str, gain_csv: str, gauss_csv: str, time_csv: str,
                        ser_length: int, gauss_no: int) -> Dict[int, PMTParam]:
    all_params: Dict[int, PMTParam] = {}

    try:
        dn_cols = ['channel_id', 'amplitude', 'decay_time', 'sigma']
        gain_cols = ['channel_id', 'gain']
        gauss_cols = ['channel_id', 'ratio', 'mean', 'sig2']
        time_cols = ['channel_id', 'time_offset']

        dn_df = pd.read_csv(dn_csv, skiprows=1, names=dn_cols)
        gain_df = pd.read_csv(gain_csv, skiprows=1, names=gain_cols)
        gauss_df = pd.read_csv(gauss_csv, skiprows=1, names=gauss_cols)
        time_df = pd.read_csv(time_csv, skiprows=1, names=time_cols)

    except FileNotFoundError as e:
        print(f"Error: One of the CSV files was not found. {e}")
        return {}
    except Exception as e:
        print(f"An error occurred while reading CSV files: {e}")
        return {}

    merged_df = pd.merge(dn_df, gain_df, on='channel_id')
    merged_df = pd.merge(merged_df, time_df, on='channel_id')

    for _, row in merged_df.iterrows():
        ichannel = int(row['channel_id'])

        current_gauss_params = gauss_df[gauss_df['channel_id'] == ichannel]

        pmt_param = PMTParam(
            channel_id=ichannel,
            amplitude=row['amplitude'],
            decay_time=row['decay_time'],
            sigma=row['sigma'],
            gain=row['gain'],
            gauss_no=gauss_no,
            ratio=current_gauss_params['ratio'].to_numpy(),
            mean=current_gauss_params['mean'].to_numpy(),
            sig2=current_gauss_params['sig2'].to_numpy(),
            time_offset=row['time_offset'],
            ser_length=ser_length
        )

        all_params[ichannel] = pmt_param

    return all_params
