import torch
import numpy as np
import pandas as pd
import uproot as ur
from dataclasses import dataclass, field
from typing import Dict, Optional
from scipy.stats import gaussian_kde
from .physics import ser_waveform


@dataclass
class PMTParam:
    channel_id: int
    # SPE parameters
    amplitude: float # A0
    decay_time: float # tau_PE
    sigma: float # sigma_PE
    gain: float # mean
    gm: float # gain mode
    # MultiGaussian
    gauss_no: int
    ratio: np.ndarray
    mean: np.ndarray
    sig2: np.ndarray
    # Template
    ser: torch.Tensor = field(init=False)
    ser_length: int = 50
    ser_template: Optional[np.ndarray] = None
    # Gamma Tweedie
    # frac: float = 0.5
    # mu_gm: float = 1.0
    # sig_gm: float = 1.0
    # lam_td: float = 1.0
    # mu_td: float = 1.0
    # sig_td: float = 1.0
    # DataHist: pre-computed KDE grid
    charge_grid: Optional[np.ndarray] = None
    charge_pdf: Optional[np.ndarray] = None
    # Charge model / prior
    charge_model: str = "DH"
    # Gain-scaled SER template for fast nll (deprecated: linear interp inaccurate for non-integer times)
    # ser_gain: Optional[np.ndarray] = None
    # ser_peak_offset: int = 0


    def __post_init__(self):
        print(f'Initializing PMT {self.channel_id}: Building template...')

        if self.ser_template is not None:
            tpl = np.asarray(self.ser_template, dtype=np.float32)
            if tpl.size == 0:
                self.ser = torch.zeros(self.ser_length, dtype=torch.float32)
            else:
                t = torch.from_numpy(tpl)
                s = torch.sum(t)
                self.ser = t / s if s > 0 else t
                self.ser_length = int(self.ser.shape[0])
            # ser_np = self.ser.numpy()
            # self.ser_gain = ser_np / np.sum(ser_np) * self.gm
            return

        center = self.ser_length
        x = torch.arange(2 * self.ser_length + 1) - center

        template_tensor = ser_waveform(x,
                                       self.amplitude,
                                       self.decay_time,
                                       self.sigma,
                                       0.0)
        threshold = 0.01 * torch.max(template_tensor) # 0.001 51s
        mask = template_tensor > threshold
        true_indices = torch.where(mask)[0]
        if len(true_indices) == 0:
            self.ser = torch.zeros(self.ser_length, dtype=torch.float32)
            return
        start = int(true_indices[0])
        end = min(start + self.ser_length, template_tensor.shape[0])
        cut = template_tensor[start:end]
        if cut.shape[0] < self.ser_length:
            pad = torch.zeros(self.ser_length - cut.shape[0], dtype=cut.dtype)
            cut = torch.cat([cut, pad], dim=0)
        self.ser = cut

        # self.ser_peak_offset = center - start
        # # Build gain-scaled template directly (same formula as ser_waveform_jit)
        # template_gain = ser_waveform_numpy(x.numpy(), self.gm, self.decay_time, self.sigma, 0.0)
        # cut_gain = template_gain[start:end]
        # if cut_gain.shape[0] < self.ser_length:
        #     cut_gain = np.pad(cut_gain, (0, self.ser_length - cut_gain.shape[0]))
        # self.ser_gain = cut_gain.astype(np.float64)


def load_all_pmt_params(ser_path: str, gain_path: str, charge_model: str,
                        ser_length: int = 50) -> Dict[int, PMTParam]:
    """Load PMT parameters.

    Parameters
    ----------
    ser_path : str
        CSV with SER fit params (channel_id, amplitude, decay_time, sigma).
    gain_path : str
        For 'DH': ROOT file with TTree 'Gain' containing ch{id} branches
        of charge samples. For 'MG'/'GT': CSV with gain shape params.
    charge_model : str
        'DH' (DataHist), 'MG' (MultiGaussian), or 'GT' (GammaTweedie).
    """
    if charge_model == 'DH':
        return _load_pmt_params_dh(ser_path, gain_path, ser_length)
    elif charge_model == 'MG':
        raise NotImplementedError('MG loading not yet implemented')
    elif charge_model == 'GT':
        raise NotImplementedError('GT loading not yet implemented')
    else:
        raise ValueError(f"Unknown charge model: {charge_model}")


def _load_pmt_params_dh(ser_csv: str, root_path: str,
                        ser_length: int) -> Dict[int, PMTParam]:
    """Load PMT params for DataHist gain shape.

    SER from CSV, charge samples from ROOT TTree.
    """
    all_params: Dict[int, PMTParam] = {}

    ser_df = pd.read_csv(ser_csv)

    with ur.open(root_path) as f:
        tree = f['Gain']
        arrays = tree.arrays(library='np')

    for _, row in ser_df.iterrows():
        cid = int(row['channel_id'])
        branch_name = f'ch{cid}'

        if branch_name not in arrays:
            print(f"[WARN] ch{cid} not found in ROOT file, skipping")
            continue

        raw = np.asarray(arrays[branch_name][0], dtype=np.float64)
        positive = raw[np.isfinite(raw) & (raw > 0)]

        if len(positive) == 0:
            print(f"[WARN] ch{cid}: no positive charge samples, skipping")
            continue

        gain = float(np.mean(positive))

        kde = gaussian_kde(positive, bw_method=0.05) # small bw to keep consistency with the hist
        lo, hi = np.min(positive), np.max(positive)
        pad = 0.1 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 2000)
        pdf = np.maximum(kde(grid), 1e-300)
        gm = float(grid[np.argmax(pdf)])

        all_params[cid] = PMTParam(
            channel_id=cid,
            amplitude=float(row['amplitude']),
            decay_time=float(row['decay_time']),
            sigma=float(row['sigma']),
            gain=gain,
            gm=gm,
            gauss_no=0,
            ratio=np.array([], dtype=np.float64),
            mean=np.array([], dtype=np.float64),
            sig2=np.array([], dtype=np.float64),
            ser_length=ser_length,
            charge_model='DH',
            charge_grid=grid,
            charge_pdf=pdf,
        )

    return all_params

def load_all_pmt_params_gaussian(dn_csv: str, gain_csv: str, gauss_csv: str, time_csv: str,
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
            gm=row['gain'],
            gauss_no=gauss_no,
            ratio=current_gauss_params['ratio'].to_numpy(),
            mean=current_gauss_params['mean'].to_numpy(),
            sig2=current_gauss_params['sig2'].to_numpy(),
            ser_length=ser_length,
            charge_model="MultiGaussian"
        )

        all_params[ichannel] = pmt_param

    return all_params

